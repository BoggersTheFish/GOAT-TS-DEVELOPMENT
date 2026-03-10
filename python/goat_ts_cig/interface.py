"""CLI for user input (seed concept or raw text/file) and output visualization (idea map, hypotheses)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goat_ts_cig.config import get_env_override, load_config, load_config_with_hardware
from goat_ts_cig.knowledge_graph import KnowledgeGraph
from goat_ts_cig.ts_engine import run_ts_propagation
from goat_ts_cig.cig_generator import generate_cig_output
from goat_ts_cig.hypothesis_engine import (
    suggest_hypotheses,
    suggest_hypotheses_from_tensions,
    id_to_label_from_graph_dict,
    format_hypotheses_text,
)

try:
    from chat_ingest import text_to_graph_dict
except ImportError:
    text_to_graph_dict = None


def ensure_seed_in_graph(graph: KnowledgeGraph, seed: str) -> int:
    """Ensure seed concept exists as a node; return its id."""
    import hashlib
    seed_id = int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16) % (10 ** 9)
    if graph.get_node(seed_id) is None:
        graph.add_node(seed_id, metadata={"label": seed, "name": seed}, influence=1.0)
    else:
        graph._nodes[seed_id]["influence"] = 1.0
    return seed_id


def run_cig(seed: str, config_path: str | None = None, output_json: bool = False) -> str | dict:
    """
    Full CIG workflow: load config, graph, inject seed, propagate, generate idea map and hypotheses.
    """
    config = load_config(config_path)
    for k, v in get_env_override().items():
        config.setdefault(k, {}).update(v)

    graph_cfg = config.get("graph", {})
    db_path = graph_cfg.get("path", "data/cig_graph.db")
    in_mem_max = graph_cfg.get("in_memory_max_nodes", 1000)
    use_memory = bool(in_mem_max and in_mem_max > 0)
    base = Path(__file__).resolve().parent.parent.parent
    graph = KnowledgeGraph(db_path=base / db_path, in_memory_only=use_memory)

    ensure_seed_in_graph(graph, seed)
    graph_dict = graph.to_dict()

    wave_cfg = config.get("wave", {})
    influences = run_ts_propagation(
        graph_dict,
        ticks=wave_cfg.get("ticks", 20),
        decay=wave_cfg.get("decay", 0.15),
        epsilon=wave_cfg.get("epsilon", 0.01),
        max_ticks=wave_cfg.get("max_ticks", 50),
    )
    graph.set_influences(influences)
    graph.persist()

    cig_out = generate_cig_output(graph, as_text=True)
    hypotheses = suggest_hypotheses(graph)
    cig_out["hypotheses"] = hypotheses
    cig_out["hypotheses_text"] = format_hypotheses_text(hypotheses)

    if output_json:
        return {
            "seed": seed,
            "idea_map": cig_out["idea_map"],
            "hypotheses": hypotheses,
            "idea_map_text": cig_out.get("idea_map_text", ""),
            "hypotheses_text": cig_out.get("hypotheses_text", ""),
        }
    return (cig_out.get("idea_map_text", "") + "\n\n" + cig_out.get("hypotheses_text", "")).strip()


def run_cig_from_text(
    text: str,
    config_path: str | None = None,
    output_json: bool = False,
    source_name: str = "input",
) -> str | dict[str, Any]:
    """
    Run CIG on raw text: ingest -> wave propagation (hardware-aware) -> tension hypotheses.
    Uses chat_ingest for extraction and suggest_hypotheses_from_tensions for hypotheses.
    """
    if text_to_graph_dict is None:
        raise RuntimeError("chat_ingest not available; add python/ to PYTHONPATH")
    config = load_config_with_hardware(config_path)
    hw = config.get("hardware", {})
    graph_dict = text_to_graph_dict(text)
    nodes = graph_dict.get("nodes", [])
    if not nodes:
        if output_json:
            return {"source": source_name, "idea_map": [], "hypotheses": [], "idea_map_text": "", "hypotheses_text": ""}
        return "No concepts extracted from text."

    max_ticks = hw.get("max_ticks", 15)
    influences = run_ts_propagation(
        graph_dict,
        ticks=max_ticks,
        decay=0.15,
        epsilon=0.01,
        max_ticks=max_ticks + 10,
    )
    id_to_label = id_to_label_from_graph_dict(graph_dict)
    hypotheses = suggest_hypotheses_from_tensions(influences, id_to_label, max_hypotheses=5)

    graph = KnowledgeGraph(in_memory_only=True)
    for n in nodes:
        graph.add_node(
            n["id"],
            node_type="Concept",
            metadata=n.get("metadata", {}),
            weight=n.get("weight", 1.0),
            influence=influences.get(n["id"], 0.0),
        )
    for e in graph_dict.get("edges", []):
        graph.add_edge(e["from"], e["to"], edge_type=e.get("type", "relates_to"), strength=e.get("strength", 1.0))

    cig_out = generate_cig_output(graph, as_text=True)
    cig_out["hypotheses"] = hypotheses
    cig_out["hypotheses_text"] = format_hypotheses_text(hypotheses)

    if output_json:
        return {
            "source": source_name,
            "idea_map": cig_out["idea_map"],
            "hypotheses": hypotheses,
            "idea_map_text": cig_out.get("idea_map_text", ""),
            "hypotheses_text": cig_out.get("hypotheses_text", ""),
        }
    return (cig_out.get("idea_map_text", "") + "\n\n" + cig_out.get("hypotheses_text", "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="GOAT-TS CIG: Contextual Information Generator")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--seed", "-s", help="Seed concept (e.g. 'AI Ethics')")
    input_group.add_argument("--text", "-t", help="Raw text to ingest and run waves + hypotheses")
    input_group.add_argument("--file", "-f", help="Path to text file to ingest")
    parser.add_argument("--config", "-c", default=None, help="Config file path (YAML/JSON)")
    parser.add_argument("--json", action="store_true", help="Output full JSON")
    args = parser.parse_args()

    if args.text:
        result = run_cig_from_text(args.text, config_path=args.config, output_json=args.json, source_name="--text")
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8", errors="replace")
        result = run_cig_from_text(text, config_path=args.config, output_json=args.json, source_name=path.name)
    else:
        result = run_cig(args.seed, config_path=args.config, output_json=args.json)

    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
