"""
Generate instruction-tuning JSONL from raw texts: ingest -> wave propagation -> tension hypotheses.
Output: Alpaca-style {"prompt": "...", "completion": "..."} to data/finetune_dataset.jsonl.
Usage (from repo root): python python/generate_dataset.py --input examples/raw_texts
  or: python python/generate_dataset.py --input path/to/file.txt --num-examples 20 --format alpaca
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Path setup when run as script or module from repo root
if __name__ == "__main__" or "__main__" in sys.modules:
    _dir = Path(__file__).resolve().parent
    _root = _dir.parent
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from chat_ingest import text_to_graph_dict
from goat_ts_cig.config import load_config_with_hardware
from goat_ts_cig.hypothesis_engine import (
    id_to_label_from_graph_dict,
    suggest_hypotheses_from_tensions,
)
from goat_ts_cig.ts_engine import run_ts_propagation


def _chunks_from_input(input_path: str) -> list[tuple[str, str]]:
    """Return list of (source_name, text) from a file or directory."""
    p = Path(input_path)
    if not p.exists():
        return []
    chunks: list[tuple[str, str]] = []
    if p.is_file():
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            chunks.append((p.name, text))
    else:
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in (".txt", ".md", ""):
                text = f.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    chunks.append((f.name, text))
    return chunks


def _process_chunk(
    chunk_text: str,
    source_name: str,
    hw: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run ingest -> waves -> tension hypotheses; return list of {prompt, completion} examples."""
    examples: list[dict[str, Any]] = []
    graph_dict = text_to_graph_dict(chunk_text)
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])
    if not nodes:
        return examples

    max_ticks = hw.get("max_ticks", 15)
    influences = run_ts_propagation(
        graph_dict,
        ticks=max_ticks,
        decay=0.15,
        epsilon=0.01,
        max_ticks=max_ticks + 10,
    )
    id_to_label = id_to_label_from_graph_dict(graph_dict)
    hypotheses = suggest_hypotheses_from_tensions(
        influences, id_to_label, max_hypotheses=5
    )

    # Compact JSON for concepts/relations (nodes + edges only)
    extract_payload = {
        "nodes": [{"id": n["id"], "label": (n.get("metadata") or {}).get("label", n["id"])} for n in nodes],
        "edges": [{"from": e["from"], "to": e["to"], "type": e.get("type", "relates_to")} for e in edges],
    }
    prompt_extract = f"Given this text:\n{chunk_text[:2000]}\n\nExtract key concepts and relations."
    completion_extract = json.dumps(extract_payload, indent=0)

    examples.append({"prompt": prompt_extract, "completion": completion_extract})

    hypotheses_text = "\n".join(
        f"{i+1}. {h['suggestion']}" for i, h in enumerate(hypotheses)
    ) or "No tensions identified."
    prompt_hyps = f"Given this text:\n{chunk_text[:2000]}\n\nGenerate 3 insightful hypotheses."
    examples.append({"prompt": prompt_hyps, "completion": hypotheses_text})

    return examples


def _to_alpaca(example: dict[str, Any]) -> dict[str, Any]:
    """Convert {prompt, completion} to Alpaca {instruction, input, output}."""
    return {
        "instruction": example["prompt"],
        "input": "",
        "output": example["completion"],
    }


def run(
    input_path: str,
    output_path: str | Path = "data/finetune_dataset.jsonl",
    num_examples: int | None = None,
    format_alpaca: bool = False,
) -> int:
    """
    Generate JSONL from input (file or folder). Appends to output_path.
    Returns number of examples written.
    """
    config = load_config_with_hardware()
    hw = config.get("hardware", {})
    chunks = _chunks_from_input(input_path)
    if not chunks:
        print(f"No text chunks from: {input_path}", file=sys.stderr)
        return 0

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_name, text in chunks:
        if num_examples is not None and count >= num_examples:
            break
        examples = _process_chunk(text, source_name, hw)
        for ex in examples:
            if num_examples is not None and count >= num_examples:
                break
            record = _to_alpaca(ex) if format_alpaca else ex
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate finetune JSONL from raw texts (ingest + waves + hypotheses)."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input file or folder (e.g. examples/raw_texts)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/finetune_dataset.jsonl",
        help="Output JSONL path (default: data/finetune_dataset.jsonl)",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=None,
        help="Max number of examples to write (default: no limit)",
    )
    parser.add_argument(
        "--format",
        choices=["prompt_completion", "alpaca"],
        default="prompt_completion",
        help="prompt_completion = {prompt, completion}; alpaca = {instruction, input, output}",
    )
    args = parser.parse_args()

    n = run(
        input_path=args.input,
        output_path=args.output,
        num_examples=args.num_examples,
        format_alpaca=(args.format == "alpaca"),
    )
    print(f"Wrote {n} examples to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
