"""
Ingest raw text for data-gen: split into sentences, extract concept nodes (rule-based),
add relates_to edges for co-occurring words. 100% local, no external LLM.
Usage (from repo root GOAT-TS-SUPERLITE): python -m chat_ingest --text "..." or --file path.txt
Or: python python/chat_ingest.py --text "..."
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

# When run as __main__ from repo root, ensure python/ and root are on path
if __name__ == "__main__" or "__main__" in sys.modules:
    _dir = Path(__file__).resolve().parent
    _root = _dir.parent
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

# Minimal English stopwords (no nltk); keep small for low-end machines.
_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with by from as is was are were been be have has had do does did will would could should may might must can shall ought i you he she it we they this that these those what which who whom".split()
)


def _sentences(text: str) -> list[str]:
    """Split text into sentences (regex, no nltk)."""
    text = (text or "").strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _words(sentence: str, min_len: int = 2) -> list[str]:
    """Extract alphabetic tokens of length >= min_len, lowercased."""
    tokens = re.findall(r"[a-zA-Z]+", sentence)
    return [t.lower() for t in tokens if len(t) >= min_len and t.lower() not in _STOPWORDS]


def _node_id(label: str) -> int:
    """Stable integer id from concept label (for graph nodes). Deterministic across runs."""
    h = hashlib.sha256(label.strip().lower().encode()).hexdigest()
    return int(h[:12], 16) % (10**9)


def extract_nodes_and_edges(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Rule-based extraction: unique words (after stopwords) as nodes,
    co-occurrence in same sentence as relates_to edges (weight=1.0).
    Returns (nodes, edges) where each node is {id, type, metadata, weight, influence}
    and each edge is {from, to, type, strength}.
    """
    nodes_by_id: dict[int, dict[str, Any]] = {}
    edge_keys: set[tuple[int, int]] = set()
    edges_out: list[dict[str, Any]] = []

    for sentence in _sentences(text):
        words_in_sentence = _words(sentence)
        seen_this_sentence: set[int] = set()
        for w in words_in_sentence:
            nid = _node_id(w)
            if nid not in nodes_by_id:
                nodes_by_id[nid] = {
                    "id": nid,
                    "type": "Concept",
                    "metadata": {"label": w, "name": w},
                    "weight": 1.0,
                    "influence": 0.0,
                }
            seen_this_sentence.add(nid)
        ids_list = list(seen_this_sentence)
        for i, a in enumerate(ids_list):
            for b in ids_list[i + 1 :]:
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges_out.append({"from": key[0], "to": key[1], "type": "relates_to", "strength": 1.0})
                else:
                    # Optional: bump strength for repeated co-occurrence (here we keep 1.0 per pair)
                    pass

    return list(nodes_by_id.values()), edges_out


def text_to_graph_dict(text: str) -> dict[str, Any]:
    """Convert raw text to graph dict {nodes, edges} for run_ts_propagation / KnowledgeGraph."""
    nodes, edges = extract_nodes_and_edges(text)
    return {"nodes": nodes, "edges": edges}


def ingest_to_knowledge_graph(graph: Any, text: str) -> None:
    """Populate a KnowledgeGraph from raw text (adds nodes and relates_to edges)."""
    nodes, edges = extract_nodes_and_edges(text)
    for n in nodes:
        graph.add_node(
            n["id"],
            node_type="Concept",
            metadata=n.get("metadata", {}),
            weight=n.get("weight", 1.0),
            influence=n.get("influence", 0.0),
        )
    for e in edges:
        graph.add_edge(e["from"], e["to"], edge_type="relates_to", strength=e.get("strength", 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest text: extract concepts and co-occurrence edges.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Raw text string")
    g.add_argument("--file", type=str, help="Path to text file")
    parser.add_argument("--json", action="store_true", help="Output graph as JSON")
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8", errors="replace")

    graph_dict = text_to_graph_dict(text)
    if args.json:
        import json
        print(json.dumps(graph_dict, indent=2))
    else:
        n, e = graph_dict["nodes"], graph_dict["edges"]
        print(f"Nodes: {len(n)}, Edges: {len(e)}")
        for node in n[:15]:
            label = (node.get("metadata") or {}).get("label", node["id"])
            print(f"  {node['id']} {label}")
        if len(n) > 15:
            print("  ...")
        for edge in e[:10]:
            print(f"  {edge['from']} --relates_to--> {edge['to']}")
        if len(e) > 10:
            print("  ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
