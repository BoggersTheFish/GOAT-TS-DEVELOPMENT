"""Helpers: data ingestion, serialization, and graph conversion for Rust interop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def graph_dict_to_rust_format(graph_dict: dict[str, Any]) -> tuple[list[tuple[int, float]], list[tuple[int, int, float]]]:
    """
    Convert Python graph representation to Rust-friendly format.
    Returns (node_list, edge_list): nodes as (id, influence), edges as (from, to, strength).
    """
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])

    if isinstance(nodes, dict):
        node_list = [(int(k), float(v.get("influence", 0.0))) for k, v in nodes.items()]
    else:
        node_list = [(int(n["id"]), float(n.get("influence", 0.0))) for n in nodes]

    edge_list = [
        (int(e["from"]), int(e["to"]), float(e.get("strength", 1.0)))
        for e in edges
    ]
    return node_list, edge_list


def rust_influences_to_dict(node_ids: list[int], influences: list[float]) -> dict[int, float]:
    """Map Rust output influence vector back to node id -> influence."""
    if len(node_ids) != len(influences):
        raise ValueError("node_ids and influences length mismatch")
    return dict(zip(node_ids, influences))


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    """Save JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
