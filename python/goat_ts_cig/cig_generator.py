"""CIG output generation: idea maps, hypothesis chains from converged node states."""

from __future__ import annotations

from typing import Any

from goat_ts_cig.knowledge_graph import KnowledgeGraph


def build_idea_map(graph: KnowledgeGraph, top_k: int = 15) -> list[dict[str, Any]]:
    """
    Build an idea map from graph: nodes ordered by influence with their strongest edges.
    """
    nodes = sorted(
        graph._nodes.values(),
        key=lambda n: n.get("influence", 0.0),
        reverse=True,
    )[:top_k]
    node_ids = {n["id"] for n in nodes}
    edges_from = {}
    for e in graph._edges:
        if e["from"] in node_ids or e["to"] in node_ids:
            edges_from.setdefault(e["from"], []).append((e["to"], e["type"], e.get("strength", 1.0)))

    idea_map = []
    for n in nodes:
        nid = n["id"]
        meta = n.get("metadata", {})
        label = meta.get("label", meta.get("name", str(nid)))
        links = [
            {"to": to_id, "type": etype, "strength": s}
            for to_id, etype, s in sorted(edges_from.get(nid, []), key=lambda x: -x[2])[:5]
        ]
        idea_map.append({
            "node_id": nid,
            "type": n["type"],
            "influence": n.get("influence", 0.0),
            "label": label,
            "links": links,
        })
    return idea_map


def format_idea_map_text(idea_map: list[dict[str, Any]]) -> str:
    """Render idea map as human-readable text."""
    lines = ["=== Idea Map (by influence) ==="]
    for item in idea_map:
        lines.append(f"  [{item['node_id']}] {item['label']} (influence: {item['influence']:.3f}) [{item['type']}]")
        for link in item["links"]:
            lines.append(f"    -> {link['to']} ({link['type']}, strength {link['strength']:.2f})")
    return "\n".join(lines)


def generate_cig_output(
    graph: KnowledgeGraph,
    idea_map_top_k: int = 15,
    as_text: bool = True,
) -> dict[str, Any]:
    """Full CIG output: idea map and optional text representation."""
    idea_map = build_idea_map(graph, top_k=idea_map_top_k)
    out = {"idea_map": idea_map}
    if as_text:
        out["idea_map_text"] = format_idea_map_text(idea_map)
    return out
