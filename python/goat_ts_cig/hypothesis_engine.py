"""Hypothesis logic: interpret converged states into exploratory suggestions."""

from __future__ import annotations

from typing import Any

from goat_ts_cig.knowledge_graph import KnowledgeGraph


def suggest_hypotheses(
    graph: KnowledgeGraph,
    top_nodes: int = 10,
    constraint_types: tuple[str, ...] = ("Causal", "Semantic", "Evidence"),
) -> list[dict[str, Any]]:
    """
    Use converged influences and edge types to suggest hypotheses.
    Returns list of { "premise_nodes", "constraint", "suggestion" }.
    """
    nodes = sorted(
        graph._nodes.values(),
        key=lambda n: n.get("influence", 0.0),
        reverse=True,
    )[:top_nodes]
    node_ids = {n["id"] for n in nodes}
    edges_in = [e for e in graph._edges if e["to"] in node_ids and e["from"] in node_ids and e["type"] in constraint_types]

    hypotheses = []
    for e in edges_in:
        from_id, to_id, etype, strength = e["from"], e["to"], e["type"], e.get("strength", 1.0)
        from_meta = (graph.get_node(from_id) or {}).get("metadata", {})
        to_meta = (graph.get_node(to_id) or {}).get("metadata", {})
        from_label = from_meta.get("label", from_meta.get("name", str(from_id)))
        to_label = to_meta.get("label", to_meta.get("name", str(to_id)))
        hypotheses.append({
            "premise_nodes": [from_id, to_id],
            "constraint": etype,
            "strength": strength,
            "suggestion": f"Explore if {from_label} (under {etype}) relates to {to_label}.",
        })
    return hypotheses[:20]


def format_hypotheses_text(hypotheses: list[dict[str, Any]]) -> str:
    """Render hypotheses as readable text."""
    lines = ["=== Hypotheses ==="]
    for i, h in enumerate(hypotheses, 1):
        lines.append(f"  {i}. {h['suggestion']}")
    return "\n".join(lines)
