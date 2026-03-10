"""Hypothesis logic: interpret converged states into exploratory suggestions."""

from __future__ import annotations

from typing import Any

from goat_ts_cig.knowledge_graph import KnowledgeGraph


def suggest_hypotheses_from_tensions(
    influences: dict[int, float],
    id_to_label: dict[int, str],
    max_hypotheses: int = 5,
) -> list[dict[str, Any]]:
    """
    Data-gen: pair high-influence nodes with low-influence nodes (tensions).
    Returns 3–5 hypotheses with template:
    "Explore connection between {node1} and {node2} under tension {influence_diff}."
    """
    if not influences or not id_to_label:
        return []
    # Sort by influence descending: (node_id, influence)
    sorted_nodes = sorted(
        [(nid, inf) for nid, inf in influences.items() if id_to_label.get(nid) is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    if len(sorted_nodes) < 2:
        return []
    n = len(sorted_nodes)
    k = min(max_hypotheses, n)
    high_nodes = sorted_nodes[:k]   # top by influence
    low_nodes = sorted_nodes[-k:]  # bottom by influence
    # Form pairs (high, low), avoid same node, sort by |influence diff| descending
    pairs: list[tuple[int, int, float, float, float]] = []
    seen: set[tuple[int, int]] = set()
    for (hid, hinf) in high_nodes:
        for (lid, linf) in low_nodes:
            if hid == lid:
                continue
            key = (min(hid, lid), max(hid, lid))
            if key in seen:
                continue
            seen.add(key)
            diff = round(hinf - linf, 4)
            pairs.append((hid, lid, hinf, linf, diff))
    # Prefer larger tension (bigger diff) first
    pairs.sort(key=lambda x: -abs(x[4]))
    hypotheses = []
    for hid, lid, hinf, linf, diff in pairs[:max_hypotheses]:
        label1 = id_to_label.get(hid, str(hid))
        label2 = id_to_label.get(lid, str(lid))
        hypotheses.append({
            "node1": hid,
            "node2": lid,
            "label1": label1,
            "label2": label2,
            "influence_high": hinf,
            "influence_low": linf,
            "influence_diff": diff,
            "suggestion": f"Explore connection between {label1} and {label2} under tension {diff}.",
        })
    return hypotheses


def id_to_label_from_graph_dict(graph_dict: dict[str, Any]) -> dict[int, str]:
    """Build node id -> label from graph dict (nodes with metadata.label or metadata.name)."""
    out: dict[int, str] = {}
    for n in graph_dict.get("nodes", []):
        nid = n.get("id")
        if nid is None:
            continue
        meta = n.get("metadata") or {}
        label = meta.get("label") or meta.get("name") or str(nid)
        out[int(nid)] = str(label)
    return out


def suggest_hypotheses(
    graph: KnowledgeGraph,
    top_nodes: int = 10,
    constraint_types: tuple[str, ...] = ("Causal", "Semantic", "Evidence", "relates_to"),
) -> list[dict[str, Any]]:
    """
    Use converged influences and edge types to suggest hypotheses.
    Returns list of { "premise_nodes", "constraint", "suggestion" }.
    Also attempts to form lightweight chains where consecutive hypotheses
    share at least one premise node, approximating unresolved tensions.
    """
    nodes = sorted(
        graph._nodes.values(),
        key=lambda n: n.get("influence", 0.0),
        reverse=True,
    )[:top_nodes]
    node_ids = {n["id"] for n in nodes}
    edges_in = [
        e
        for e in graph._edges
        if e["to"] in node_ids and e["from"] in node_ids and e["type"] in constraint_types
    ]

    # Base hypotheses scored by edge strength and target influence.
    raw: list[dict[str, Any]] = []
    for e in edges_in:
        from_id, to_id, etype, strength = e["from"], e["to"], e["type"], e.get("strength", 1.0)
        from_meta = (graph.get_node(from_id) or {}).get("metadata", {})
        to_meta = (graph.get_node(to_id) or {}).get("metadata", {})
        from_label = from_meta.get("label", from_meta.get("name", str(from_id)))
        to_label = to_meta.get("label", to_meta.get("name", str(to_id)))
        target_inf = (graph.get_node(to_id) or {}).get("influence", 0.0)
        raw.append(
            {
                "premise_nodes": [from_id, to_id],
                "constraint": etype,
                "strength": float(strength),
                "target_influence": float(target_inf),
                "suggestion": f"Explore if {from_label} (under {etype}) relates to {to_label}.",
            }
        )

    # Sort by a simple tension score: strong edge + high target influence.
    raw.sort(key=lambda h: (h["strength"], h["target_influence"]), reverse=True)

    # Build chains: each new hypothesis is attached to the first existing chain
    # that shares a premise node; otherwise start a new chain.
    chains: list[list[dict[str, Any]]] = []
    for h in raw:
        added = False
        for chain in chains:
            chain_nodes = {nid for item in chain for nid in item["premise_nodes"]}
            if any(nid in chain_nodes for nid in h["premise_nodes"]):
                chain.append(h)
                added = True
                break
        if not added:
            chains.append([h])

    # Flatten chains back into a list, annotating chain_id and step index.
    hypotheses: list[dict[str, Any]] = []
    for chain_id, chain in enumerate(chains, start=1):
        for step, h in enumerate(chain, start=1):
            enriched = dict(h)
            enriched["chain_id"] = chain_id
            enriched["chain_step"] = step
            hypotheses.append(enriched)

    return hypotheses[:20]


def format_hypotheses_text(hypotheses: list[dict[str, Any]]) -> str:
    """Render hypotheses as readable text."""
    if not hypotheses:
        return "=== Hypotheses ==="

    # Group by chain for a more narrative structure.
    chains: dict[int, list[dict[str, Any]]] = {}
    for h in hypotheses:
        cid = int(h.get("chain_id", 0) or 0)
        chains.setdefault(cid, []).append(h)

    lines = ["=== Hypotheses ==="]
    for cid, items in sorted(chains.items(), key=lambda kv: kv[0] or 0):
        if cid:
            lines.append(f"-- Chain {cid} --")
        for i, h in enumerate(sorted(items, key=lambda x: x.get("chain_step", 0)), 1):
            lines.append(f"  {i}. {h['suggestion']}")

    return "\n".join(lines)
