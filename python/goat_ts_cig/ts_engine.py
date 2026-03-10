"""Core TS framework orchestration: load graph, call Rust for propagation, converge."""

from __future__ import annotations

from typing import Any

from goat_ts_cig.utils import graph_dict_to_rust_format, rust_influences_to_dict

try:
    from bindings.rust_bindings import simulate_waves_flat
    HAS_RUST = simulate_waves_flat is not None
except ImportError:
    simulate_waves_flat = None
    HAS_RUST = False


def run_ts_propagation(
    graph_dict: dict[str, Any],
    ticks: int = 20,
    decay: float = 0.15,
    epsilon: float = 0.01,
    max_ticks: int = 50,
) -> dict[int, float]:
    """
    Run wave propagation. Converts graph to Rust format, calls Rust engine, returns node id -> influence.
    Falls back to Python propagation if Rust extension is not available.
    """
    node_list, edge_list = graph_dict_to_rust_format(graph_dict)
    if not node_list:
        return {}

    node_ids = [n[0] for n in node_list]
    initial_influences = [n[1] for n in node_list]

    if HAS_RUST:
        updated = simulate_waves_flat(
            node_ids=node_ids,
            initial_influences=initial_influences,
            edges=edge_list,
            ticks=ticks,
            decay=decay,
            epsilon=epsilon,
            max_ticks=max_ticks,
        )
        return rust_influences_to_dict(node_ids, updated)
    else:
        return _python_fallback_propagate(
            node_ids, initial_influences, edge_list, ticks, decay, epsilon, max_ticks
        )


def _python_fallback_propagate(
    node_ids: list[int],
    initial: list[float],
    edges: list[tuple[int, int, float]],
    ticks: int,
    decay: float,
    epsilon: float,
    max_ticks: int,
) -> dict[int, float]:
    """Simple propagation when Rust is not built: neighbor sum * (1 - decay)."""
    import math
    n = len(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (a, b, w) in edges:
        i, j = id_to_idx.get(a), id_to_idx.get(b)
        if i is not None and j is not None:
            neighbors[i].append((j, w))

    influences = list(initial)
    for _ in range(min(ticks, max_ticks)):
        new_inf = [0.0] * n
        for i in range(n):
            s = sum(influences[j] * w for j, w in neighbors[i])
            new_inf[i] = (influences[i] * 0.5 + s * 0.5) * (1.0 - decay)
        delta = max(abs(influences[i] - new_inf[i]) for i in range(n))
        influences = new_inf
        if not math.isfinite(delta) or delta < epsilon:
            break
    return dict(zip(node_ids, influences))
