"""Basic tests for CIG (no Rust required). Run from repo root: python -m pytest tests/ -v"""

import sys
from pathlib import Path

# Add python dir so goat_ts_cig and bindings are importable
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "python"))


def test_ts_propagation_python_fallback():
    """Wave propagation runs (Python fallback if Rust not built)."""
    from goat_ts_cig.utils import graph_dict_to_rust_format, rust_influences_to_dict
    from goat_ts_cig.ts_engine import run_ts_propagation, HAS_RUST

    graph = {
        "nodes": [
            {"id": 1, "influence": 1.0},
            {"id": 2, "influence": 0.0},
            {"id": 3, "influence": 0.0},
        ],
        "edges": [
            {"from": 1, "to": 2, "strength": 0.8},
            {"from": 2, "to": 3, "strength": 0.5},
        ],
    }
    out = run_ts_propagation(graph, ticks=5, decay=0.1, epsilon=0.01, max_ticks=10)
    assert isinstance(out, dict)
    assert 1 in out and 2 in out and 3 in out
    assert out[1] >= 0 and out[2] >= 0 and out[3] >= 0
    # Seed node 1 should have highest influence after a few ticks
    assert out[1] >= out[3]


def test_ts_propagation_converges_and_uses_rust_when_available():
    """Wave propagation converges (delta < epsilon) and prefers Rust when built."""
    from goat_ts_cig.ts_engine import run_ts_propagation, HAS_RUST

    graph = {
        "nodes": [
            {"id": 1, "influence": 1.0},
            {"id": 2, "influence": 0.0},
        ],
        "edges": [
            {"from": 1, "to": 2, "strength": 0.5},
            {"from": 2, "to": 1, "strength": 0.5},
        ],
    }
    out = run_ts_propagation(graph, ticks=50, decay=0.1, epsilon=1e-6, max_ticks=200)
    assert isinstance(out, dict)
    assert 1 in out and 2 in out
    # Both nodes should end up with similar influence once fully converged.
    assert abs(out[1] - out[2]) < 0.05
    # Smoke check that Rust path doesn't raise when available.
    if HAS_RUST:
        # A second call should also succeed and re-use the compiled extension.
        out2 = run_ts_propagation(graph, ticks=10, decay=0.1, epsilon=0.01, max_ticks=20)
        assert isinstance(out2, dict)


def test_knowledge_graph_and_idea_map():
    """Knowledge graph CRUD and idea map generation."""
    from goat_ts_cig.knowledge_graph import KnowledgeGraph
    from goat_ts_cig.cig_generator import build_idea_map

    g = KnowledgeGraph(in_memory_only=True)
    g.add_node(10, metadata={"label": "A"})
    g.add_node(20, metadata={"label": "B"})
    g.add_edge(10, 20, strength=0.9)
    g._nodes[10]["influence"] = 0.8
    g._nodes[20]["influence"] = 0.3

    idea_map = build_idea_map(g, top_k=5)
    assert len(idea_map) == 2
    assert idea_map[0]["node_id"] == 10
    assert idea_map[0]["label"] == "A"
    assert len(idea_map[0]["links"]) == 1
    assert idea_map[0]["links"][0]["to"] == 20


def test_run_cig_smoke():
    """Full run_cig workflow (seed only, no extra data)."""
    from goat_ts_cig.interface import run_cig

    result = run_cig("TestConcept", output_json=True)
    assert "seed" in result
    assert result["seed"] == "TestConcept"
    assert "idea_map" in result
    assert "hypotheses" in result
    assert isinstance(result["idea_map"], list)
    assert len(result["idea_map"]) >= 1
