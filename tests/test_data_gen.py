"""Tests for data-generation pipeline: hardware config, ingestion, JSONL format."""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "python"))
sys.path.insert(0, str(root))


def test_hardware_config_shape_and_ranges():
    """Hardware autodetection returns expected keys and value ranges."""
    from goat_ts_cig.config import get_hardware_config, load_config_with_hardware

    hw = get_hardware_config()
    assert "max_ticks" in hw
    assert "batch_size" in hw
    assert "parallelism" in hw
    assert "ram_gb" in hw
    assert "num_cores" in hw
    assert hw["max_ticks"] in (5, 15)
    assert hw["batch_size"] in (50, 200)
    assert isinstance(hw["parallelism"], bool)
    assert hw["ram_gb"] is None or (isinstance(hw["ram_gb"], (int, float)) and hw["ram_gb"] >= 0)
    assert isinstance(hw["num_cores"], int) and hw["num_cores"] >= 1

    # Clear env overrides so load_config_with_hardware matches get_hardware_config
    saved = {}
    for key in ("CIG_BATCH_SIZE", "CIG_PARALLELISM", "CIG_MAX_TICKS"):
        if key in os.environ:
            saved[key] = os.environ.pop(key, None)

    try:
        config = load_config_with_hardware()
        assert "hardware" in config
        assert config["hardware"]["max_ticks"] == hw["max_ticks"]
        assert config["hardware"]["batch_size"] == hw["batch_size"]
        assert config["hardware"]["parallelism"] == hw["parallelism"]
    finally:
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val


def test_chat_ingest_extraction():
    """chat_ingest: sentences, node extraction, relates_to edges."""
    from chat_ingest import _sentences, _words, extract_nodes_and_edges, text_to_graph_dict

    text = "AI and ethics matter. Ethics and policy are linked. AI policy is important."
    sents = _sentences(text)
    assert len(sents) >= 2
    words1 = _words(sents[0])
    assert "ai" in words1 or "ethics" in words1
    nodes, edges = extract_nodes_and_edges(text)
    assert len(nodes) >= 2
    assert all("id" in n and "metadata" in n for n in nodes)
    assert all("from" in e and "to" in e and e.get("type") == "relates_to" for e in edges)
    graph_dict = text_to_graph_dict(text)
    assert "nodes" in graph_dict and "edges" in graph_dict
    assert len(graph_dict["nodes"]) == len(nodes) and len(graph_dict["edges"]) == len(edges)


def test_suggest_hypotheses_from_tensions():
    """Tension-based hypothesis gen: high vs low influence pairs, 3–5 suggestions."""
    from goat_ts_cig.hypothesis_engine import (
        id_to_label_from_graph_dict,
        suggest_hypotheses_from_tensions,
    )

    id_to_label = {1: "ai", 2: "ethics", 3: "policy", 4: "learning"}
    influences = {1: 0.9, 2: 0.6, 3: 0.3, 4: 0.1}
    hyps = suggest_hypotheses_from_tensions(influences, id_to_label, max_hypotheses=5)
    assert 1 <= len(hyps) <= 5
    for h in hyps:
        assert "suggestion" in h
        assert "Explore connection between" in h["suggestion"]
        assert "under tension" in h["suggestion"]
        assert "influence_diff" in h
        assert h["node1"] != h["node2"]
    graph_dict = {
        "nodes": [
            {"id": 10, "metadata": {"label": "machine"}},
            {"id": 20, "metadata": {"label": "learning"}},
        ],
        "edges": [],
    }
    labels = id_to_label_from_graph_dict(graph_dict)
    assert labels == {10: "machine", 20: "learning"}


def test_generate_dataset_jsonl_format():
    """generate_dataset produces valid JSONL with prompt/completion or alpaca keys."""
    import json
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "python"))
    sys.path.insert(0, str(root))

    from generate_dataset import run

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("AI and ethics matter. Policy and technology are linked.")
        tmp_path = f.name
    out_path = tempfile.mktemp(suffix=".jsonl")
    try:
        n = run(
            input_path=tmp_path,
            output_path=out_path,
            num_examples=4,
            format_alpaca=False,
        )
        assert n >= 2
        with open(out_path, encoding="utf-8") as fp:
            lines = fp.readlines()
        assert len(lines) >= 2
        for line in lines:
            obj = json.loads(line)
            assert "prompt" in obj and "completion" in obj
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Short.")
        tmp_path = f.name
    out_path2 = tempfile.mktemp(suffix=".jsonl")
    try:
        n2 = run(
            input_path=tmp_path,
            output_path=out_path2,
            num_examples=10,
            format_alpaca=True,
        )
        with open(out_path2, encoding="utf-8") as fp:
            lines = fp.readlines()
        if lines:
            obj = json.loads(lines[0])
            assert "instruction" in obj and "output" in obj
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(out_path2).unlink(missing_ok=True)


def test_simple_wave_decay_stub():
    """Rust simple_wave_decay (stub) runs and respects parallelism flag."""
    try:
        from bindings.rust_bindings import simple_wave_decay
    except ImportError:
        simple_wave_decay = None
    if simple_wave_decay is None:
        return
    influences = [1.0, 0.5, 0.25]
    out_serial = simple_wave_decay(influences, ticks=2, decay=0.1, parallelism=False)
    out_parallel = simple_wave_decay(influences, ticks=2, decay=0.1, parallelism=True)
    assert out_serial == out_parallel
    # Each tick: x *= 0.9. After 2 ticks: 1.0 -> 0.81, 0.5 -> 0.405, 0.25 -> 0.2025
    assert abs(out_serial[0] - 0.81) < 1e-9
    assert abs(out_serial[1] - 0.405) < 1e-9
    assert abs(out_serial[2] - 0.2025) < 1e-9
