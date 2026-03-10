"""YAML/JSON config loader for graph backend, wave params, and system settings."""

import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

DEFAULT_CONFIG = {
    "graph": {
        "backend": "sqlite",
        "path": "data/cig_graph.db",
        "in_memory_max_nodes": 1000,
    },
    "wave": {
        "ticks": 20,
        "decay": 0.15,
        "epsilon": 0.01,
        "max_ticks": 50,
    },
    "system": {
        "max_threads": None,
        "lazy_load": True,
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from file or return defaults. Supports .json and .yaml."""
    if path is None:
        base = Path(__file__).resolve().parent.parent.parent
        for name in ("config.yaml", "config.yml", "config.json"):
            candidate = base / name
            if candidate.exists():
                path = candidate
                break
        else:
            return dict(DEFAULT_CONFIG)

    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_CONFIG)

    with open(path, encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise RuntimeError("PyYAML required for YAML config. pip install pyyaml")
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)

    def merge(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out

    return merge(DEFAULT_CONFIG, data)


def get_env_override() -> dict[str, Any]:
    """Optional overrides from environment."""
    overrides = {}
    if os.environ.get("CIG_GRAPH_PATH"):
        overrides.setdefault("graph", {})["path"] = os.environ["CIG_GRAPH_PATH"]
    if os.environ.get("CIG_MAX_TICKS"):
        try:
            overrides.setdefault("wave", {})["max_ticks"] = int(os.environ["CIG_MAX_TICKS"])
        except ValueError:
            pass
    return overrides
