"""YAML/JSON config loader for graph backend, wave params, and system settings.
Includes hardware autodetection (psutil) for data-gen / LITE: RAM and core count
drive max_ticks, batch_size, and parallelism (off if <2 cores or low RAM)."""

import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

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
    "hardware": {
        "max_ticks": 15,
        "batch_size": 200,
        "parallelism": True,
        "ram_gb": None,
        "num_cores": None,
    },
}


def get_hardware_config() -> dict[str, Any]:
    """
    Autodetect RAM and cores; return settings for data-gen / wave simulation.
    - RAM < 4 GB: max_ticks=5, batch_size=50, parallelism=False
    - Else: max_ticks=15, batch_size=200, parallelism=True
    - parallelism is forced False if physical cores < 2 (for low-end Pentium).
    Values are intended to be passed to Rust (max_ticks, batch_size) and to
    control use of rayon (parallelism).
    """
    ram_gb: float | None = None
    num_cores: int | None = None

    if HAS_PSUTIL:
        try:
            vmem = psutil.virtual_memory()
            ram_gb = vmem.total / (1024**3)
        except Exception:
            pass
        try:
            num_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        except Exception:
            num_cores = 1

    if ram_gb is None:
        ram_gb = 4.0
    if num_cores is None:
        num_cores = 1

    low_ram = ram_gb < 4.0
    few_cores = num_cores < 2

    if low_ram:
        max_ticks = 5
        batch_size = 50
        parallelism = False
    else:
        max_ticks = 15
        batch_size = 200
        parallelism = True

    if few_cores:
        parallelism = False

    return {
        "max_ticks": max_ticks,
        "batch_size": batch_size,
        "parallelism": parallelism,
        "ram_gb": round(ram_gb, 2),
        "num_cores": num_cores,
    }


def get_suggested_rayon_threads() -> int:
    """
    Suggested Rayon thread pool size for Rust (0 = use serial path).
    Uses Rust get_hardware_limits() when available; otherwise hardware config.
    """
    try:
        from bindings.rust_bindings import get_hardware_limits
        if get_hardware_limits is not None:
            _max_ticks, batch_hint = get_hardware_limits()
            hw = get_hardware_config()
            if not hw.get("parallelism", True):
                return 0
            return max(1, (batch_hint // 256) if batch_hint else 2)
    except Exception:
        pass
    hw = get_hardware_config()
    return 0 if not hw.get("parallelism", True) else 1


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
    if os.environ.get("CIG_BATCH_SIZE"):
        try:
            overrides.setdefault("hardware", {})["batch_size"] = int(os.environ["CIG_BATCH_SIZE"])
        except ValueError:
            pass
    if os.environ.get("CIG_PARALLELISM", "").lower() in ("0", "false", "no"):
        overrides.setdefault("hardware", {})["parallelism"] = False
    elif os.environ.get("CIG_PARALLELISM", "").lower() in ("1", "true", "yes"):
        overrides.setdefault("hardware", {})["parallelism"] = True
    return overrides


def load_config_with_hardware(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load config from file, apply env overrides, then fill/override 'hardware'
    from autodetection (get_hardware_config()). Callers can use
    config['hardware']['max_ticks'], config['hardware']['batch_size'],
    config['hardware']['parallelism'] for data-gen and for passing to Rust.
    """
    config = load_config(path)
    for k, v in get_env_override().items():
        config.setdefault(k, {}).update(v)
    hw = get_hardware_config()
    config.setdefault("hardware", {})
    config["hardware"].update(hw)
    for k, v in get_env_override().get("hardware", {}).items():
        config["hardware"][k] = v
    return config
