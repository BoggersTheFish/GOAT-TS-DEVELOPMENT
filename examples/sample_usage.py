"""Example: run CIG with a seed concept (no Rust extension required; uses Python fallback)."""

import sys
from pathlib import Path

# Add project python dir so goat_ts_cig and bindings are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from goat_ts_cig.interface import run_cig

if __name__ == "__main__":
    result = run_cig("AI Ethics", output_json=False)
    print(result)
