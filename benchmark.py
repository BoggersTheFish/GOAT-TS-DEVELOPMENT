"""
Quick benchmark for GOAT-TS CIG: run with larger seeds and report timing / hardware limits.
Usage: python benchmark.py [--seed "your seed"] [--repeat 3]
"""
import argparse
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "python"))
sys.path.insert(0, str(root))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="AI ethics and policy", help="Seed concept for run_cig")
    ap.add_argument("--repeat", type=int, default=3, help="Number of runs for timing")
    args = ap.parse_args()

    from goat_ts_cig.interface import run_cig

    try:
        from bindings.rust_bindings import get_hardware_limits
        max_ticks, batch_hint = get_hardware_limits()
        print(f"Hardware limits (LITE): max_ticks={max_ticks}, batch_hint={batch_hint}")
    except Exception:
        print("Rust get_hardware_limits not available")

    print(f"Running run_cig(seed={args.seed!r}) x {args.repeat}...")
    times: list[float] = []
    for i in range(args.repeat):
        t0 = time.perf_counter()
        run_cig(args.seed, output_json=False)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")
    avg = sum(times) / len(times)
    print(f"Average: {avg:.3f}s (measure CPU/RAM with Task Manager or htop)")


if __name__ == "__main__":
    main()
