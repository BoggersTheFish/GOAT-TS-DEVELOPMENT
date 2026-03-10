"""Run tests without pytest. Usage: python run_tests.py (from repo root)."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "python"))
sys.path.insert(0, str(root))


def run():
    from tests.test_cig import (
        test_ts_propagation_python_fallback,
        test_knowledge_graph_and_idea_map,
        test_run_cig_smoke,
    )
    tests = [
        ("TS propagation (Python fallback)", test_ts_propagation_python_fallback),
        ("Knowledge graph + idea map", test_knowledge_graph_and_idea_map),
        ("run_cig smoke", test_run_cig_smoke),
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  OK  {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed.append((name, e))
    if failed:
        print(f"\n{len(failed)} failed")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
