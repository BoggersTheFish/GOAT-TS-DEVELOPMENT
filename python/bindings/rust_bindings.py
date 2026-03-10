"""
PyO3-imported Rust modules. Requires the goat_ts_core extension to be built and installed:
  cd goat-ts-cig/rust && maturin develop
"""

try:
    from goat_ts_core import RustGraph, simulate_waves_flat, propagate_waves
except ImportError:
    RustGraph = None
    simulate_waves_flat = None
    propagate_waves = None

__all__ = ["RustGraph", "simulate_waves_flat", "propagate_waves"]
