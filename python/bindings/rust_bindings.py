"""
PyO3-imported Rust modules. Requires the goat_ts_core extension to be built and installed:
  cd GOAT-TS-SUPERLITE/rust && maturin develop
  On Python 3.13+, set PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 before maturin develop.
"""

try:
    from goat_ts_core import (
        RustGraph,
        get_hardware_limits,
        simulate_waves_flat,
        simulate_waves_flat_buf,
        propagate_waves,
        simple_wave_decay,
    )
except ImportError:
    RustGraph = None
    get_hardware_limits = None
    simulate_waves_flat = None
    simulate_waves_flat_buf = None
    propagate_waves = None
    simple_wave_decay = None

__all__ = [
    "RustGraph",
    "get_hardware_limits",
    "simulate_waves_flat",
    "simulate_waves_flat_buf",
    "propagate_waves",
    "simple_wave_decay",
]
