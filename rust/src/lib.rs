mod graph_engine;
mod wave_engine;

use pyo3::prelude::*;

#[pymodule]
fn goat_ts_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<graph_engine::RustGraph>()?;
    m.add_function(wrap_pyfunction!(wave_engine::propagate_waves, m)?)?;
    m.add_function(wrap_pyfunction!(wave_engine::simulate_waves_flat, m)?)?;
    Ok(())
}
