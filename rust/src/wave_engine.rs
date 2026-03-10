//! Wave propagation engine: TS-style influence spread with decay and convergence.
//! Supports LITE-style auto-modes (RAM/core detection), interference (signed edge weights),
//! and optimized double-buffered iteration for low-core (e.g. 2–4 core) machines.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use sysinfo::System;

/// LITE-style hardware limits: max_ticks cap and a batch-size hint from RAM and core count.
fn hardware_limits() -> (usize, usize) {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu();

    let total_mem_mb = (sys.total_memory() / 1024).max(1);
    let cores = sys.physical_core_count().unwrap_or(1).max(1);

    let max_ticks_cap = if total_mem_mb < 4 * 1024 {
        25
    } else if total_mem_mb < 8 * 1024 {
        50
    } else {
        100
    };

    let batch_hint = (if total_mem_mb < 4 * 1024 {
        256
    } else if total_mem_mb < 8 * 1024 {
        512
    } else {
        1024
    }) * cores;

    (max_ticks_cap, batch_hint)
}

/// Expose LITE auto-mode limits to Python: (max_ticks_cap, batch_size_hint).
#[pyfunction]
pub fn get_hardware_limits() -> (usize, usize) {
    hardware_limits()
}

/// Simple parallel decay over a vector of influences (example binding from design doc).
#[pyfunction]
pub fn propagate_waves(nodes: Vec<f64>, decay: f64) -> PyResult<Vec<f64>> {
    if nodes.is_empty() {
        return Err(PyErr::new::<PyValueError, _>("Empty nodes"));
    }
    let updated: Vec<f64> = nodes
        .par_iter()
        .map(|&influence| influence * (1.0 - decay))
        .collect();
    Ok(updated)
}

/// Stub wave decay for data-gen: apply decay each tick; serial or parallel per Python hardware config.
/// Each tick: influence *= (1.0 - decay). Returns updated vector (same length as influences).
#[pyfunction]
pub fn simple_wave_decay(
    influences: Vec<f64>,
    ticks: usize,
    decay: f64,
    parallelism: bool,
) -> PyResult<Vec<f64>> {
    if influences.is_empty() {
        return Err(PyErr::new::<PyValueError, _>("Empty influences"));
    }
    let factor = 1.0 - decay;
    let mut out = influences;
    for _ in 0..ticks {
        if parallelism {
            out = out.par_iter().map(|&x| x * factor).collect();
        } else {
            for x in out.iter_mut() {
                *x *= factor;
            }
        }
    }
    Ok(out)
}

/// Full wave simulation: build graph from flat arrays, propagate until convergence.
/// - node_ids: external ids in order (Python ints -> i64)
/// - initial_influences: one per node
/// - edges: list of (from_id, to_id, strength); strength can be negative (interference / inhibition).
/// Returns updated influences in same order as node_ids.
/// Uses double-buffering and rayon work-stealing; LITE-style tick cap applied from RAM.
#[pyfunction]
#[pyo3(signature = (node_ids, initial_influences, edges, ticks, decay, epsilon, max_ticks))]
pub fn simulate_waves_flat(
    node_ids: Vec<i64>,
    initial_influences: Vec<f64>,
    edges: Vec<(i64, i64, f64)>,
    ticks: usize,
    decay: f64,
    epsilon: f64,
    max_ticks: usize,
) -> PyResult<Vec<f64>> {
    if node_ids.is_empty() {
        return Err(PyErr::new::<PyValueError, _>("Empty node_ids"));
    }
    if node_ids.len() != initial_influences.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "node_ids and initial_influences length mismatch",
        ));
    }

    let n = node_ids.len();
    let id_to_idx: HashMap<i64, usize> = node_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Neighbors: for each node index, list of (neighbor_index, edge_strength). Strength can be negative (interference).
    let mut neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (from_id, to_id, strength) in edges {
        if let (Some(&i), Some(&j)) = (id_to_idx.get(&from_id), id_to_idx.get(&to_id)) {
            neighbors[i].push((j, strength));
            neighbors[j].push((i, strength));
        }
    }

    let (hw_max_ticks, _batch_hint) = hardware_limits();
    let max_iters = ticks.min(max_ticks).min(hw_max_ticks);

    // Double-buffer: reuse two buffers to avoid per-iteration allocations (good for 2–4 core / low RAM).
    let mut a = initial_influences;
    let mut b = vec![0.0_f64; n];

    for _ in 0..max_iters {
        let (read, write) = (&a, &mut b);
        write
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out)| {
                let sum: f64 = neighbors[i]
                    .iter()
                    .map(|&(j, w)| read[j] * w)
                    .sum();
                *out = (read[i] * 0.5 + sum * 0.5) * (1.0 - decay);
            });

        let delta: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);

        std::mem::swap(&mut a, &mut b);
        if delta < epsilon {
            break;
        }
    }

    Ok(a)
}

/// Zero-copy input variant: read initial_influences from a buffer (e.g. numpy array, array.array).
/// Same semantics as simulate_waves_flat; use this when you have a large f64 buffer to avoid list conversion.
#[pyfunction]
#[pyo3(signature = (node_ids, initial_influences_buf, edges, ticks, decay, epsilon, max_ticks))]
pub fn simulate_waves_flat_buf(
    py: Python<'_>,
    node_ids: Vec<i64>,
    initial_influences_buf: &Bound<PyAny>,
    edges: Vec<(i64, i64, f64)>,
    ticks: usize,
    decay: f64,
    epsilon: f64,
    max_ticks: usize,
) -> PyResult<Vec<f64>> {
    let buf = PyBuffer::<f64>::get_bound(initial_influences_buf)?;
    let slice = buf.as_slice(py).ok_or_else(|| {
        PyErr::new::<PyValueError, _>("initial_influences_buf must be a contiguous f64 buffer")
    })?;
    let initial_influences: Vec<f64> = slice.iter().map(|c| c.get()).collect();
    drop(buf);
    simulate_waves_flat(
        node_ids,
        initial_influences,
        edges,
        ticks,
        decay,
        epsilon,
        max_ticks,
    )
}
