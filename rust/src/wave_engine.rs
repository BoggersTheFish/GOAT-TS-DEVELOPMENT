//! Wave propagation engine: TS-style influence spread with decay and convergence.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

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

/// Full wave simulation: build graph from flat arrays, propagate until convergence.
/// - node_ids: external ids in order (Python ints -> i64)
/// - initial_influences: one per node
/// - edges: list of (from_id, to_id, strength); ids must be in node_ids
/// Returns updated influences in same order as node_ids.
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

    // Neighbors: for each node index, list of (neighbor_index, edge_strength)
    let mut neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (from_id, to_id, strength) in edges {
        if let (Some(&i), Some(&j)) = (id_to_idx.get(&from_id), id_to_idx.get(&to_id)) {
            neighbors[i].push((j, strength));
            neighbors[j].push((i, strength));
        }
    }

    let mut influences = initial_influences;
    let max_iters = ticks.min(max_ticks);

    for _ in 0..max_iters {
        let new_influences: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let sum: f64 = neighbors[i]
                    .iter()
                    .map(|&(j, w)| influences[j] * w)
                    .sum();
                (influences[i] * 0.5 + sum * 0.5) * (1.0 - decay)
            })
            .collect();

        let delta: f64 = influences
            .iter()
            .zip(new_influences.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        influences = new_influences;
        if delta < epsilon {
            break;
        }
    }

    Ok(influences)
}
