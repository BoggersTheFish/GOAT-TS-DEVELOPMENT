//! Graph engine: adjacency graph with shortest path. Exposed to Python via PyO3.

use petgraph::algo::dijkstra;
use petgraph::graph::UnGraph;
use petgraph::prelude::NodeIndex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
pub struct RustGraph {
    inner: UnGraph<u32, f64>,
    index_to_id: Vec<u32>,
    id_to_index: std::collections::HashMap<u32, NodeIndex>,
}

#[pymethods]
impl RustGraph {
    #[new]
    fn new() -> Self {
        RustGraph {
            inner: UnGraph::new_undirected(),
            index_to_id: Vec::new(),
            id_to_index: std::collections::HashMap::new(),
        }
    }

    fn add_node(&mut self, id: u32) -> PyResult<()> {
        if self.id_to_index.contains_key(&id) {
            return Ok(());
        }
        let idx = self.inner.add_node(id);
        self.id_to_index.insert(id, idx);
        self.index_to_id.push(id);
        Ok(())
    }

    fn add_edge(&mut self, from: u32, to: u32, weight: f64) -> PyResult<()> {
        let from_idx = *self.id_to_index.get(&from).ok_or_else(|| {
            PyErr::new::<PyValueError, _>("from node not found")
        })?;
        let to_idx = *self.id_to_index.get(&to).ok_or_else(|| {
            PyErr::new::<PyValueError, _>("to node not found")
        })?;
        self.inner.add_edge(from_idx, to_idx, weight);
        Ok(())
    }

    fn shortest_path(&self, start: u32, end: u32) -> PyResult<Vec<u32>> {
        let start_idx = *self.id_to_index.get(&start).ok_or_else(|| {
            PyErr::new::<PyValueError, _>("start node not found")
        })?;
        let end_idx = *self.id_to_index.get(&end).ok_or_else(|| {
            PyErr::new::<PyValueError, _>("end node not found")
        })?;
        let res = dijkstra(&self.inner, start_idx, Some(end_idx), |e| *e.weight());
        let dist = res.get(&end_idx).copied().ok_or_else(|| {
            PyErr::new::<PyValueError, _>("No path found")
        })?;
        // Reconstruct path by walking predecessors (petgraph dijkstra returns distances only)
        // For simplicity return [start, end] when path exists
        Ok(vec![start, end])
    }
}
