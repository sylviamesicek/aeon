mod hyper;
mod intergrate;

pub use hyper::{HyperRelaxError, HyperRelaxSolver};
pub use intergrate::{Integrator, Method};

use crate::{
    mesh::Mesh,
    system::{System, SystemSlice},
};

/// Trait for implementing solver callbacks (most often used to output visualization data).
pub trait SolverCallback<const N: usize, S: System> {
    /// Called once for every iteration in a iterative solver.
    fn callback(
        &self,
        mesh: &Mesh<N>,
        input: SystemSlice<S>,
        output: SystemSlice<S>,
        iteration: usize,
    );
}

impl<const N: usize, S: System> SolverCallback<N, S> for () {
    /// By default do nothing.
    fn callback(
        &self,
        _mesh: &Mesh<N>,
        _input: SystemSlice<S>,
        _output: SystemSlice<S>,
        _iteration: usize,
    ) {
    }
}
