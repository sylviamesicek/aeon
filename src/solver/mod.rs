mod hyper;
mod intergrate;

use std::convert::Infallible;

pub use hyper::{HyperRelaxError, HyperRelaxSolver};
pub use intergrate::{Integrator, Method};

use crate::{
    mesh::Mesh,
    system::{System, SystemSlice},
};

/// Trait for implementing solver callbacks (most often used to output visualization data).
pub trait SolverCallback<const N: usize, S: System> {
    type Error;

    /// Called once for every iteration in a iterative solver.
    fn callback(
        &self,
        _mesh: &Mesh<N>,
        _input: SystemSlice<S>,
        _output: SystemSlice<S>,
        _iteration: usize,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<const N: usize, S: System> SolverCallback<N, S> for () {
    type Error = Infallible;
}

impl<const N: usize, S: System, F: Fn(&Mesh<N>, SystemSlice<S>, SystemSlice<S>, usize)>
    SolverCallback<N, S> for F
{
    type Error = Infallible;
    fn callback(
        &self,
        mesh: &Mesh<N>,
        input: SystemSlice<S>,
        output: SystemSlice<S>,
        iteration: usize,
    ) -> Result<(), Infallible> {
        self(mesh, input, output, iteration);
        Ok(())
    }
}
