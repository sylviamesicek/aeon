mod hyper;
mod intergrate;

use std::convert::Infallible;

pub use hyper::{HyperRelaxError, HyperRelaxSolver};
pub use intergrate::{Integrator, Method};

use crate::{image::ImageRef, mesh::Mesh};

/// Trait for implementing solver callbacks (most often used to output visualization data).
pub trait SolverCallback<const N: usize> {
    type Error;

    /// Called once for every iteration in a iterative solver.
    fn callback(
        &mut self,
        _mesh: &Mesh<N>,
        _input: ImageRef,
        _output: ImageRef,
        _iteration: usize,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<const N: usize> SolverCallback<N> for () {
    type Error = Infallible;
}

impl<const N: usize, F: Fn(&Mesh<N>, ImageRef, ImageRef, usize)> SolverCallback<N> for F {
    type Error = Infallible;
    fn callback(
        &mut self,
        mesh: &Mesh<N>,
        input: ImageRef,
        output: ImageRef,
        iteration: usize,
    ) -> Result<(), Infallible> {
        self(mesh, input, output, iteration);
        Ok(())
    }
}
