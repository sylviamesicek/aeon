mod hyper;
mod intergrate;

pub use hyper::{HyperRelaxError, HyperRelaxSolver};

use crate::{
    mesh::{Function, Mesh},
    system::SystemSlice,
};

pub trait SolverCallback<const N: usize>: Function<N> {
    fn callback(
        &self,
        _mesh: &Mesh<N>,
        _input: SystemSlice<Self::Input>,
        _output: SystemSlice<Self::Output>,
        _iteration: usize,
    ) {
    }
}
