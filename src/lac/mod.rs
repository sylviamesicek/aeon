mod bicgstab;

use std::fmt::Debug;

pub use bicgstab::{BiCGStabConfig, BiCGStabError, BiCGStabSolver};

/// A linear map between vectors of
/// a given dimension.
pub trait LinearMap {
    /// Dimension of the linear map.
    fn dimension(&self) -> usize;

    /// Application of the linear map.
    fn apply(&mut self, src: &[f64], dest: &mut [f64]);

    /// An optional callback for logging residuals and iterations.
    fn callback(&self, iteration: usize, residual: f64, solution: &[f64]) {
        _ = iteration;
        _ = residual;
        _ = solution;
    }
}

/// A matrix-free iterative linear solver.
pub trait LinearSolver {
    type Error: Debug;
    type Config;

    fn new(dimension: usize, config: &Self::Config) -> Self;

    /// Dimension of the linear solver.
    fn dimension(&self) -> usize;

    /// Inverts a linear problem.
    fn solve<M: LinearMap>(
        &mut self,
        map: M,
        rhs: &[f64],
        solution: &mut [f64],
    ) -> Result<(), Self::Error>;
}

/// An identity map.
pub struct IdentityMap {
    dimension: usize,
}

impl IdentityMap {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl LinearMap for IdentityMap {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(&mut self, src: &[f64], dest: &mut [f64]) {
        dest.clone_from_slice(src);
    }
}
