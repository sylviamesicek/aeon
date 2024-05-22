//! Generic integration methods for vector valued ODEs.
//!
//! This is used by the Method of Lines hyperbolic solver, and the Hyperbolic Relaxation elliptic solver.

#![allow(clippy::needless_range_loop)]

mod forward_euler;
mod rk4;

pub use forward_euler::ForwardEuler;
pub use rk4::Rk4;

/// A vector valued ordinary differential equation.
pub trait Ode {
    /// Dimension of problem.
    fn dim(&self) -> usize;
    /// Preprocess system (for instance, setting hanging nodes
    /// apply boundary conditions, copy ghost nodes, etc.)
    fn preprocess(&mut self, system: &mut [f64]);
    /// Compute temporal derivative.
    fn derivative(&mut self, system: &[f64], result: &mut [f64]);
}
