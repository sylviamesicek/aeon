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

    /// Compute temporal derivative of system, then stores it in system.
    fn derivative(&mut self, system: &mut [f64]);

    fn copy_slice(&mut self, source: &[f64], dest: &mut [f64]) {
        dest.copy_from_slice(source);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_ode() {
        struct ExpOde;

        impl Ode for ExpOde {
            fn dim(&self) -> usize {
                1
            }

            fn derivative(&mut self, system: &mut [f64]) {
                system[0] = 2.0 * system[0];
            }
        }

        let mut integrator = Rk4::new();

        let mut state = [1.0];
        for _ in 0..1000 {
            integrator.step(1.0 / 1000.0, &mut ExpOde, &mut state);
        }

        assert!((state[0] - f64::exp(2.0)).abs() <= 1e-8);
    }
}
