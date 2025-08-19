use crate::{geometry::IndexSpace, prelude::HyperBox};
use faer::linalg::svd::SvdError;
use std::array;

mod approx;
mod basis;
mod helpers;
mod operations;
mod support;

pub use approx::*;
pub use basis::*;
pub use helpers::*;
pub use operations::*;
pub use support::*;

/// Performs uniform interpolation on a [-1, 1]ᴺ hypercube.
#[derive(Default, Clone)]
pub struct UniformInterpolate<const N: usize> {
    approx: ApproxOperator,
    num_points: usize,
}

impl<const N: usize> UniformInterpolate<N> {
    pub fn build(
        &mut self,
        support: usize,
        order: usize,
        bounds: HyperBox<N>,
        point: [f64; N],
    ) -> Result<(), SvdError> {
        let local = bounds.global_to_local(point);
        let cube: [f64; N] = array::from_fn(|axis| local[axis] * 2.0 - 1.0);

        self.approx.build(
            &Uniform::new([support]),
            &Monomials::new([order]),
            &ProductValue::new(cube),
        )?;
        self.num_points = support;
        Ok(())
    }

    pub fn apply(&self, values: &[f64]) -> f64 {
        let weights: [_; N] = array::from_fn(|axis| self.approx.weights(axis));

        let mut result = 0.0;

        let space = IndexSpace::new([self.num_points; N]);
        for vertex in space.iter() {
            let index = space.linear_from_cartesian(vertex);

            let weight: f64 = array::from_fn::<_, N, _>(|axis| weights[axis][vertex[axis]])
                .iter()
                .product();

            result += weight * values[index];
        }

        result
    }
}
