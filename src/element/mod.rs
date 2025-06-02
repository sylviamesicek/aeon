use crate::geometry::IndexSpace;
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

#[derive(Default, Clone)]
pub struct UniformInterpolate<const N: usize> {
    approx: ApproxOperator,
    num_points: usize,
}

impl<const N: usize> UniformInterpolate<N> {
    pub fn build(&mut self, support: usize, order: usize, point: [f64; N]) -> Result<(), SvdError> {
        self.approx.build(
            &Uniform::new([support]),
            &Monomials::new([order]),
            &ProductValue::new(point),
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
