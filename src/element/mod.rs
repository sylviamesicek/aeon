use crate::geometry::IndexSpace;
use faer::{
    Conj, Mat, RowRef,
    col::generic::Col,
    linalg::{matmul::dot, svd::SvdError},
};
use helpers::LeastSquares;
use reborrow::{Reborrow, ReborrowMut};
use std::array;

mod basis;
mod helpers;
mod operations;
mod support;

pub use basis::*;
pub use operations::*;
pub use support::*;

#[derive(Clone)]
pub struct ApproxOperator {
    lls: LeastSquares,
    /// Cache for a vandermonde matrix.
    vandermonde: Mat<f64>,
    /// Cache for rhs of computation.
    rhs: Mat<f64>,
    /// Shape functions for each operation.
    shape: Mat<f64>,
}

impl ApproxOperator {
    pub fn build<const N: usize>(
        &mut self,
        support: &impl Support<N>,
        basis: &impl Basis<N>,
        ops: &impl LinearOperator<N>,
    ) -> Result<(), SvdError> {
        self.vandermonde
            .resize_with(support.num_points(), basis.order(), |i, j| {
                let point = support.point(i);
                basis.func(j).value(point)
            });

        self.rhs
            .resize_with(basis.order(), ops.num_operations(), |deg, i| {
                ops.apply(i, &basis.func(deg))
            });

        self.lls.least_squares(
            self.vandermonde.transpose(),
            self.shape.rb_mut(),
            self.rhs.rb(),
        )?;

        Ok(())
    }

    pub fn apply(&self, i: usize, values: &[f64]) -> f64 {
        dot::inner_prod(
            self.shape.row(i),
            Conj::Yes,
            Col::from_slice(values),
            Conj::No,
        )
    }

    pub fn weights(&self, i: usize) -> RowRef<f64> {
        self.shape.row(i)
    }
}

impl Default for ApproxOperator {
    fn default() -> Self {
        Self {
            lls: Default::default(),
            vandermonde: Mat::zeros(0, 0),
            rhs: Mat::zeros(0, 0),
            shape: Mat::zeros(0, 0),
        }
    }
}

#[derive(Default, Clone)]
pub struct ProlongEngine<const N: usize> {
    approx: ApproxOperator,
    num_points: usize,
}

impl<const N: usize> ProlongEngine<N> {
    pub fn build(
        &mut self,
        support: &impl Support<1>,
        basis: &impl Basis<1>,
        point: [f64; N],
    ) -> Result<(), SvdError> {
        self.approx
            .build(support, basis, &ProductValue::new(point))?;
        self.num_points = support.num_points();
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
