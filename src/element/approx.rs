use crate::element::{Basis, BasisFunction as _, LeastSquares, LinearOperator, Support};
use faer::{
    ColRef, Conj, Mat, MatRef,
    col::generic::Col,
    linalg::{matmul::dot, svd::SvdError},
};
use reborrow::{Reborrow, ReborrowMut};

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

        self.shape
            .resize_with(support.num_points(), ops.num_operations(), |_, _| 0.0);

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

    pub fn weights(&self, i: usize) -> ColRef<'_, f64> {
        self.shape.col(i)
    }

    pub fn shape(&self) -> MatRef<'_, f64> {
        self.shape.rb()
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
