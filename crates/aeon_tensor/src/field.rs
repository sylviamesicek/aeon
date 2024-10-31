use crate::{Matrix, Metric, Space, Tensor, Tensor3, Tensor4, Vector};

// ************************
// Fields *****************
// ************************

#[derive(Clone, Copy)]
pub struct ScalarFieldC1<const N: usize> {
    pub value: f64,
    pub derivs: Vector<N>,
}

impl<const N: usize> From<ScalarFieldC2<N>> for ScalarFieldC1<N> {
    fn from(value: ScalarFieldC2<N>) -> Self {
        ScalarFieldC1 {
            value: value.value,
            derivs: value.derivs,
        }
    }
}

impl<const N: usize> ScalarFieldC1<N> {
    pub fn lie_derivative(&self, direction: VectorFieldC1<N>) -> f64 {
        let s = Space::<N>;
        s.sum(|[m]| direction.value[m] * self.derivs[m])
    }

    pub fn gradient(&self, _metric: &Metric<N>) -> Vector<N> {
        self.derivs
    }
}

#[derive(Clone, Copy)]
pub struct ScalarFieldC2<const N: usize> {
    pub value: f64,
    pub derivs: Vector<N>,
    pub second_derivs: Matrix<N>,
}

impl<const N: usize> ScalarFieldC2<N> {
    pub fn lie_derivative(&self, direction: VectorFieldC1<N>) -> f64 {
        let c1: ScalarFieldC1<N> = self.clone().into();
        c1.lie_derivative(direction)
    }

    pub fn gradient(&self, _metric: &Metric<N>) -> Vector<N> {
        self.derivs
    }

    pub fn hessian(&self, metric: &Metric<N>) -> Matrix<N> {
        VectorFieldC1 {
            value: self.derivs,
            derivs: self.second_derivs,
        }
        .gradient(metric)
    }
}

#[derive(Clone)]
pub struct VectorFieldC1<const N: usize> {
    pub value: Vector<N>,
    pub derivs: Matrix<N>,
}

impl<const N: usize> VectorFieldC1<N> {
    pub fn lie_derivative(&self, direction: VectorFieldC1<N>) -> Vector<N> {
        let s = Space::<N>;

        s.vector(|i| {
            s.sum(|[m]| {
                direction.value[m] * self.derivs[[i, m]] + self.value[m] * direction.derivs[[m, i]]
            })
        })
    }

    pub fn gradient(&self, metric: &Metric<N>) -> Matrix<N> {
        let s = Space::<N>;

        Matrix::from_fn(|[i, j]| {
            self.derivs[[i, j]] - s.sum(|[m]| metric.christoffel_2nd()[[m, i, j]] * self.value[m])
        })
    }
}

#[derive(Clone)]
pub struct MatrixFieldC1<const N: usize> {
    pub value: Matrix<N>,
    pub derivs: Tensor3<N>,
}

impl<const N: usize> MatrixFieldC1<N> {
    pub fn lie_derivative(&self, direction: VectorFieldC1<N>) -> Matrix<N> {
        let s = Space::<N>;

        Tensor::from_fn(|[i, j]| {
            s.sum(|[m]| {
                direction.value[m] * self.derivs[[i, j, m]]
                    + self.value[[i, m]] * direction.derivs[[m, j]]
                    + self.value[[m, j]] * direction.derivs[[m, i]]
            })
        })
    }

    pub fn gradient(&self, metric: &Metric<N>) -> Tensor3<N> {
        let s = Space::<N>;

        Tensor3::from_fn(|[i, j, k]| {
            let term1 = self.derivs[[i, j, k]];
            let term2 = s.sum(|[m]| {
                metric.christoffel_2nd()[[m, i, k]] * self.value[[m, j]]
                    + metric.christoffel_2nd()[[m, j, k]] * self.value[[i, m]]
            });
            term1 - term2
        })
    }
}

#[derive(Clone)]
pub struct MatrixFieldC2<const N: usize> {
    pub value: Matrix<N>,
    pub derivs: Tensor3<N>,
    pub second_derivs: Tensor4<N>,
}

impl<const N: usize> From<MatrixFieldC2<N>> for MatrixFieldC1<N> {
    fn from(value: MatrixFieldC2<N>) -> Self {
        MatrixFieldC1 {
            value: value.value,
            derivs: value.derivs,
        }
    }
}

impl<const N: usize> MatrixFieldC2<N> {
    pub fn lie_derivative(&self, direction: VectorFieldC1<N>) -> Matrix<N> {
        let c1: MatrixFieldC1<N> = self.clone().into();
        c1.lie_derivative(direction)
    }

    pub fn gradient(&self, metric: &Metric<N>) -> Tensor3<N> {
        let c1: MatrixFieldC1<N> = self.clone().into();
        c1.gradient(metric)
    }
}
