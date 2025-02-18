use crate::{
    Matrix, MatrixFieldC2, Space, Tensor, Tensor3, Tensor4, TensorLayout, Vector, VectorFieldC1,
};

#[derive(Clone, Default)]
pub struct Metric<const N: usize> {
    g: Matrix<N>,
    g_derivs: Tensor3<N>,
    g_second_derivs: Tensor4<N>,

    gdet: f64,
    gdet_derivs: Vector<N>,

    ginv: Matrix<N>,
    ginv_derivs: Tensor3<N>,

    gamma: Tensor3<N>,
    gamma_derivs: Tensor4<N>,

    gamma_2nd: Tensor3<N>,
    gamma_2nd_derivs: Tensor4<N>,
}

impl Metric<2> {
    pub fn new(g: MatrixFieldC2<2>) -> Self {
        // Compute determinant and its derivatives.
        let gdet = g.value[[0, 0]] * g.value[[1, 1]] - g.value[[0, 1]] * g.value[[1, 0]];
        let gdet_derivs = Vector::from_fn(|[i]| {
            g.value[[0, 0]] * g.derivs[[1, 1, i]] + g.derivs[[0, 0, i]] * g.value[[1, 1]]
                - g.derivs[[0, 1, i]] * g.value[[1, 0]]
                - g.value[[0, 1]] * g.derivs[[1, 0, i]]
        });

        // Compute inverse and its derivatives
        let ginv = Matrix::from([
            [g.value[[1, 1]] / gdet, -g.value[[1, 0]] / gdet],
            [-g.value[[0, 1]] / gdet, g.value[[0, 0]] / gdet],
        ]);
        let ginv_derivs = Tensor3::from({
            let grr_inv_derivs = *Vector::from_fn(|[i]| {
                g.derivs[[1, 1, i]] / gdet - g.value[[1, 1]] / (gdet * gdet) * gdet_derivs[[i]]
            })
            .inner();

            let grz_inv_derivs = *Vector::from_fn(|[i]| {
                -g.derivs[[0, 1, i]] / gdet + g.value[[0, 1]] / (gdet * gdet) * gdet_derivs[[i]]
            })
            .inner();

            let gzz_inv_derivs = *Vector::from_fn(|[i]| {
                g.derivs[[0, 0, i]] / gdet - g.value[[0, 0]] / (gdet * gdet) * gdet_derivs[[i]]
            })
            .inner();

            [
                [grr_inv_derivs, grz_inv_derivs],
                [grz_inv_derivs, gzz_inv_derivs],
            ]
        });

        // Set fields of result
        let mut result = Self {
            g: g.value,
            g_derivs: g.derivs,
            g_second_derivs: g.second_derivs,

            gdet,
            gdet_derivs,

            ginv,
            ginv_derivs,

            ..Self::default()
        };

        result.compute_christoffel();

        result
    }
}

impl<const N: usize> Metric<N> {
    /// Computes and caches the metric's christofell symbols from the currently set g, ginv, and derivatives.
    pub fn compute_christoffel(&mut self) {
        let space = Space::<N>;
        // Compute christoffel symbols of the first kind.
        self.gamma = Tensor::from_fn(|[i, j, k]| {
            0.5 * (self.g_derivs[[i, j, k]] + self.g_derivs[[k, i, j]] - self.g_derivs[[j, k, i]])
        });
        self.gamma_derivs = Tensor::from_fn(|[i, j, k, l]| {
            0.5 * (self.g_second_derivs[[i, j, k, l]] + self.g_second_derivs[[k, i, j, l]]
                - self.g_second_derivs[[j, k, i, l]])
        });

        // Compute christoffel symbolcs of the second kind.
        self.gamma_2nd =
            Tensor::from_fn(|[i, j, k]| space.sum(|[m]| self.ginv[[i, m]] * self.gamma[[m, j, k]]));

        self.gamma_2nd_derivs = Tensor::from_fn(|[i, j, k, l]| {
            space.sum(|[m]| {
                self.ginv[[i, m]] * self.gamma_derivs[[m, j, k, l]]
                    + self.ginv_derivs[[i, m, l]] * self.gamma[[m, j, k]]
            })
        });
    }

    /// Returns the value of the metric.
    pub fn value(&self) -> &Matrix<N> {
        &self.g
    }

    /// Returns partial derivatives for each metric component.
    pub fn derivs(&self) -> &Tensor3<N> {
        &self.g_derivs
    }

    /// Returns second derivatives for each metric component.
    pub fn second_derivs(&self) -> &Tensor4<N> {
        &self.g_second_derivs
    }

    /// Returns the inverse of the metric.
    pub fn inv(&self) -> &Matrix<N> {
        &self.ginv
    }

    /// Derivatives of the inverse metric.
    pub fn inv_derivs(&self) -> &Tensor3<N> {
        &self.ginv_derivs
    }

    /// Determinate of the metric.
    pub fn det(&self) -> f64 {
        self.gdet
    }

    /// Partial derivatives of the determinate of the metric.
    pub fn det_derivs(&self) -> &Vector<N> {
        &self.gdet_derivs
    }

    /// Returns christoffel symbols of the first kind Γₐᵦᵧ.
    pub fn christoffel(&self) -> &Tensor3<N> {
        &self.gamma
    }

    /// Returns chirstoffel symbolcs of the second kind Γᵃᵦᵧ.
    pub fn christoffel_2nd(&self) -> &Tensor3<N> {
        &self.gamma_2nd
    }

    // Computes killing's equation 𝓛ₓgₐᵦ for the given vector field X.
    pub fn killing(&self, vector: VectorFieldC1<N>) -> Matrix<N> {
        MatrixFieldC2 {
            value: self.g,
            derivs: self.g_derivs,
            second_derivs: self.g_second_derivs,
        }
        .lie_derivative(vector)
    }

    /// Computes the ricci tensor for the given metric.
    pub fn ricci(&self) -> Matrix<N> {
        let s = Space::<N>;

        Tensor::from_fn(|[i, j]| {
            let term1 = s.sum(|[m]| {
                self.gamma_2nd_derivs[[m, i, j, m]] - self.gamma_2nd_derivs[[m, m, i, j]]
            });
            let term2 = s.sum(|[m, n]| {
                self.gamma_2nd[[m, m, n]] * self.gamma_2nd[[n, i, j]]
                    - self.gamma_2nd[[m, i, n]] * self.gamma_2nd[[n, m, j]]
            });
            term1 + term2
        })
    }

    /// Raises the first index of the given tensor.
    pub fn raise<const RANK: usize, L: TensorLayout<N, RANK>>(
        &self,
        tensor: Tensor<N, RANK, L>,
    ) -> Tensor<N, RANK, L> {
        Tensor::from_fn(|mut index| {
            let mut result = 0.0;

            let i = index[0];

            for m in 0..N {
                index[0] = m;
                result += self.ginv[[i, m]] * tensor[index];
            }

            result
        })
    }

    /// Computes the trace of some tensor Tₐᵦ by contracting it with the inverse metric.
    /// `T = gᵃᵝ Tₐᵦ`.
    pub fn cotrace<L: TensorLayout<N, 2>>(&self, tensor: Tensor<N, 2, L>) -> f64 {
        let s = Space::<N>;
        s.sum(|[a, b]| self.ginv[[a, b]] * tensor[[a, b]])
    }

    /// Computes the covariant derivative of a tensor `∇ᵧTₐᵦ`.
    pub fn gradient<T: Derivatives<N>>(
        &self,
        value: &T,
        partials: &T::Derivative,
    ) -> T::Derivative {
        value.covariant_gradient(partials, self)
    }

    /// Computes the second covariant derivative of a tensor `∇ₚ∇ᵩTₐᵦ`.
    pub fn hessian<T: SecondDerivatives<N>>(
        &self,
        value: &T,
        partials: &T::Derivative,
        second_partials: &T::SecondDerivative,
    ) -> T::SecondDerivative {
        value.covariant_hessian(partials, second_partials, self)
    }
}

/// Computes 𝓛ᵪTₐ = χⁿ∂ₙTₐ + Tₙ∂ₐχⁿ
pub fn lie_derivative<const N: usize, T: Derivatives<N>>(
    value: &T,
    partials: &T::Derivative,
    flow: &Vector<N>,
    flow_partials: &Matrix<N>,
) -> T {
    value.lie_derivative(partials, flow, flow_partials)
}

// *****************************
// Derivative Impls ************
// *****************************

pub trait Derivatives<const N: usize> {
    type Derivative;

    fn covariant_gradient(
        &self,
        partials: &Self::Derivative,
        metric: &Metric<N>,
    ) -> Self::Derivative;

    fn lie_derivative(
        &self,
        partials: &Self::Derivative,
        flow: &Vector<N>,
        flow_partials: &Matrix<N>,
    ) -> Self;
}

pub trait SecondDerivatives<const N: usize> {
    type Derivative;
    type SecondDerivative;

    fn covariant_hessian(
        &self,
        partials: &Self::Derivative,
        second_partials: &Self::SecondDerivative,
        metric: &Metric<N>,
    ) -> Self::SecondDerivative;
}

impl<const N: usize> Derivatives<N> for f64 {
    type Derivative = Vector<N>;

    fn covariant_gradient(
        &self,
        partials: &Self::Derivative,
        _metric: &Metric<N>,
    ) -> Self::Derivative {
        partials.clone()
    }

    fn lie_derivative(
        &self,
        partials: &Self::Derivative,
        flow: &Vector<N>,
        _flow_partials: &Matrix<N>,
    ) -> Self {
        let s = Space::<N>;
        s.sum(|[m]| flow[m] * partials[m])
    }
}

impl<const N: usize> Derivatives<N> for Vector<N> {
    type Derivative = Matrix<N>;

    fn covariant_gradient(
        &self,
        partials: &Self::Derivative,
        metric: &Metric<N>,
    ) -> Self::Derivative {
        let s = Space::<N>;
        Matrix::from_fn(|[i, j]| {
            partials[[i, j]] - s.sum(|[m]| metric.christoffel_2nd()[[m, i, j]] * self[m])
        })
    }

    fn lie_derivative(
        &self,
        partials: &Self::Derivative,
        flow: &Vector<N>,
        flow_partials: &Matrix<N>,
    ) -> Self {
        let s = Space::<N>;
        s.vector(|i| s.sum(|[m]| flow[m] * partials[[i, m]] + self[m] * flow_partials[[m, i]]))
    }
}

impl<const N: usize> Derivatives<N> for Matrix<N> {
    type Derivative = Tensor3<N>;

    fn covariant_gradient(
        &self,
        partials: &Self::Derivative,
        metric: &Metric<N>,
    ) -> Self::Derivative {
        let s = Space::<N>;
        Tensor3::from_fn(|[i, j, k]| {
            partials[[i, j, k]]
                - s.sum(|[m]| metric.christoffel_2nd()[[m, i, k]] * self[[m, j]])
                - s.sum(|[m]| metric.christoffel_2nd()[[m, j, k]] * self[[i, m]])
        })
    }

    fn lie_derivative(
        &self,
        partials: &Self::Derivative,
        flow: &Vector<N>,
        flow_partials: &Matrix<N>,
    ) -> Self {
        let s = Space::<N>;
        s.matrix(|i, j| {
            s.sum(|[m]| {
                flow[m] * partials[[i, j, m]]
                    + self[[m, j]] * flow_partials[[m, i]]
                    + self[[i, m]] * flow_partials[[m, j]]
            })
        })
    }
}

impl<const N: usize> SecondDerivatives<N> for f64 {
    type Derivative = Vector<N>;
    type SecondDerivative = Matrix<N>;

    fn covariant_hessian(
        &self,
        partials: &Self::Derivative,
        second_partials: &Self::SecondDerivative,
        metric: &Metric<N>,
    ) -> Self::SecondDerivative {
        partials.covariant_gradient(second_partials, metric)
    }
}
