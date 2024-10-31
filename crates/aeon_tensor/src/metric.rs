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
            let grr_inv_derivs = Vector::from_fn(|[i]| {
                g.derivs[[1, 1, i]] / gdet - g.value[[1, 1]] / (gdet * gdet) * gdet_derivs[[i]]
            })
            .inner()
            .clone();

            let grz_inv_derivs = Vector::from_fn(|[i]| {
                -g.derivs[[0, 1, i]] / gdet + g.value[[0, 1]] / (gdet * gdet) * gdet_derivs[[i]]
            })
            .inner()
            .clone();

            let gzz_inv_derivs = Vector::from_fn(|[i]| {
                g.derivs[[0, 0, i]] / gdet - g.value[[0, 0]] / (gdet * gdet) * gdet_derivs[[i]]
            })
            .inner()
            .clone();

            [
                [grr_inv_derivs, grz_inv_derivs],
                [grz_inv_derivs, gzz_inv_derivs],
            ]
        });

        // Set fields of result
        let mut result = Self::default();

        result.g = g.value.clone();
        result.g_derivs = g.derivs.clone();
        result.g_second_derivs = g.second_derivs.clone();

        result.gdet = gdet;
        result.gdet_derivs = gdet_derivs;

        result.ginv = ginv;
        result.ginv_derivs = ginv_derivs;

        result.compute_christoffel();

        result
    }
}

impl<const N: usize> Metric<N> {
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

    pub fn value(&self) -> &Matrix<N> {
        &self.g
    }

    pub fn derivs(&self) -> &Tensor3<N> {
        &self.g_derivs
    }

    pub fn second_derivs(&self) -> &Tensor4<N> {
        &self.g_second_derivs
    }

    pub fn inv(&self) -> &Matrix<N> {
        &self.ginv
    }

    pub fn inv_derivs(&self) -> &Tensor3<N> {
        &self.ginv_derivs
    }

    pub fn det(&self) -> f64 {
        self.gdet
    }

    pub fn det_derivs(&self) -> &Vector<N> {
        &self.gdet_derivs
    }

    /// Returns christoffel symbols of the first kind.
    pub fn christoffel(&self) -> &Tensor3<N> {
        &self.gamma
    }

    pub fn christoffel_2nd(&self) -> &Tensor3<N> {
        &self.gamma_2nd
    }

    // Computes killing's equation for the given vector field.
    pub fn killing(&self, vector: VectorFieldC1<N>) -> Matrix<N> {
        MatrixFieldC2 {
            value: self.g.clone(),
            derivs: self.g_derivs.clone(),
            second_derivs: self.g_second_derivs.clone(),
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

    pub fn cotrace<L: TensorLayout<N, 2>>(&self, tensor: Tensor<N, 2, L>) -> f64 {
        let s = Space::<N>;
        s.sum(|[a, b]| self.ginv[[a, b]] * tensor[[a, b]])
    }
}
