use crate::{
    lie_derivative, Space, Static, Tensor, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4,
    TensorFieldC1, TensorFieldC2, TensorIndex, TensorProd, TensorRank,
};

#[derive(Default)]
pub struct Metric<const N: usize> {
    g: Tensor2<N>,
    g_derivs: Tensor3<N>,
    g_second_derivs: Tensor4<N>,

    gdet: f64,
    gdet_derivs: Tensor1<N>,

    ginv: Tensor2<N>,
    ginv_derivs: Tensor3<N>,

    gamma: Tensor3<N>,
    gamma_derivs: Tensor4<N>,

    gamma_2nd: Tensor3<N>,
    gamma_2nd_derivs: Tensor4<N>,
}

impl Metric<2> {
    pub fn new(g: TensorFieldC2<2, Static<2>>) -> Self {
        let space = Space::<2>;

        // Compute determinant and its derivatives.
        let gdet = g.value[[0, 0]] * g.value[[1, 1]] - g.value[[0, 1]] * g.value[[1, 0]];
        let gdet_derivs = Tensor1::from_fn(|[i]| {
            g.value[[0, 0]] * g.derivs[[1, 1, i]] + g.derivs[[0, 0, i]] * g.value[[1, 1]]
                - g.derivs[[0, 1, i]] * g.value[[1, 0]]
                - g.value[[0, 1]] * g.derivs[[1, 0, i]]
        });

        // Compute inverse and its derivatives
        let ginv = Tensor2::from_storage([
            [g.value[[1, 1]] / gdet, -g.value[[1, 0]] / gdet],
            [-g.value[[0, 1]] / gdet, g.value[[0, 0]] / gdet],
        ]);
        let ginv_derivs = Tensor3::from_storage({
            let grr_inv_derivs = Tensor1::from_fn(|[i]| {
                g.derivs[[1, 1, i]] / gdet - g.value[[1, 1]] / (gdet * gdet) * gdet_derivs[[i]]
            })
            .into_storage();

            let grz_inv_derivs = Tensor1::from_fn(|[i]| {
                -g.derivs[[0, 1, i]] / gdet + g.value[[0, 1]] / (gdet * gdet) * gdet_derivs[[i]]
            })
            .into_storage();

            let gzz_inv_derivs = Tensor1::from_fn(|[i]| {
                g.derivs[[0, 0, i]] / gdet - g.value[[0, 0]] / (gdet * gdet) * gdet_derivs[[i]]
            })
            .into_storage();

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

        // Compute christoffel symbols of the first kind.
        result.gamma = space.tensor(|[i, j, k]| {
            0.5 * (g.derivs[[i, j, k]] + g.derivs[[k, i, j]] - g.derivs[[j, k, i]])
        });
        result.gamma_derivs = space.tensor(|[i, j, k, l]| {
            g.second_derivs[[i, j, k, l]] + g.second_derivs[[k, i, j, l]]
                - g.second_derivs[[j, k, i, l]]
        });

        // Compute christoffel symbolcs of the second kind.
        result.gamma_2nd = space
            .tensor(|[i, j, k]| space.sum(|[m]| result.ginv[[i, m]] * result.gamma[[m, j, k]]));

        result.gamma_2nd_derivs = space.tensor(|[i, j, k, l]| {
            space.sum(|[m]| {
                result.ginv[[i, m]] * result.gamma_derivs[[m, j, k, l]]
                    + result.ginv_derivs[[i, m, l]] * result.gamma[[m, j, k]]
            })
        });

        result
    }
}

impl<const N: usize> Metric<N> {
    pub fn value(&self) -> &Tensor2<N> {
        &self.g
    }

    pub fn derivs(&self) -> &Tensor3<N> {
        &self.g_derivs
    }

    pub fn second_derivs(&self) -> &Tensor4<N> {
        &self.g_second_derivs
    }

    pub fn inv(&self) -> &Tensor2<N> {
        &self.ginv
    }

    pub fn inv_derivs(&self) -> &Tensor3<N> {
        &self.ginv_derivs
    }

    pub fn det(&self) -> f64 {
        self.gdet
    }

    pub fn det_derivs(&self) -> &Tensor1<N> {
        &self.gdet_derivs
    }

    /// Returns christoffel symbols of the first kind.
    pub fn christoffel(&self) -> &Tensor3<N> {
        &self.gamma
    }

    // Computes killing's equation for the given vector field.
    pub fn killing(&self, vector: TensorFieldC1<N, Static<1>>) -> Tensor2<N> {
        lie_derivative(
            vector,
            TensorFieldC2 {
                value: self.g.clone(),
                derivs: self.g_derivs.clone(),
                second_derivs: self.g_second_derivs.clone(),
            },
        )
    }

    /// Computes the ricci tensor for the given metric.
    pub fn ricci(&self) -> Tensor2<N> {
        let term1 = Tensor2::from_contract(|[i, j], [m]| {
            self.gamma_2nd_derivs[[m, i, j, m]] - self.gamma_2nd_derivs[[m, m, i, j]]
        });

        let term2 = Tensor2::from_contract(|[i, j], [m, n]| {
            self.gamma_2nd[[m, m, n]] * self.gamma_2nd[[n, i, j]]
                - self.gamma_2nd[[m, i, n]] * self.gamma_2nd[[n, m, j]]
        });

        term1 + term2
    }

    /// Computes the gradient of an arbitrary C1 tensor field.
    pub fn gradiant<R: TensorProd<N, Static<1>>>(
        &self,
        tensor: TensorFieldC1<N, R>,
    ) -> Tensor<N, R::Result> {
        let mut result = tensor.derivs.clone();

        if const { <R::Idx as TensorIndex<N>>::RANK == 0 } {
            return result;
        }

        for findex in <R::Result as TensorRank<N>>::Idx::enumerate() {
            let (index, [k]) = R::index_split(findex);

            for r in 0..<R::Idx as TensorIndex<N>>::RANK {
                // Retrieve 'r'th component of index
                let i = index.as_ref()[r];

                for m in 0..N {
                    // Build mutable version and set 'r'th component to m
                    let mut index = index;
                    index.as_mut()[r] = m;

                    // Add to result
                    result[findex] -= self.gamma_2nd[[m, i, k]] * tensor.value[index]
                }
            }
        }

        result
    }

    /// Computes the hessian of an arbiraty c2 tensor field.
    pub fn hessian<R>(
        &self,
        tensor: TensorFieldC2<N, R>,
    ) -> Tensor<N, <<R as TensorProd<N, Static<1>>>::Result as TensorProd<N, Static<1>>>::Result>
    where
        R: TensorProd<N, Static<1>>,
        <R as TensorProd<N, Static<1>>>::Result: TensorProd<N, Static<1>>,
    {
        let field = TensorFieldC1 {
            value: tensor.derivs,
            derivs: tensor.second_derivs,
        };

        let result = self.gradiant(field);

        if const { <R::Idx as TensorIndex<N>>::RANK == 0 } {
            return result;
        } else {
            panic!("Hessian function unimplemented for non rank 0 tensors");
        }
    }

    /// Computes the traces of a covariant rank 2 tensor.
    pub fn trace(&self, tensor: Tensor<N, Static<2>>) -> f64 {
        Tensor0::<N>::from_contract(|[], [i, j]| self.ginv[[i, j]] * tensor[[i, j]]).into_storage()
    }

    /// Raises the first index of the given tensor.
    pub fn raise<R: TensorRank<N>>(&self, tensor: Tensor<N, R>) -> Tensor<N, R> {
        const {
            assert!(R::Idx::RANK > 0);
        }

        let mut result = Tensor::zeros();

        for findex in R::Idx::enumerate() {
            let i = findex.as_ref()[0];

            for m in 0..N {
                let mut index = findex;
                index.as_mut()[0] = m;

                result[findex] += self.inv()[[i, m]] * tensor[index]
            }
        }

        result
    }
}
