use crate::{
    Static, Tensor, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, TensorFieldC1, TensorFieldC2,
    TensorIndex, TensorProd, TensorRank,
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
        result.gamma = Tensor3::from_fn(|[i, j, k]| {
            0.5 * (g.derivs[[i, j, k]] + g.derivs[[k, i, j]] - g.derivs[[j, k, i]])
        });
        result.gamma_derivs = Tensor4::from_fn(|[i, j, k, l]| {
            g.second_derivs[[i, j, k, l]] + g.second_derivs[[k, i, j, l]]
                - g.second_derivs[[j, k, i, l]]
        });

        // Compute christoffel symbolcs of the second kind.
        result.gamma_2nd =
            Tensor3::from_contract(|[i, j, k], [m]| result.ginv[[i, m]] * result.gamma[[m, j, k]]);
        result.gamma_2nd_derivs = Tensor4::from_contract(|[i, j, k, l], [m]| {
            result.ginv[[i, m]] * result.gamma_derivs[[m, j, k, l]]
                + result.ginv_derivs[[i, m, l]] * result.gamma[[m, j, k]]
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

    pub fn det(&self) -> f64 {
        self.gdet
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
}

pub struct AxisymmetricScale {
    lam_regular_co: Tensor1<2>,
    lam_regular_con: Tensor1<2>,
    lam_hess: Tensor2<2>,
    on_axis: bool,
}

impl AxisymmetricScale {
    pub fn new(seed: TensorFieldC2<2, Static<0>>, metric: &Metric<2>, pos: [f64; 2]) -> Self {
        let on_axis = pos[0].abs() <= 10e-10;

        let s = seed.value[[]];
        let s_derivs = seed.derivs;
        let s_second_derivs = seed.second_derivs;

        // Plus 1/r on axis for r component
        let mut lam_reg_co = Tensor::zeros();
        // Plus 1/(r * grr) on axis for r component
        let mut lam_reg_con = Tensor::zeros();
        // Fully regular
        let mut lam_hess = Tensor::zeros();

        {
            let g_derivs_term =
                Tensor1::<2>::from_fn(|[i]| metric.g_derivs[[0, 0, i]] / metric.g[[0, 0]]);

            let g_second_derivs_term = Tensor2::<2>::from_fn(|[i, j]| {
                metric.second_derivs()[[0, 0, i, j]] / metric.value()[[0, 0]]
                    - metric.derivs()[[0, 0, i]] * metric.derivs()[[0, 0, j]]
                        / (metric.value()[[0, 0]] * metric.value()[[0, 0]])
            });

            // Decompose lam_r into a regular part and an Order(1/r) part.
            let lam_r = s + pos[0] * s_derivs[[0]] + 0.5 * g_derivs_term[[0]]; // + 1.0 / pos[0]
            let lam_z = pos[0] * s_derivs[[1]] + 0.5 * g_derivs_term[[1]];

            lam_reg_co[[0]] = lam_r;
            lam_reg_co[[1]] = lam_z;

            lam_reg_con[[0]] = metric.inv()[[0, 0]] * lam_r + metric.inv()[[0, 1]] * lam_z;
            lam_reg_con[[1]] = metric.inv()[[1, 0]] * lam_r + metric.inv()[[1, 1]] * lam_z;

            if !on_axis {
                lam_reg_co[[0]] += 1.0 / pos[0];

                lam_reg_con[[0]] += metric.inv()[[0, 0]] / pos[0];
                lam_reg_con[[1]] += metric.inv()[[1, 0]] / pos[0];
            } else {
                lam_reg_con[[1]] += -metric.derivs()[[0, 1, 0]] / metric.det();
            }

            let mut gamma_regular = Tensor2::<2>::from_contract(|[i, j], [m]| {
                lam_reg_con[[m]] * metric.gamma[[m, i, j]]
            });

            if on_axis {
                gamma_regular[[0, 0]] +=
                    0.5 * metric.second_derivs()[[0, 0, 0, 0]] / metric.value()[[0, 0]];
                gamma_regular[[0, 1]] += 0.0; // + 0.5 * g_par[0][0][1] / (pos[0] * g[0][0])
                gamma_regular[[1, 1]] += 0.5
                    * (2.0 * metric.second_derivs()[[0, 1, 0, 1]]
                        - metric.second_derivs()[[1, 1, 0, 0]])
                    / metric.value()[[0, 0]];
            }

            let lam_rr = {
                // Plus a -1/r^2 term that gets cancelled by lam_r * lam_r
                let term1 = 2.0 * s_derivs[[0]]
                    + pos[0] * s_second_derivs[[0, 0]]
                    + 0.5 * g_second_derivs_term[[0, 0]]; // -1.0 / pos[0].powi(2)
                let term2 = lam_r * lam_r; // + 1.0 / pos[0].powi(2)
                let term3 = if on_axis {
                    // Use lhopital's rule to compute on axis lam_r / r and lam_z / r on axis.
                    let lam_r_lhopital = 2.0 * s_derivs[[0]]
                        + 0.5 * metric.second_derivs()[[0, 0, 0, 0]] / metric.value()[[0, 0]];
                    2.0 * lam_r_lhopital
                } else {
                    2.0 * lam_r / pos[0]
                };

                term1 + term2 + term3 - gamma_regular[[0, 0]]
            };

            let lam_rz = {
                let term1: f64 = s_derivs[[1]]
                    + pos[0] * s_second_derivs[[0, 1]]
                    + 0.5 * g_second_derivs_term[[0, 1]];
                let term2 = lam_r * lam_z;
                let term3 = if on_axis {
                    s_derivs[[1]] // + 0.5 * g_par[0][0][1] / (g[0][0] * pos[0]);
                } else {
                    lam_z / pos[0]
                };
                term1 + term2 + term3 - gamma_regular[[0, 1]]
            };

            let lam_zz = pos[0] * s_second_derivs[[1, 1]]
                + 0.5 * g_second_derivs_term[[1, 1]]
                + lam_z * lam_z
                - gamma_regular[[1, 1]];

            lam_hess[[0, 0]] = lam_rr;
            lam_hess[[0, 1]] = lam_rz;
            lam_hess[[1, 0]] = lam_rz;
            lam_hess[[1, 1]] = lam_zz;
        }

        Self {
            lam_regular_co: lam_reg_co,
            lam_regular_con: lam_reg_con,
            lam_hess,
            on_axis,
        }
    }
}
