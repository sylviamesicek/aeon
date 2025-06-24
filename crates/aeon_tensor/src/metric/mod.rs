//! Module containing common operations on manifolds with metrics.

use crate::{
    Gen, Sym, SymSym, SymVec, Tensor, TensorIndex, TensorStorageOwned, TensorStorageRef, VecSym,
    VecSymVec,
};

mod dims;

pub use dims::d2;

pub trait Space<const N: usize>: Clone + Copy {
    type VecStore: TensorStorageOwned + Default + Clone;
    type MatStore: TensorStorageOwned + Default + Clone;
    type SymStore: TensorStorageOwned + Default + Clone;
    type SymVecStore: TensorStorageOwned + Default + Clone;
    type SymSymStore: TensorStorageOwned + Default + Clone;
    type SymVecVecStore: TensorStorageOwned + Default + Clone;

    /// Sums over all indices of the given rank.
    fn sum<const R: usize>(f: impl Fn([usize; R]) -> f64) -> f64 {
        let mut result = 0.0;
        <Gen as TensorIndex<N, R>>::for_each_index(|idx| result += f(idx));
        result
    }

    /// Constructs an `N` dimensional vector.
    fn vector(f: impl Fn([usize; 1]) -> f64) -> Tensor<N, 1, Gen, Self::VecStore> {
        Tensor::from_fn(f)
    }

    /// Constructs an `N` dimensional symmetric matrix.
    fn symmetric(f: impl Fn([usize; 2]) -> f64) -> Tensor<N, 2, Sym, Self::SymStore> {
        Tensor::from_fn(f)
    }

    /// Constructs an `N` dimensional general matrix.
    fn matrix(f: impl Fn([usize; 2]) -> f64) -> Tensor<N, 2, Gen, Self::MatStore> {
        Tensor::from_fn(f)
    }
}

/// Space where all tensors are stored statically and unboxed.
#[derive(Clone, Copy)]
pub struct Static;

/// The number of components of an `n` dimensional symmetric matrix.
const fn sym(n: usize) -> usize {
    n * (n + 1) / 2
}

macro_rules! impl_space {
    ($N:literal) => {
        impl Space<$N> for Static {
            type VecStore = [f64; const { $N }];
            type MatStore = [f64; const { $N * $N }];
            type SymStore = [f64; const { sym($N) }];
            type SymVecStore = [f64; const { sym($N) * $N }];
            type SymSymStore = [f64; const { sym($N) * sym($N) }];
            type SymVecVecStore = [f64; const { sym($N) * $N * $N }];
        }
    };
}

impl_space!(1);
impl_space!(2);

/// A metric (along with first and second derivatives) defined on a point on a manifold.
pub struct Metric<const N: usize, S: Space<N>> {
    pub value: Tensor<N, 2, Sym, S::SymStore>,
    pub derivs: Tensor<N, 3, SymVec, S::SymVecStore>,
    pub derivs2: Tensor<N, 4, SymSym, S::SymSymStore>,
}

impl<const N: usize, S: Space<N>> Metric<N, S> {
    /// Constructs a new metric from the constituent components and partial derivatives.
    pub fn new(
        value: Tensor<N, 2, Sym, S::SymStore>,
        derivs: Tensor<N, 3, SymVec, S::SymVecStore>,
        derivs2: Tensor<N, 4, SymSym, S::SymSymStore>,
    ) -> Self {
        Self {
            value,
            derivs,
            derivs2,
        }
    }
}

impl<S: Space<2>> Metric<2, S> {
    /// Computes the determinate of a metric.
    pub fn det(&self) -> MetricDet<2, S> {
        let value = self.value[[0, 0]] * self.value[[1, 1]] - self.value[[1, 0]].powi(2);
        let derivs = Tensor::from_fn(|[a]| {
            self.derivs[[0, 0, a]] * self.value[[1, 1]]
                + self.value[[0, 0]] * self.derivs[[1, 1, a]]
                - 2.0 * self.value[[0, 1]] * self.derivs[[0, 1, a]]
        });

        MetricDet { value, derivs }
    }

    /// Computes the inverse of a metric.
    pub fn inv(&self, det: &MetricDet<2, S>) -> MetricInv<2, S> {
        let factor = det.value.recip();
        let factor_derivs: Tensor<2, 1, Gen, S::VecStore> =
            Tensor::from_fn(|[a]| -factor.powi(2) * det.derivs[[a]]);

        let mut value = Tensor::new();
        value[[0, 0]] = factor * self.value[[1, 1]];
        value[[1, 0]] = -factor * self.value[[1, 0]];
        debug_assert_eq!(value[[1, 0]], value[[0, 1]]);
        value[[1, 1]] = factor * self.value[[0, 0]];

        let mut derivs = Tensor::new();

        for a in 0..2 {
            derivs[[0, 0, a]] =
                factor_derivs[[a]] * self.value[[1, 1]] + factor * self.derivs[[1, 1, a]];
            derivs[[1, 0, a]] =
                -factor_derivs[[a]] * self.value[[1, 0]] - factor * self.derivs[[1, 0, a]];
            debug_assert_eq!(derivs[[0, 1, a]], derivs[[1, 0, a]]);
            derivs[[1, 1, a]] =
                factor_derivs[[a]] * self.value[[0, 0]] + factor * self.derivs[[0, 0, a]];
        }

        MetricInv { value, derivs }
    }
}

impl<const N: usize, S: Space<N>> Metric<N, S> {
    // Computes killing's equation 𝓛ₓgₐᵦ for the given vector field X.
    pub fn killing(&self, vector: &VectorC1<N, S>) -> Tensor<N, 2, Sym, S::SymStore> {
        SymmetricC1 {
            value: self.value.clone(),
            derivs: self.derivs.clone(),
        }
        .lie_derivative(vector)
    }
}

/// The determinate of a metric, along with partial derivatives.
pub struct MetricDet<const N: usize, S: Space<N>> {
    pub value: f64,
    pub derivs: Tensor<N, 1, Gen, S::VecStore>,
}

/// The inverse of a metric, along with partial derivatives.
pub struct MetricInv<const N: usize, S: Space<N>> {
    pub value: Tensor<N, 2, Sym, S::SymStore>,
    pub derivs: Tensor<N, 3, SymVec, S::SymVecStore>,
}

impl<const N: usize, S: Space<N>> MetricInv<N, S> {
    /// Computes the trace of a fully covariant 2-tensor.
    pub fn cotrace<I: TensorIndex<N, 2>, St: TensorStorageRef>(
        &self,
        matrix: &Tensor<N, 2, I, St>,
    ) -> f64 {
        S::sum(|[a, b]| self.value[[a, b]] * matrix[[a, b]])
    }

    /// Raises the first index of a general r-tensor.
    pub fn raise_first<const R: usize, I: TensorIndex<N, R>, St: TensorStorageOwned + Default>(
        &self,
        tensor: &Tensor<N, R, I, St>,
    ) -> Tensor<N, R, I, St> {
        const {
            if R == 0 {
                panic!("R must be > 0");
            }
        }

        Tensor::from_fn(|idx| {
            S::sum(|[a]| {
                let mut tidx = idx;
                tidx[0] = a;
                self.value[[idx[0], a]] * tensor[tidx]
            })
        })
    }

    /// Raises the last index of a general r-tensor.
    pub fn raise_last<const R: usize, I: TensorIndex<N, R>, St: TensorStorageOwned + Default>(
        &self,
        tensor: &Tensor<N, R, I, St>,
    ) -> Tensor<N, R, I, St> {
        const {
            if R == 0 {
                panic!("R must be > 0");
            }
        }

        Tensor::from_fn(|idx| {
            S::sum(|[a]| {
                let mut tidx = idx;
                tidx[R - 1] = a;
                self.value[[idx[R - 1], a]] * tensor[tidx]
            })
        })
    }
}

/// Christoffel connection symbols and their derivatives defined on
/// a general metric.
pub struct ChristoffelSymbol<const N: usize, S: Space<N>> {
    pub first_kind: Tensor<N, 3, VecSym, S::SymVecStore>,
    pub first_kind_derivs: Tensor<N, 4, VecSymVec, S::SymVecVecStore>,
    pub second_kind: Tensor<N, 3, VecSym, S::SymVecStore>,
    pub second_kind_derivs: Tensor<N, 4, VecSymVec, S::SymVecVecStore>,
}

impl<const N: usize, S: Space<N>> ChristoffelSymbol<N, S> {
    /// Computes the ricci tensor from the christoffel symbols.
    pub fn ricci(&self) -> Tensor<N, 2, Sym, S::SymStore> {
        Tensor::from_eq(|[i, j], [a]| {
            let term1: f64 =
                self.second_kind_derivs[[a, i, j, a]] - self.second_kind_derivs[[a, a, i, j]];
            let term2 = S::sum(|[b]| {
                self.second_kind[[a, a, b]] * self.second_kind[[b, i, j]]
                    - self.second_kind[[a, i, b]] * self.second_kind[[b, a, j]]
            });

            term1 + term2
        })
    }
}

impl<const N: usize, S: Space<N>> Metric<N, S> {
    /// Computes Christoffel_symbols for a given metric.
    pub fn christoffel_symbol(&self, inv: &MetricInv<N, S>) -> ChristoffelSymbol<N, S> {
        let first_kind = Tensor::from_fn(|[a, b, c]| {
            0.5 * (self.derivs[[a, c, b]] + self.derivs[[b, a, c]] - self.derivs[[b, c, a]])
        });
        let first_kind_derivs = Tensor::from_fn(|[a, b, c, d]| {
            0.5 * (self.derivs2[[a, c, b, d]] + self.derivs2[[b, a, c, d]]
                - self.derivs2[[b, c, a, d]])
        });

        let second_kind =
            Tensor::from_eq(|[a, b, c], [m]| inv.value[[a, m]] * first_kind[[m, b, c]]);

        let second_kind_derivs = Tensor::from_eq(|[a, b, c, d], [m]| {
            inv.derivs[[a, m, d]] * first_kind[[m, b, c]]
                + inv.value[[a, m]] * first_kind_derivs[[m, b, c, d]]
        });

        ChristoffelSymbol {
            first_kind,
            first_kind_derivs,
            second_kind,
            second_kind_derivs,
        }
    }
}

/// A C1 scalar field at a single point.
#[derive(Debug)]
pub struct ScalarC1<const N: usize, S: Space<N>> {
    pub value: f64,
    pub derivs: Tensor<N, 1, Gen, S::VecStore>,
}

impl<const N: usize, S: Space<N>> ScalarC1<N, S> {
    pub fn gradient(&self, _connect: &ChristoffelSymbol<N, S>) -> Tensor<N, 1, Gen, S::VecStore> {
        self.derivs.clone()
    }

    pub fn lie_derivative(&self, flow: &VectorC1<N, S>) -> f64 {
        S::sum(|[a]| flow.value[[a]] * self.derivs[[a]])
    }
}

impl<const N: usize, S: Space<N>> From<ScalarC2<N, S>> for ScalarC1<N, S> {
    fn from(value: ScalarC2<N, S>) -> Self {
        Self {
            value: value.value,
            derivs: value.derivs,
        }
    }
}

impl<const N: usize, S: Space<N>> Default for ScalarC1<N, S> {
    fn default() -> Self {
        Self {
            value: Default::default(),
            derivs: Default::default(),
        }
    }
}

impl<const N: usize, S: Space<N>> Clone for ScalarC1<N, S> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            derivs: self.derivs.clone(),
        }
    }
}

/// A C2 scalar field at a single point.
pub struct ScalarC2<const N: usize, S: Space<N>> {
    pub value: f64,
    pub derivs: Tensor<N, 1, Gen, S::VecStore>,
    pub derivs2: Tensor<N, 2, Sym, S::SymStore>,
}

impl<const N: usize, S: Space<N>> ScalarC2<N, S> {
    pub fn gradient(&self, _connect: &ChristoffelSymbol<N, S>) -> Tensor<N, 1, Gen, S::VecStore> {
        self.derivs.clone()
    }

    pub fn lie_derivative(&self, flow: &VectorC1<N, S>) -> f64 {
        S::sum(|[a]| flow.value[[a]] * self.derivs[[a]])
    }

    pub fn hessian(&self, connect: &ChristoffelSymbol<N, S>) -> Tensor<N, 2, Sym, S::SymStore> {
        Tensor::from_fn(|[a, b]| {
            let term1 = self.derivs2[[a, b]];
            let term2 = S::sum(|[d]| -connect.second_kind[[d, a, b]] * self.derivs[[d]]);
            term1 + term2
        })
    }
}

impl<const N: usize, S: Space<N>> Default for ScalarC2<N, S> {
    fn default() -> Self {
        Self {
            value: Default::default(),
            derivs: Default::default(),
            derivs2: Default::default(),
        }
    }
}

impl<const N: usize, S: Space<N>> Clone for ScalarC2<N, S> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            derivs: self.derivs.clone(),
            derivs2: self.derivs2.clone(),
        }
    }
}

/// A C1 vector field at a single point.
pub struct VectorC1<const N: usize, S: Space<N>> {
    pub value: Tensor<N, 1, Gen, S::VecStore>,
    pub derivs: Tensor<N, 2, Gen, S::MatStore>,
}

impl<const N: usize, S: Space<N>> VectorC1<N, S> {
    pub fn gradient(&self, connect: &ChristoffelSymbol<N, S>) -> Tensor<N, 2, Gen, S::MatStore> {
        Tensor::from_fn(|[a, c]| {
            let term1 = self.derivs[[a, c]];
            let term2 = S::sum(|[d]| -connect.second_kind[[d, a, c]] * self.value[[d]]);
            term1 + term2
        })
    }

    pub fn lie_derivative(&self, flow: &VectorC1<N, S>) -> Tensor<N, 1, Gen, S::VecStore> {
        Tensor::from_fn(|[a]| {
            S::sum(|[i]| {
                flow.value[[i]] * self.derivs[[a, i]] + flow.derivs[[i, a]] * self.value[[i]]
            })
        })
    }
}

impl<const N: usize, S: Space<N>> Default for VectorC1<N, S> {
    fn default() -> Self {
        Self {
            value: Default::default(),
            derivs: Default::default(),
        }
    }
}

impl<const N: usize, S: Space<N>> Clone for VectorC1<N, S> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            derivs: self.derivs.clone(),
        }
    }
}

/// A C1 symmetric matrix field at a single point.
pub struct SymmetricC1<const N: usize, S: Space<N>> {
    pub value: Tensor<N, 2, Sym, S::SymStore>,
    pub derivs: Tensor<N, 3, SymVec, S::SymVecStore>,
}

impl<const N: usize, S: Space<N>> SymmetricC1<N, S> {
    pub fn gradient(
        &self,
        connect: &ChristoffelSymbol<N, S>,
    ) -> Tensor<N, 3, SymVec, S::SymVecStore> {
        Tensor::from_fn(|[i, j, k]| {
            self.derivs[[i, j, k]]
                - S::sum(|[m]| connect.second_kind[[m, i, k]] * self.value[[m, j]])
                - S::sum(|[m]| connect.second_kind[[m, j, k]] * self.value[[m, i]])
        })
    }

    pub fn lie_derivative(&self, flow: &VectorC1<N, S>) -> Tensor<N, 2, Sym, S::SymStore> {
        Tensor::from_fn(|[i, j]| {
            S::sum(|[m]| {
                flow.value[[m]] * self.derivs[[i, j, m]]
                    + self.value[[m, j]] * flow.derivs[[m, i]]
                    + self.value[[i, m]] * flow.derivs[[m, j]]
            })
        })
    }
}

impl<const N: usize, S: Space<N>> Default for SymmetricC1<N, S> {
    fn default() -> Self {
        Self {
            value: Default::default(),
            derivs: Default::default(),
        }
    }
}

impl<const N: usize, S: Space<N>> Clone for SymmetricC1<N, S> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            derivs: self.derivs.clone(),
        }
    }
}
