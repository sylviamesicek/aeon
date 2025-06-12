macro_rules! impl_dim {
    ($name:ident, $N:literal) => {
        pub mod $name {
            use crate::{metric, metric::Space};
            use crate::{Gen, Sym, SymSym, SymVec, Tensor, VecSym};

            #[derive(Clone, Copy)]
            pub struct Static;

            impl metric::Space<$N> for Static {
                type VecStore = <metric::Static as metric::Space<$N>>::VecStore;
                type MatStore = <metric::Static as metric::Space<$N>>::MatStore;
                type SymStore = <metric::Static as metric::Space<$N>>::SymStore;
                type SymVecStore = <metric::Static as metric::Space<$N>>::SymVecStore;
                type SymSymStore = <metric::Static as metric::Space<$N>>::SymSymStore;
                type SymVecVecStore = <metric::Static as metric::Space<$N>>::SymVecVecStore;
            }

            pub type Metric = metric::Metric<$N, Static>;
            pub type MetricInv = metric::MetricInv<$N, Static>;
            pub type MetricDet = metric::MetricDet<$N, Static>;
            pub type ChristoffelSymbol = metric::ChristoffelSymbol<$N, Static>;

            pub type Vector = Tensor<$N, 1, Gen, <Static as Space<$N>>::VecStore>;
            pub type Symmetric = Tensor<$N, 2, Sym, <Static as Space<$N>>::SymStore>;
            pub type Matrix = Tensor<$N, 2, Gen, <Static as Space<$N>>::MatStore>;
            pub type SymmetricDeriv = Tensor<$N, 3, SymVec, <Static as Space<$N>>::SymVecStore>;
            pub type Connection = Tensor<$N, 3, VecSym, <Static as Space<$N>>::SymVecStore>;
            pub type SymmetricSymmetric = Tensor<$N, 4, SymSym, <Static as Space<$N>>::SymSymStore>;

            pub type ScalarC1 = metric::ScalarC1<$N, Static>;
            pub type ScalarC2 = metric::ScalarC2<$N, Static>;
            pub type VectorC1 = metric::VectorC1<$N, Static>;
            pub type SymmetricC1 = metric::SymmetricC1<$N, Static>;
        }
    };
}

impl_dim! { d2, 2 }
