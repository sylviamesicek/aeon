extern crate self as aeon_tensor;

use paste::paste;
use std::{
    iter::once,
    ops::{Add, Index, IndexMut, Mul, Sub},
};

pub mod axisymmetry;
mod field;
mod metric;

pub use field::{lie_derivative, TensorFieldC0, TensorFieldC1, TensorFieldC2};
pub use metric::Metric;

/// An index that can be used for a tensor. Essentially types of the form `[usize; R]` for some rank
/// `R`.
pub trait TensorIndex<const N: usize>: AsMut<[usize]> + AsRef<[usize]> + Clone + Copy {
    const RANK: usize;

    fn enumerate() -> impl Iterator<Item = Self> + 'static;
}

impl<const N: usize, const RANK: usize> TensorIndex<N> for [usize; RANK] {
    const RANK: usize = RANK;

    fn enumerate() -> impl Iterator<Item = Self> + 'static {
        struct TensorIndexIter<const N: usize, const RANK: usize> {
            cursor: [usize; RANK],
        }

        impl<const N: usize, const RANK: usize> Iterator for TensorIndexIter<N, RANK> {
            type Item = [usize; RANK];

            fn next(&mut self) -> Option<Self::Item> {
                if RANK == 0 || self.cursor[RANK - 1] >= N {
                    return None;
                }

                let cursor = self.cursor;

                for rank in 0..RANK {
                    if self.cursor[rank] >= N {
                        self.cursor[rank] = 0;
                    } else {
                        self.cursor[rank] += 1;
                    }
                }

                Some(cursor)
            }
        }

        if const { RANK == 0 } {
            once([0; RANK])
                .skip(0)
                .chain(TensorIndexIter::<N, RANK> { cursor: [0; RANK] })
        } else {
            once([0; RANK])
                .skip(1)
                .chain(TensorIndexIter::<N, RANK> { cursor: [0; RANK] })
        }
    }
}

pub trait TensorRank<const N: usize>: Clone {
    type Idx: TensorIndex<N>;
    type Storage: Sized + Clone;

    fn index(storage: &Self::Storage, indices: Self::Idx) -> &f64;
    fn index_mut(storage: &mut Self::Storage, indices: Self::Idx) -> &mut f64;

    fn zeros() -> Self::Storage;
    fn from_fn<F: Fn(Self::Idx) -> f64>(f: F) -> Self::Storage {
        let mut storage = Self::zeros();

        for indices in Self::Idx::enumerate() {
            *Self::index_mut(&mut storage, indices) = f(indices);
        }

        storage
    }
}

pub struct Tensor<const N: usize, R: TensorRank<N>>(R::Storage);

impl<const N: usize, R: TensorRank<N>> Clone for Tensor<N, R> {
    fn clone(&self) -> Self {
        Self::from_storage(self.0.clone())
    }
}

impl<const N: usize, R: TensorRank<N>> Default for Tensor<N, R> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<const N: usize, R: TensorRank<N>> Tensor<N, R> {
    pub fn from_storage(storage: R::Storage) -> Self {
        Tensor(storage)
    }

    pub fn into_storage(self) -> R::Storage {
        self.0
    }

    pub fn zeros() -> Self {
        Self::from_storage(R::zeros())
    }

    pub fn from_fn<F: Fn(R::Idx) -> f64>(f: F) -> Self {
        Self::from_storage(R::from_fn(f))
    }
}

impl<const N: usize, R: TensorRank<N>> Index<R::Idx> for Tensor<N, R> {
    type Output = f64;

    fn index(&self, indices: R::Idx) -> &Self::Output {
        R::index(&self.0, indices)
    }
}

impl<const N: usize, R: TensorRank<N>> IndexMut<R::Idx> for Tensor<N, R> {
    fn index_mut(&mut self, indices: R::Idx) -> &mut Self::Output {
        R::index_mut(&mut self.0, indices)
    }
}

impl<const N: usize, R: TensorRank<N>> Add<Tensor<N, R>> for Tensor<N, R> {
    type Output = Tensor<N, R>;

    fn add(self, rhs: Tensor<N, R>) -> Self::Output {
        Tensor::from_fn(|indices| self[indices] + rhs[indices])
    }
}

impl<const N: usize, R: TensorRank<N>> Sub<Tensor<N, R>> for Tensor<N, R> {
    type Output = Tensor<N, R>;

    fn sub(self, rhs: Tensor<N, R>) -> Self::Output {
        Tensor::from_fn(|indices| self[indices] - rhs[indices])
    }
}

impl<const N: usize, R: TensorRank<N>> Mul<f64> for Tensor<N, R> {
    type Output = Tensor<N, R>;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor::from_fn(|indices| self[indices] * rhs)
    }
}

impl<const N: usize, R: TensorRank<N>> Mul<Tensor<N, R>> for f64 {
    type Output = Tensor<N, R>;

    fn mul(self, rhs: Tensor<N, R>) -> Self::Output {
        Tensor::from_fn(|indices| rhs[indices] * self)
    }
}

// *******************************
// Tensor Products ***************
// *******************************

pub trait TensorProd<const N: usize, Other: TensorRank<N>>: TensorRank<N> {
    type Result: TensorRank<N>;

    fn index_split(indices: <Self::Result as TensorRank<N>>::Idx) -> (Self::Idx, Other::Idx);
    fn index_combine(this: Self::Idx, other: Other::Idx) -> <Self::Result as TensorRank<N>>::Idx;
}

impl<const N: usize, A: TensorRank<N>, B: TensorRank<N>> Mul<Tensor<N, B>> for Tensor<N, A>
where
    A: TensorProd<N, B>,
{
    type Output = Tensor<N, A::Result>;

    fn mul(self, rhs: Tensor<N, B>) -> Self::Output {
        Tensor::from_fn(|indices| {
            let (l, r) = A::index_split(indices);
            self[l] * rhs[r]
        })
    }
}

// ************************************
// Static tensor rank *****************
// ************************************

#[derive(Clone)]
pub struct Static<const RANK: usize>;

impl<const N: usize> TensorRank<N> for Static<0> {
    type Storage = f64;
    type Idx = [usize; 0];

    fn index(storage: &Self::Storage, _indices: [usize; 0]) -> &f64 {
        storage
    }

    fn index_mut(storage: &mut Self::Storage, _indices: [usize; 0]) -> &mut f64 {
        storage
    }

    fn zeros() -> Self::Storage {
        0.0
    }
}

macro_rules! StaticStorageTypeRec {
    ($head:literal $($rest:literal)*) => {
        [StaticStorageTypeRec!($($rest)*);N]
    };
    () => {
        f64
    };
}

macro_rules! static_storage_zeros {
    ($head:literal $($rest:literal)*) => {
        [static_storage_zeros!($($rest)*);N]
    };
    () => {
        0.0
    };
}

macro_rules! impl_static_rank {
    ($rank:literal | $($i:literal)*) => {
        impl<const N: usize> TensorRank<N> for Static<$rank>  {
            type Storage = StaticStorageTypeRec!($($i)*);
            type Idx = [usize; $rank];


            fn index(storage: &Self::Storage, indices: [usize; $rank]) -> &f64 {
                &storage$([indices[$i]])*
            }

            fn index_mut(storage: &mut Self::Storage, indices: [usize; $rank]) -> &mut f64 {
                &mut storage$([indices[$i]])*
            }

            fn zeros() -> Self::Storage {
                static_storage_zeros! { $($i)* }
            }
        }
    };
}

impl_static_rank! { 1 | 0 }
impl_static_rank! { 2 | 0 1 }
impl_static_rank! { 3 | 0 1 2 }
impl_static_rank! { 4 | 0 1 2 3 }
impl_static_rank! { 5 | 0 1 2 3 4 }
impl_static_rank! { 6 | 0 1 2 3 4 5 }

impl<const N: usize, Other: TensorRank<N>> TensorProd<N, Other> for Static<0> {
    type Result = Other;

    fn index_split(
        indices: <Self::Result as TensorRank<N>>::Idx,
    ) -> (Self::Idx, <Other as TensorRank<N>>::Idx) {
        ([], indices)
    }

    fn index_combine(
        _this: Self::Idx,
        other: <Other as TensorRank<N>>::Idx,
    ) -> <Self::Result as TensorRank<N>>::Idx {
        other
    }
}

macro_rules! impl_static_prod {
    ($($i:literal)+ | $($j:literal)+) => {

        paste! {
            const [<I_RANK_ $($i)+ _ $($j)+>]: usize = ::aeon_tensor::count_repetitions!($($i)+);
            const [<J_RANK_ $($i)+ _ $($j)+>]: usize = ::aeon_tensor::count_repetitions!($($j)+);
            const [<K_RANK_ $($i)+ _ $($j)+>]: usize = [<I_RANK_ $($i)+ _ $($j)+>] +  [<J_RANK_ $($i)+ _ $($j)+>];

            impl<const N: usize> TensorProd<N, Static<[<J_RANK_ $($i)+ _ $($j)+>]>> for Static<[<I_RANK_ $($i)+ _ $($j)+>]> {
                type Result = Static<[<K_RANK_ $($i)+ _ $($j)+>]>;

                fn index_split(
                    indices: [usize; [<K_RANK_ $($i)+ _ $($j)+>]],
                ) -> ([usize; [<I_RANK_ $($i)+ _ $($j)+>]], [usize; [<J_RANK_ $($i)+ _ $($j)+>]]) {
                    const I_RANK: usize = [<I_RANK_ $($i)+ _ $($j)+>];
                    ([$(indices[$i],)+], [$(indices[I_RANK + $j],)+])
                }

                fn index_combine(this: [usize; [<I_RANK_ $($i)+ _ $($j)+>]], other: [usize; [<J_RANK_ $($i)+ _ $($j)+>]]) -> [usize; [<K_RANK_ $($i)+ _ $($j)+>]] {
                    [$(this[$i],)+ $(other[$j]),+]
                }
            }
        }

    };
}

macro_rules! impl_static_prods {
    ($($i:literal)+ | $head:literal $($rest:literal)+) => {
        impl_static_prods! { $($i)+ | $head | $($rest)+ }
    };
    ($($i:literal)+ | $head:literal) => {
        impl_static_prod! { $($i)+ | $head }
    };

    ($($i:literal)+ | $($front:literal)* | $current:literal $($back:literal)+) => {
        impl_static_prods! { $($i)+ | $($front)* $current | $($back)+}
    };
    ($($i:literal)+ | $($front:literal)* | $tail:literal) => {
        impl_static_prod! { $($i)+ | $($front)+ $tail }
        impl_static_prods! { $($i)+ | $($front)+ }
    };
    ($($i:literal)+ |)  => {};
}

impl_static_prods! {0 | 0 1 2 3 4}
impl_static_prods! {0 1 | 0 1 2 3}
impl_static_prods! {0 1 2 | 0 1 2}
impl_static_prods! {0 1 2 3 | 0 1}
impl_static_prods! {0 1 2 3 4 | 0}

pub type Tensor0<const N: usize> = Tensor<N, Static<0>>;
pub type Tensor1<const N: usize> = Tensor<N, Static<1>>;
pub type Tensor2<const N: usize> = Tensor<N, Static<2>>;
pub type Tensor3<const N: usize> = Tensor<N, Static<3>>;
pub type Tensor4<const N: usize> = Tensor<N, Static<4>>;

impl<const N: usize, R: TensorRank<N>> Tensor<N, R> {
    pub fn contract<Output: TensorRank<N>, const O: usize>(
        &self,
        f: impl Fn(Output::Idx, [usize; O]) -> R::Idx,
    ) -> Tensor<N, Output> {
        Tensor::from_fn(|index| {
            let mut result = 0.0;

            for sum in <[usize; O] as TensorIndex<N>>::enumerate() {
                let r = f(index, sum);
                result += self[r];
            }

            result
        })
    }

    /// Forms a tensor of rank `R` by contracting over indices of the form `[N, O]`.
    pub fn from_contract<const O: usize>(f: impl Fn(R::Idx, [usize; O]) -> f64) -> Self {
        Self::from_fn(|index| {
            let mut result = 0.0;

            for sum in <[usize; O] as TensorIndex<N>>::enumerate() {
                result += f(index, sum);
            }

            result
        })
    }
}

pub struct Space<const N: usize>;

impl<const N: usize> Space<N> {
    pub fn tensor<const R: usize>(&self, f: impl Fn([usize; R]) -> f64) -> Tensor<N, Static<R>>
    where
        Static<R>: TensorRank<N, Idx = [usize; R]>,
    {
        Tensor::from_fn(|index| f(index))
    }

    pub fn vector(&self, f: impl Fn(usize) -> f64) -> Tensor<N, Static<1>> {
        Tensor::from_fn(|[i]: [usize; 1]| f(i))
    }

    pub fn sum<const R: usize>(&self, f: impl Fn([usize; R]) -> f64) -> f64 {
        let mut result = 0.0;

        for i in <[usize; R] as TensorIndex<N>>::enumerate() {
            result += f(i);
        }
        result
    }
}

// ************************************
// Helper Macros **********************
// ************************************

#[macro_export]
macro_rules! count_repetitions {
    ($head:literal $($rest:literal)*) => {
        (1usize + count_repetitions!($($rest)*))
    };
    () => {
        0usize
    }
}
