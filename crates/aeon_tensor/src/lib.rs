extern crate self as aeon_tensor;

// use paste::paste;
use std::ops::{Add, Index, IndexMut, Mul};

// pub trait TensorLayout {
//     fn dim(&self) -> usize;
//     fn rank(&self) -> usize;

//     type Idx: AsMut<[usize]> + AsRef<[usize]> + Clone + Copy;

//     fn index(&self, indices: Self::Idx) -> &f64;
//     fn index_mut(&mut self, indices: Self::Idx) -> &mut f64;

//     fn indices(&self) -> impl Iterator<Item = Self::Idx>;
// }

pub trait TensorRank {
    type Idx: AsMut<[usize]> + AsRef<[usize]> + Clone + Copy;
    type Storage: Clone;

    fn dim(storage: &Self::Storage) -> usize;
    fn rank(storage: &Self::Storage) -> usize;

    fn index(storage: &Self::Storage, indices: Self::Idx) -> &f64;
    fn index_mut(storage: &mut Self::Storage, indices: Self::Idx) -> &mut f64;

    fn indices(storage: &Self::Storage) -> impl Iterator<Item = Self::Idx> + 'static;
}

pub trait TensorInit: TensorRank {
    fn zeros() -> Self::Storage;

    fn from_fn<F: Fn(Self::Idx) -> f64>(f: F) -> Self::Storage {
        let mut storage = Self::zeros();

        for indices in Self::indices(&storage) {
            *Self::index_mut(&mut storage, indices) = f(indices);
        }

        storage
    }
}

pub struct Static<const N: usize, const RANK: usize>;

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

pub struct StaticIndicesIter<const N: usize, const RANK: usize> {
    cursor: [usize; RANK],
}

impl<const N: usize, const RANK: usize> Iterator for StaticIndicesIter<N, RANK> {
    type Item = [usize; RANK];

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor[RANK - 1] >= N {
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

macro_rules! impl_static_rank {
    ($rank:literal | $($i:literal)+) => {
        impl<const N: usize> TensorRank for Static<N, $rank>  {
            type Storage = StaticStorageTypeRec!($($i)+);
            type Idx = [usize; $rank];

            fn dim(_storage: &Self::Storage) -> usize {
                N
            }

            fn rank(_storage: &Self::Storage) -> usize {
                $rank
            }

            fn index(storage: &Self::Storage, indices: [usize; $rank]) -> &f64 {
                &storage$([indices[$i]])+
            }

            fn index_mut(storage: &mut Self::Storage, indices: [usize; $rank]) -> &mut f64 {
                &mut storage$([indices[$i]])+
            }


            fn indices(_storage: &Self::Storage) -> impl Iterator<Item = Self::Idx> + 'static {
                StaticIndicesIter::<N, $rank> {
                    cursor: [0; $rank],
                }
            }
        }

        impl<const N: usize> TensorInit for Static<N, $rank> {
            fn zeros() -> Self::Storage {
                static_storage_zeros! { $($i)+ }
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

// macro_rules! impl_static_prod {
//     ($($i:literal)+ | $($j:literal)+) => {

//         paste! {
//             const [<I_RANK_ $($i)+ _ $($j)+>]: usize = ::aeon_tensor::count_repetitions!($($i)+);
//             const [<J_RANK_ $($i)+ _ $($j)+>]: usize = ::aeon_tensor::count_repetitions!($($j)+);
//             const [<K_RANK_ $($i)+ _ $($j)+>]: usize =  [<I_RANK_ $($i)+ _ $($j)+>] +  [<J_RANK_ $($i)+ _ $($j)+>];

//             impl<const N: usize> TensorProd<Static<N, [<J_RANK_ $($i)+ _ $($j)+>]>> for Static<N, [<I_RANK_ $($i)+ _ $($j)+>]> {
//                 type Result = Static<N, [<K_RANK_ $($i)+ _ $($j)+>]>;

//                 fn index_prod(
//                     indices: <Self::Result as TensorRank>::Idx,
//                 ) -> (<Self as TensorRank>::Idx, <Static<N, [<J_RANK_ $($i)+ _ $($j)+>]> as TensorRank>::Idx) {
//                     const I_RANK: usize = [<I_RANK_ $($i)+ _ $($j)+>];
//                     ([$(indices[$i],)+], [$(indices[I_RANK + $j],)+])
//                 }
//             }
//         }

//     };
// }

// macro_rules! impl_static_prods {
//     ($($i:literal)+ | $head:literal $($rest:literal)+) => {
//         impl_static_prods! { $($i)+ | $head | $($rest)+ }
//     };
//     ($($i:literal)+ | $head:literal) => {
//         impl_static_prod! { $($i)+ | $head }
//     };

//     ($($i:literal)+ | $($front:literal)* | $current:literal $($back:literal)+) => {
//         impl_static_prods! { $($i)+ | $($front)* $current | $($back)+}
//     };
//     ($($i:literal)+ | $($front:literal)* | $tail:literal) => {
//         impl_static_prod! { $($i)+ | $($front)+ $tail }
//         impl_static_prods! { $($i)+ | $($front)+ }
//     };
//     ($($i:literal)+ |)  => {};
// }

// impl_static_prods! {0 | 0 1 2 3 4}
// impl_static_prods! {0 1 | 0 1 2 3}
// impl_static_prods! {0 1 2 | 0 1 2}
// impl_static_prods! {0 1 2 3 | 0 1}
// impl_static_prods! {0 1 2 3 4 | 0}

pub struct Tensor<R: TensorRank>(R::Storage);

impl<R: TensorRank> Clone for Tensor<R> {
    fn clone(&self) -> Self {
        Self::from_parts(self.0.clone())
    }
}

impl<R: TensorRank> Tensor<R> {
    pub fn from_parts(storage: R::Storage) -> Self {
        Tensor(storage)
    }

    pub fn into_storage(self) -> R::Storage {
        self.0
    }
}

impl<R: TensorInit> Tensor<R> {
    pub fn from_fn<F: Fn(R::Idx) -> f64>(f: F) -> Self {
        Self::from_parts(R::from_fn(f))
    }

    pub fn zeros() -> Self {
        Self::from_parts(R::zeros())
    }
}

impl<R: TensorRank> Index<R::Idx> for Tensor<R> {
    type Output = f64;

    fn index(&self, indices: R::Idx) -> &Self::Output {
        R::index(&self.0, indices)
    }
}

impl<R: TensorRank> IndexMut<R::Idx> for Tensor<R> {
    fn index_mut(&mut self, indices: R::Idx) -> &mut Self::Output {
        R::index_mut(&mut self.0, indices)
    }
}

impl<R: TensorInit> Add<Tensor<R>> for Tensor<R> {
    type Output = Tensor<R>;

    fn add(self, rhs: Tensor<R>) -> Self::Output {
        Tensor::from_fn(|indices| self[indices] + rhs[indices])
    }
}

impl<S: TensorInit> Mul<f64> for Tensor<S> {
    type Output = Tensor<S>;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor::from_fn(|indices| self[indices] * rhs)
    }
}

impl<S: TensorInit> Mul<Tensor<S>> for f64 {
    type Output = Tensor<S>;

    fn mul(self, rhs: Tensor<S>) -> Self::Output {
        Tensor::from_fn(|indices| rhs[indices] * self)
    }
}

pub trait TensorProd<Other: TensorRank>: TensorInit {
    type Result: TensorInit;

    fn index_prod(
        indices: <Self::Result as TensorRank>::Idx,
    ) -> (<Self as TensorRank>::Idx, Other::Idx);
}

impl<Other: TensorRank, R: TensorProd<Other>> Mul<Tensor<Other>> for Tensor<R> {
    type Output = Tensor<R::Result>;

    fn mul(self, rhs: Tensor<Other>) -> Self::Output {
        Tensor::<R::Result>::from_fn(|indices| {
            let (l, r) = R::index_prod(indices);
            self[l] * rhs[r]
        })
    }
}

pub trait TensorContract: TensorRank {
    type Result: TensorRank;

    fn index_contract<const I: usize, const J: usize>(
        indices: <Self::Result as TensorRank>::Idx,
        val: usize,
    ) -> <Self as TensorRank>::Idx;
}

pub type Tensor1<const N: usize> = Tensor<Static<N, 1>>;
pub type Tensor2<const N: usize> = Tensor<Static<N, 2>>;
pub type Tensor3<const N: usize> = Tensor<Static<N, 3>>;
pub type Tensor4<const N: usize> = Tensor<Static<N, 4>>;

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
