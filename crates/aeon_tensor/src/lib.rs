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
    const DIM: usize;
    const RANK: usize;

    type Idx: AsMut<[usize]> + AsRef<[usize]> + Clone + Copy;

    fn index(&self, indices: Self::Idx) -> &f64;
    fn index_mut(&mut self, indices: Self::Idx) -> &mut f64;

    fn from_fn<F: Fn(Self::Idx) -> f64>(f: F) -> Self;
}

macro_rules! RankTypeRec {
    ($head:literal $($rest:literal)*) => {
        [RankTypeRec!($($rest)*);N]
    };
    () => {
        f64
    };
}

macro_rules! from_fn_rec {
    ($f:ident, $($indices:ident)*) => { from_fn_rec!($f, $($indices)* | $($indices)*) };
    ($f:ident, $($indices:ident)* | $head:ident $($rest:ident)*) => {
        ::std::array::from_fn::<_, N, _>(|$head| from_fn_rec!($f, $($indices)* | $($rest)*))
    };
    ($f:ident, $($indices:ident)* |) => {
        $f([$($indices,)*])
    };
}

macro_rules! impl_rank {
    ($rank:literal | $($i:literal)+ | $($name:ident)+) => {
        impl<const N: usize> TensorRank for RankTypeRec! { $($i)+ }  {
            const DIM: usize = N;
            const RANK: usize = $rank;

            type Idx = [usize; $rank];

            fn index(&self, indices: [usize; $rank]) -> &f64 {
                &self$([indices[$i]])+
            }

            fn index_mut(&mut self, indices: [usize; $rank]) -> &mut f64 {
                &mut self$([indices[$i]])+
            }

            fn from_fn<F: Fn([usize; $rank]) -> f64>(f: F) -> Self {
                from_fn_rec!(f, $($name)+)
            }

        }
    };
}

impl_rank! { 1 | 0 | i0 }
impl_rank! { 2 | 0 1 | i0 i1 }
impl_rank! { 3 | 0 1 2 | i0 i1 i2 }
impl_rank! { 4 | 0 1 2 3 | i0 i1 i2 i3 }
impl_rank! { 5 | 0 1 2 3 4 | i0 i1 i2 i3 i4 }
impl_rank! { 6 | 0 1 2 3 4 5 | i0 i1 i2 i3 i4 i5 }

macro_rules! impl_prod {
    ($rank:literal | $($i:literal)+ | $($j:literal)+) => {
        impl<const N: usize> TensorProd<RankTypeRec!($($j)+)> for RankTypeRec!($($i)+) {
            type Result = RankTypeRec!($($i)+ $($j)+);

            fn index_prod(
                indices: <Self::Result as TensorRank>::Idx,
            ) -> (<Self as TensorRank>::Idx, <RankTypeRec!($($j)+) as TensorRank>::Idx) {
                ([$(indices[$i],)+], [$(indices[$rank + $j],)+])
            }
        }
    };
}

macro_rules! impl_prods {
    ($rank:literal | $($i:literal)+ | $head:literal $($rest:literal)+) => {
        impl_prods! { $rank | $($i)+ | $head | $($rest)+ }
    };
    ($rank:literal | $($i:literal)+ | $head:literal) => {
        impl_prod! { $rank | $($i)+ | $head }
    };

    ($rank:literal | $($i:literal)+ | $($front:literal)* | $current:literal $($back:literal)+) => {
        impl_prods! { $rank | $($i)+ | $($front)* $current | $($back)+}
    };
    ($rank:literal | $($i:literal)+ | $($front:literal)* | $tail:literal) => {
        impl_prod! { $rank | $($i)+ | $($front)+ $tail }
        impl_prods! { $rank | $($i)+ | $($front)+ }
    };
    ($rank:literal | $($i:literal)+ |)  => {};
}

impl_prods! {1 | 0 | 0 1 2 3 4}
impl_prods! {2 | 0 1 | 0 1 2 3}
impl_prods! {3 | 0 1 2 | 0 1 2}
impl_prods! {3 | 0 1 2 3 | 0 1}
impl_prods! {3 | 0 1 2 3 4 | 0}

pub struct Tensor<S>(S);

impl<S: TensorRank> Tensor<S> {
    pub fn from_storage(storage: S) -> Self {
        Tensor(storage)
    }

    pub fn into_storage(self) -> S {
        self.0
    }

    pub fn from_fn<F: Fn(S::Idx) -> f64>(f: F) -> Self {
        Tensor(S::from_fn(f))
    }

    pub fn zeros() -> Self {
        Self::from_fn(|_| 0.0)
    }
}

impl<S: TensorRank> Index<S::Idx> for Tensor<S> {
    type Output = f64;

    fn index(&self, indices: S::Idx) -> &Self::Output {
        self.0.index(indices)
    }
}

impl<S: TensorRank> IndexMut<S::Idx> for Tensor<S> {
    fn index_mut(&mut self, indices: S::Idx) -> &mut Self::Output {
        self.0.index_mut(indices)
    }
}

impl<S: TensorRank> Add<Tensor<S>> for Tensor<S> {
    type Output = Tensor<S>;

    fn add(self, rhs: Tensor<S>) -> Self::Output {
        Tensor::from_fn(|indices| self[indices] + rhs[indices])
    }
}

impl<S: TensorRank> Mul<f64> for Tensor<S> {
    type Output = Tensor<S>;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor::from_fn(|indices| self[indices] * rhs)
    }
}

pub trait TensorProd<Other: TensorRank>: TensorRank {
    type Result: TensorRank;

    fn index_prod(
        indices: <Self::Result as TensorRank>::Idx,
    ) -> (<Self as TensorRank>::Idx, Other::Idx);
}

impl<Other: TensorRank, S: TensorProd<Other>> Mul<Tensor<Other>> for Tensor<S> {
    type Output = Tensor<S::Result>;

    fn mul(self, rhs: Tensor<Other>) -> Self::Output {
        Tensor::<S::Result>::from_fn(|indices| {
            let (l, r) = S::index_prod(indices);
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

pub type Tensor1<const N: usize> = Tensor<[f64; N]>;
pub type Tensor2<const N: usize> = Tensor<[[f64; N]; N]>;
pub type Tensor3<const N: usize> = Tensor<[[[f64; N]; N]; N]>;
pub type Tensor4<const N: usize> = Tensor<[[[[f64; N]; N]; N]; N]>;
