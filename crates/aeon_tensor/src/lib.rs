#![allow(clippy::needless_range_loop)]

extern crate self as aeon_tensor;

use paste::paste;
use std::{
    array,
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};

pub mod axisymmetry;
mod field;
mod metric;

pub use field::{MatrixFieldC1, MatrixFieldC2, ScalarFieldC1, ScalarFieldC2, VectorFieldC1};
pub use metric::Metric;

pub use metric::lie_derivative;

pub fn for_each_index<const N: usize, const RANK: usize>(mut f: impl FnMut([usize; RANK])) {
    let mut cursor = [0; RANK];

    f(cursor);

    'l: loop {
        for axis in 0..RANK {
            cursor[axis] += 1;

            if cursor[axis] < N {
                f(cursor);
                continue 'l;
            }

            cursor[axis] = 0;
        }

        break;
    }
}

// *********************************
// Tensor **************************
// *********************************

#[derive(Clone, Copy, Debug)]
pub struct Tensor<const N: usize, const RANK: usize, L>(L);

// *********************************
// Tensor Layout *******************
// *********************************

pub trait TensorLayout<const N: usize, const RANK: usize>: Clone {
    fn index(&self, indices: [usize; RANK]) -> &f64;
    fn index_mut(&mut self, indices: [usize; RANK]) -> &mut f64;

    fn zeros() -> Self;
    fn from_fn<F: Fn([usize; RANK]) -> f64>(f: F) -> Self {
        let mut result = Self::zeros();
        for_each_index::<N, RANK>(|index| *result.index_mut(index) = f(index));
        result
    }
}

impl<const N: usize, const RANK: usize, L> Tensor<N, RANK, L> {
    pub fn inner(&self) -> &L {
        &self.0
    }
}

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> Tensor<N, RANK, L> {
    pub fn zeros() -> Self {
        Self(L::zeros())
    }

    pub fn from_fn<F: Fn([usize; RANK]) -> f64>(f: F) -> Self {
        Self(L::from_fn(f))
    }
}

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> From<L> for Tensor<N, RANK, L> {
    fn from(value: L) -> Self {
        Tensor(value)
    }
}

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> Default for Tensor<N, RANK, L> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> Index<[usize; RANK]>
    for Tensor<N, RANK, L>
{
    type Output = f64;

    fn index(&self, index: [usize; RANK]) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> IndexMut<[usize; RANK]>
    for Tensor<N, RANK, L>
{
    fn index_mut(&mut self, index: [usize; RANK]) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const N: usize, L: TensorLayout<N, 1>> Index<usize> for Tensor<N, 1, L> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index([index])
    }
}

impl<const N: usize, L: TensorLayout<N, 1>> IndexMut<usize> for Tensor<N, 1, L> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut([index])
    }
}

// ***********************************
// Vector space structure ************
// ***********************************

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> Add for Tensor<N, RANK, L> {
    type Output = Tensor<N, RANK, L>;

    fn add(self, rhs: Tensor<N, RANK, L>) -> Self::Output {
        Tensor::from_fn(|index| self[index] + rhs[index])
    }
}

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> Sub for Tensor<N, RANK, L> {
    type Output = Tensor<N, RANK, L>;

    fn sub(self, rhs: Tensor<N, RANK, L>) -> Self::Output {
        Tensor::from_fn(|index| self[index] - rhs[index])
    }
}

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> Mul<f64> for Tensor<N, RANK, L> {
    type Output = Tensor<N, RANK, L>;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor::from_fn(|index| self[index] * rhs)
    }
}

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> Mul<Tensor<N, RANK, L>> for f64 {
    type Output = Tensor<N, RANK, L>;

    fn mul(self, rhs: Tensor<N, RANK, L>) -> Self::Output {
        Tensor::from_fn(|index| self * rhs[index])
    }
}

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> Div<f64> for Tensor<N, RANK, L> {
    type Output = Tensor<N, RANK, L>;

    fn div(self, rhs: f64) -> Self::Output {
        Tensor::from_fn(|index| self[index] / rhs)
    }
}

// *******************************
// Full Layout *******************
// *******************************

macro_rules! FullLayoutType {
    ($head:literal $($rest:literal)*) => {
        [FullLayoutType!($($rest)*); N]
    };
    () => {
        f64
    };
}

macro_rules! full_layout_zeros {
    ($head:literal $($rest:literal)*) => {
        [full_layout_zeros!($($rest)*);N]
    };
    () => {
        0.0
    };
}

pub type Full0<const N: usize> = FullLayoutType!();
pub type Full1<const N: usize> = FullLayoutType!(0);
pub type Full2<const N: usize> = FullLayoutType!(0 1);
pub type Full3<const N: usize> = FullLayoutType!(0 1 2);
pub type Full4<const N: usize> = FullLayoutType!(0 1 2 3);
pub type Full5<const N: usize> = FullLayoutType!(0 1 2 3 4);
pub type Full6<const N: usize> = FullLayoutType!(0 1 2 3 4 5);

macro_rules! impl_full_layout {
    ($rank:literal | $($i:literal)*) => {
        impl<const N: usize> TensorLayout<N, $rank> for FullLayoutType!($($i)*) {
            fn index(&self, indices: [usize; $rank]) -> &f64 {
                &self$([indices[$i]])*
            }

            fn index_mut(&mut self, indices: [usize; $rank]) -> &mut f64 {
                &mut self$([indices[$i]])*
            }

            fn zeros() -> Self {
                full_layout_zeros!($($i)*)
            }
        }
    };
}

impl<const N: usize> TensorLayout<N, 0> for f64 {
    fn index(&self, _indices: [usize; 0]) -> &f64 {
        self
    }

    fn index_mut(&mut self, _indices: [usize; 0]) -> &mut f64 {
        self
    }

    fn zeros() -> Self {
        0.0
    }
}

impl_full_layout! { 1 | 0 }
impl_full_layout! { 2 | 0 1 }
impl_full_layout! { 3 | 0 1 2 }
impl_full_layout! { 4 | 0 1 2 3 }
impl_full_layout! { 5 | 0 1 2 3 4 }
impl_full_layout! { 6 | 0 1 2 3 4 5 }

// ************************************
// Tensor Products ********************
// ************************************

macro_rules! impl_full_product {
    ($lrank:literal $rrank:literal : $($i:literal)* times $($j:literal)*) => {
        paste! {
            const [<RES_RANK_ $lrank _ $rrank>]: usize = ::aeon_tensor::count_repetitions!($($i)* $($j)*);

            impl<const N: usize> Mul<Tensor<N, $rrank, FullLayoutType!($($j)*)>> for Tensor<N, $lrank, FullLayoutType!($($i)*)> {
                type Output = Tensor<N, [<RES_RANK_ $lrank _ $rrank>], FullLayoutType!($($i)* $($j)*)>;

                fn mul(self, rhs: Tensor<N, $rrank, FullLayoutType!($($j)*)>) -> Self::Output {
                    Tensor::from_fn(|index| {
                        let lindex: [usize; $lrank] = array::from_fn(|i| index[i]);
                        let rindex: [usize; $rrank] = array::from_fn(|i| index[$lrank + i]);

                        self[lindex] * rhs[rindex]
                    })
                }
            }
        }

    };
}

macro_rules! impl_full_products {
    ($lhead:literal $($llist:literal)+ times $rhead:literal $($rlist:literal)+) => {
        impl_full_product! { $lhead $rhead : $($llist)+ times $($rlist)+ }
        impl_full_products! { $lhead $($llist)+ times $($rlist)+ }
    };
    ($lhead:literal $($llist:literal)+ times $rhead:literal) => {
        impl_full_product! { $lhead $rhead : $($llist)+ times }
    };
    ($lhead:literal times $rhead:literal $($rlist:literal)+) => {
        impl_full_product! { $lhead $rhead : times $($rlist)+}
        impl_full_products! { $lhead times $($rlist)+ }

    };
    ($lhead:literal times $rhead:literal) => {
        impl_full_product! { $lhead $rhead : times }
    };
}

impl_full_products! { 4 3 2 1 0 times 2 1 0}
impl_full_products! { 3 2 1 0 times 3 2 1 0}
impl_full_products! { 2 1 0 times 4 3 2 1 0}
impl_full_products! { 1 0 times 5 4 3 2 1 0}
impl_full_products! { 0 times 6 5 4 3 2 1 0}

// ***************************************
// Aliases *******************************
// ***************************************

pub type Scalar<const N: usize> = Tensor<N, 0, f64>;
pub type Vector<const N: usize> = Tensor<N, 1, Full1<N>>;
pub type Matrix<const N: usize> = Tensor<N, 2, Full2<N>>;
pub type Tensor3<const N: usize> = Tensor<N, 3, Full3<N>>;
pub type Tensor4<const N: usize> = Tensor<N, 4, Full4<N>>;

impl<const N: usize, L: TensorLayout<N, 2>> Tensor<N, 2, L> {
    pub fn trace(&self) -> f64 {
        let mut result = 0.0;

        for axis in 0..N {
            result += self[[axis, axis]]
        }

        result
    }
}

impl<const N: usize, const RANK: usize, L: TensorLayout<N, RANK>> Tensor<N, RANK, L> {
    pub fn contract<const ORANK: usize, O: TensorLayout<N, ORANK>>(
        &self,
        a: usize,
        b: usize,
    ) -> Tensor<N, ORANK, O> {
        const {
            assert!(RANK == ORANK + 2);
        }
        assert!(a < RANK);
        assert!(b < RANK);
        assert!(a != b);

        Tensor::from_fn(|oindex| {
            let mut result = 0.0;

            let mut index = [0; RANK];

            let mut i = 0;
            while i < a && i < b {
                index[i] = oindex[i];
                i += 1;
            }

            while i < a || i < b {
                index[i + 1] = oindex[i];
                i += 1;
            }

            while i < RANK {
                index[i + 2] = oindex[i];
                i += 1;
            }

            for axis in 0..N {
                index[a] = axis;
                index[b] = axis;

                result += self[index];
            }

            result
        })
    }
}

pub struct Space<const N: usize>;

impl<const N: usize> Space<N> {
    pub fn vector(&self, f: impl Fn(usize) -> f64) -> Vector<N> {
        Vector::from_fn(|[i]: [usize; 1]| f(i))
    }

    pub fn matrix(&self, f: impl Fn(usize, usize) -> f64) -> Matrix<N> {
        Matrix::from_fn(|[i, j]: [usize; 2]| f(i, j))
    }

    pub fn sum<const R: usize>(&self, f: impl Fn([usize; R]) -> f64) -> f64 {
        let mut result = 0.0;

        for_each_index::<N, R>(|i| {
            result += f(i);
        });

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_indices() {
        let mut indices = Vec::new();
        for_each_index::<3, 2>(|i| indices.push(i));

        assert_eq!(indices[0], [0, 0]);
        assert_eq!(indices[1], [1, 0]);
        assert_eq!(indices[2], [2, 0]);
        assert_eq!(indices[3], [0, 1]);
        assert_eq!(indices[4], [1, 1]);
        assert_eq!(indices[5], [2, 1]);
        assert_eq!(indices[6], [0, 2]);
        assert_eq!(indices[7], [1, 2]);
        assert_eq!(indices[8], [2, 2]);
    }
}
