//! Crate for manipulating tensors and tensorial quantaties in Rust.

extern crate self as aeon_tensor;

use std::fmt::Debug;
use std::{marker::PhantomData, ops};

mod compound;
mod indices;
pub mod metric;
mod storage;

pub use compound::{Compound, CompoundIndices, SymSym, SymVec, VecSym, VecSymVec};
pub use indices::{Gen, GenIndices, Sym, SymIndices, TensorIndex};
pub use storage::{TensorStorageMut, TensorStorageOwned, TensorStorageRef};

/// Basic tensor object. Depends on dimension (`N`), rank (`R`), Symmetries
/// (`I`) and storage array (`S`).
pub struct Tensor<const N: usize, const R: usize, I, S> {
    /// Internal storage for tensor, simply wraps around `S`.
    storage: S,
    _marker: PhantomData<I>,
}

impl<const N: usize, const R: usize, I, S> Tensor<N, R, I, S> {
    /// Retrieves the dimension of the tensor.
    pub fn dim() -> usize {
        N
    }

    /// Retrieves the rank of the tensor.
    pub fn rank() -> usize {
        R
    }
}

impl<const N: usize, const R: usize, I: TensorIndex<N, R>, S: TensorStorageOwned + Default>
    Tensor<N, R, I, S>
{
    /// Constructs a new tensor with undefined internal values.
    pub fn new() -> Self {
        let mut storage = S::default();
        storage.resize(I::count());

        Self {
            storage,
            _marker: PhantomData,
        }
    }

    /// Constructs a new tensor by repeatidly calling a function on each index.
    pub fn from_fn(f: impl Fn([usize; R]) -> f64) -> Self {
        let mut result = Self::new();
        result.fill_from_fn(f);
        result
    }

    /// Constructs a tensor with all components initialized to v.
    pub fn splat(v: f64) -> Self {
        let mut result = Self::new();
        result.fill(v);
        result
    }

    /// Constructs a tensor initialized with all components inititialized to zero.
    pub fn zeros() -> Self {
        let mut result = Self::new();
        result.fill(0.0);
        result
    }

    /// Constructs a tensor from a tensorial expression.
    pub fn from_eq<const C: usize>(f: impl Fn([usize; R], [usize; C]) -> f64) -> Self {
        Self::from_fn(|index| {
            let mut result = 0.0;
            <Gen as TensorIndex<N, C>>::for_each_index(|sum| {
                result += f(index, sum);
            });
            result
        })
    }
}

impl<const N: usize, const R: usize, I: TensorIndex<N, R>, S: TensorStorageRef> From<S>
    for Tensor<N, R, I, S>
{
    fn from(value: S) -> Self {
        assert!(value.buffer().len() == I::count());

        Self {
            storage: value,
            _marker: PhantomData,
        }
    }
}

impl<const N: usize, const R: usize, I: TensorIndex<N, R>, S: TensorStorageMut> Tensor<N, R, I, S> {
    /// Sets all free components of the tensor to v.
    pub fn fill(&mut self, v: f64) {
        let buffer = self.storage.buffer_mut();
        buffer[..I::count()].fill(v);
    }

    /// Sets values of components of the tensor be invoking the given function.
    pub fn fill_from_fn(&mut self, f: impl Fn([usize; R]) -> f64) {
        let buffer = self.storage.buffer_mut();

        let mut offset = 0;

        I::for_each_index(|index| {
            buffer[offset] = f(index);
            offset += 1;
        });
    }
}

impl<const N: usize, const R: usize, I: TensorIndex<N, R>, S: TensorStorageRef> Tensor<N, R, I, S> {
    /// Retrieves the component at the given index of the tensor.
    pub fn get(&self, index: [usize; R]) -> &f64 {
        let offset = I::offset_from_index(index);
        let buffer = self.storage.buffer();
        &buffer[offset]
    }
}

impl<const N: usize, const R: usize, I: TensorIndex<N, R>, S: TensorStorageMut> Tensor<N, R, I, S> {
    /// Retrieves a mutable reference to the degree of freedom corresponding to the given index
    /// of the tensor.
    pub fn get_mut(&mut self, index: [usize; R]) -> &mut f64 {
        let offset = I::offset_from_index(index);
        let buffer = self.storage.buffer_mut();
        &mut buffer[offset]
    }
}

impl<const N: usize, const R: usize, I, S: Clone> Clone for Tensor<N, R, I, S> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            _marker: self._marker.clone(),
        }
    }
}

impl<const N: usize, const R: usize, I, S: Copy> Copy for Tensor<N, R, I, S> {}

impl<const N: usize, const R: usize, I, S: Debug> Debug for Tensor<N, R, I, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.storage.fmt(f)
    }
}

impl<const N: usize, const R: usize, I: TensorIndex<N, R>, S: TensorStorageOwned + Default> Default
    for Tensor<N, R, I, S>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const R: usize, I: TensorIndex<N, R>, S: TensorStorageRef>
    ops::Index<[usize; R]> for Tensor<N, R, I, S>
{
    type Output = f64;

    fn index(&self, index: [usize; R]) -> &Self::Output {
        self.get(index)
    }
}

impl<const N: usize, const R: usize, I: TensorIndex<N, R>, S: TensorStorageMut>
    ops::IndexMut<[usize; R]> for Tensor<N, R, I, S>
{
    fn index_mut(&mut self, index: [usize; R]) -> &mut Self::Output {
        self.get_mut(index)
    }
}

// *****************************
// Tests ***********************
// *****************************

#[cfg(test)]
mod tests {
    use crate::{Gen, Sym, SymSym, SymVec, Tensor, TensorIndex, VecSym, VecSymVec};

    fn test_index_axioms<const N: usize, const R: usize, I: TensorIndex<N, R>>() {
        assert_eq!(
            I::count(),
            I::indices().count(),
            "I::count length ({}) doesn't match I::indices().count() ({})",
            I::count(),
            I::indices().count()
        );

        let mut offset = 0;
        let mut indices = I::indices();

        I::for_each_index(|i| {
            let Some(j) = indices.next() else {
                panic!("I::indices length doesn't match I::for_each_index");
            };

            assert_eq!(
                i, j,
                "I::for_each_index doesn't iterate in the same order as I::indices"
            );

            assert_eq!(
                offset,
                I::offset_from_index(i),
                "I::offset_from_index doesn't iterate in same order as I::indices"
            );

            offset += 1;
        });

        assert_eq!(
            None,
            indices.next(),
            "I::indices length doesn't match I::for_each_index"
        );
    }

    #[test]
    fn index_axioms() {
        test_index_axioms::<4, 5, Gen>();
        test_index_axioms::<4, 2, Sym>();
        test_index_axioms::<4, 3, VecSym>();
        test_index_axioms::<4, 3, SymVec>();
        test_index_axioms::<3, 4, SymSym>();
        test_index_axioms::<3, 4, VecSymVec>();
    }

    #[test]
    fn symmetric() {
        let tensor =
            Tensor::<2, 4, SymSym, [f64; 9]>::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // rr
        assert_eq!(tensor[[0, 0, 0, 0]], 1.0);
        assert_eq!(tensor[[0, 0, 0, 1]], 2.0);
        assert_eq!(tensor[[0, 0, 1, 0]], 2.0);
        assert_eq!(tensor[[0, 0, 1, 1]], 3.0);
        // rz
        assert_eq!(tensor[[0, 1, 0, 0]], 4.0);
        assert_eq!(tensor[[0, 1, 0, 1]], 5.0);
        assert_eq!(tensor[[0, 1, 1, 0]], 5.0);
        assert_eq!(tensor[[0, 1, 1, 1]], 6.0);
        // zr
        assert_eq!(tensor[[1, 0, 0, 0]], 4.0);
        assert_eq!(tensor[[1, 0, 0, 1]], 5.0);
        assert_eq!(tensor[[1, 0, 1, 0]], 5.0);
        assert_eq!(tensor[[1, 0, 1, 1]], 6.0);
        // zz
        assert_eq!(tensor[[1, 1, 0, 0]], 7.0);
        assert_eq!(tensor[[1, 1, 0, 1]], 8.0);
        assert_eq!(tensor[[1, 1, 1, 0]], 8.0);
        assert_eq!(tensor[[1, 1, 1, 1]], 9.0);

        let mut indices = <SymSym as TensorIndex<2, 4>>::indices();

        assert_eq!(indices.next(), Some([0, 0, 0, 0]));
        assert_eq!(indices.next(), Some([0, 0, 1, 0]));
        assert_eq!(indices.next(), Some([0, 0, 1, 1]));
        assert_eq!(indices.next(), Some([1, 0, 0, 0]));
        assert_eq!(indices.next(), Some([1, 0, 1, 0]));
        assert_eq!(indices.next(), Some([1, 0, 1, 1]));
        assert_eq!(indices.next(), Some([1, 1, 0, 0]));
        assert_eq!(indices.next(), Some([1, 1, 1, 0]));
        assert_eq!(indices.next(), Some([1, 1, 1, 1]));
        assert_eq!(indices.next(), None);
    }

    #[test]
    fn general() {
        let tensor =
            Tensor::<3, 2, Gen, [f64; 9]>::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        assert_eq!(tensor[[0, 0]], 1.0);
        assert_eq!(tensor[[0, 1]], 2.0);
        assert_eq!(tensor[[0, 2]], 3.0);
        assert_eq!(tensor[[1, 0]], 4.0);
        assert_eq!(tensor[[1, 1]], 5.0);
        assert_eq!(tensor[[1, 2]], 6.0);
        assert_eq!(tensor[[2, 0]], 7.0);
        assert_eq!(tensor[[2, 1]], 8.0);
        assert_eq!(tensor[[2, 2]], 9.0);
    }
}
