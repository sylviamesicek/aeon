use std::{
    fmt::{Debug, Write},
    ops::{Index, IndexMut},
};

/// A helper trait for array types which can be indexed and iterated, with
/// compile time known length. Use of this trait can be removed if generic_const_exprs
/// is every stabilized.
pub trait ArrayLike:
    Index<usize, Output = Self::Elem>
    + IndexMut<usize, Output = Self::Elem>
    + IntoIterator<Item = Self::Elem>
{
    /// Length of array, known at compile time.
    const LEN: usize;

    /// Type of elements in the array
    type Elem;

    /// Creates an array of the given length and type by repeatly calling the given function.
    fn from_fn<F: FnMut(usize) -> Self::Elem>(cb: F) -> Self;
}

impl<T, const N: usize> ArrayLike for [T; N] {
    const LEN: usize = N;

    type Elem = T;

    fn from_fn<F: FnMut(usize) -> T>(cb: F) -> Self {
        core::array::from_fn::<T, N, F>(cb)
    }
}

pub struct Array<I: ArrayLike>(I);

impl<I: ArrayLike> From<I> for Array<I> {
    fn from(value: I) -> Self {
        Array(value)
    }
}

impl<I: ArrayLike> Index<usize> for Array<I> {
    type Output = I::Elem;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<I: ArrayLike> IndexMut<usize> for Array<I> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<I: ArrayLike> IntoIterator for Array<I> {
    type Item = I::Elem;
    type IntoIter = I::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<I: ArrayLike> Default for Array<I>
where
    I::Elem: Default,
{
    fn default() -> Self {
        Array(I::from_fn(|_| I::Elem::default()))
    }
}

impl<I: ArrayLike> Clone for Array<I>
where
    I::Elem: Clone,
{
    fn clone(&self) -> Self {
        Array(I::from_fn(|i| self[i].clone()))
    }
}

impl<I: ArrayLike> Debug for Array<I>
where
    I::Elem: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('[')?;

        for i in 0..I::LEN {
            self[i].fmt(f)?;
            if i != I::LEN - 1 {
                f.write_char(',')?;
            }
        }

        f.write_char(']')?;

        Ok(())
    }
}
