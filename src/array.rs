use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

/// A helper trait for array types which can be indexed and iterated, with
/// compile time known length.
pub trait Array<T: Debug>:
    Index<usize, Output = T> + IndexMut<usize, Output = T> + IntoIterator<Item = T> + Debug
{
    /// Length of array, known at compile time.
    const LEN: usize;
}

impl<T: Debug, const N: usize> Array<T> for [T; N] {
    const LEN: usize = N;
}
