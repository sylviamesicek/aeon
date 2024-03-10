use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

pub trait Array<T: Debug>:
    Index<usize, Output = T> + IndexMut<usize, Output = T> + IntoIterator<Item = T> + Debug
{
    const LEN: usize;
}

impl<T: Debug, const N: usize> Array<T> for [T; N] {
    const LEN: usize = N;
}
