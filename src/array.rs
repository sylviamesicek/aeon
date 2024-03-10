use std::ops::{Index, IndexMut};

pub trait Array<T>:
    Index<usize, Output = T> + IndexMut<usize, Output = T> + IntoIterator<Item = T>
{
    const LEN: usize;
}

impl<T, const N: usize> Array<T> for [T; N] {
    const LEN: usize = N;
}
