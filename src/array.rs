//! This modules contains several "hacky" utilities to work around the current lack of generic_const_exprs in
//! Rust.
//!
//! In several parts of the codebase (`System`s, `Kernel`s, etc.) we have to creates arrays whose
//! length is determined by an associated constant of a trait. Seeing as this is impossible in stable Rust, we instead
//! use the following pattern:
//!
//! ```rust
//! # use aeon::array::ArrayLike;
//! trait MyTrait {
//!     type Weights: ArrayLike<Elem = f64>;
//! }
//!
//! struct MyStruct;
//!
//! impl MyTrait for MyStruct {
//!     type Weights = [f64; 10];
//! }
//! ```

use serde::de::{Deserialize, Error as _, Visitor};
use serde::ser::{Serialize, SerializeSeq};
use std::borrow::{Borrow, BorrowMut};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::{
    fmt::{Debug, Write},
    ops::{Index, IndexMut},
};

/// A helper trait for array types which can be indexed and iterated, with
/// compile time known length. Use of this trait can be removed if `generic_const_exprs`
/// is ever stabilized.
pub trait ArrayLike:
    Index<usize, Output = Self::Elem>
    + IndexMut<usize, Output = Self::Elem>
    + IntoIterator<Item = Self::Elem>
    + Borrow<[Self::Elem]>
    + BorrowMut<[Self::Elem]>
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

/// A wrapper around an `ArrayLike` which implements several common traits depending on the elements of `I`.
/// This includes `Default`, `Clone`, `From`, and serialization assuming the element type also satisfies those traits.
#[repr(transparent)]
pub struct Array<I: ArrayLike>(I);

impl<I: ArrayLike> Array<I> {
    /// Unwraps the array, returning the inner type.
    pub fn inner(self) -> I {
        self.0
    }
}

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

impl<I: ArrayLike> Serialize for Array<I>
where
    I::Elem: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(I::LEN))?;

        for i in 0..I::LEN {
            seq.serialize_element(&self[i])?;
        }

        seq.end()
    }
}

struct ArrayVisitor<I: ArrayLike>(PhantomData<I>);

impl<'de, I: ArrayLike> Visitor<'de> for ArrayVisitor<I>
where
    I::Elem: Deserialize<'de> + Default,
{
    type Value = Array<I>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_fmt(format_args!("Array of Length {}", I::LEN))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut result = Array::<I>::default();

        for i in 0..I::LEN {
            result[i] = seq.next_element::<I::Elem>()?.ok_or_else(|| {
                A::Error::custom::<String>(String::from(
                    "Sequence length does not match array length",
                ))
            })?;
        }

        Ok(result)
    }
}

impl<'de, I: ArrayLike> Deserialize<'de> for Array<I>
where
    I::Elem: Deserialize<'de> + Default,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(ArrayVisitor::<I>(PhantomData))
    }
}

/// Converts a vector of arraylike values to a vector of those values wrapped in the newtype
pub fn wrap_vec_of_arrays<I: ArrayLike>(vec: Vec<I>) -> Vec<Array<I>> {
    unsafe {
        let mut v = ManuallyDrop::new(vec);
        Vec::from_raw_parts(v.as_mut_ptr() as *mut Array<I>, v.len(), v.capacity())
    }
}

/// Converts a vector of wrapped arrays into a vector of their underlying arraylike values.
pub fn unwrap_vec_of_arrays<I: ArrayLike>(vec: Vec<Array<I>>) -> Vec<I> {
    unsafe {
        let mut v = ManuallyDrop::new(vec);
        Vec::from_raw_parts(v.as_mut_ptr() as *mut I, v.len(), v.capacity())
    }
}
