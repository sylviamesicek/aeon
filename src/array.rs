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
//!     type Weights: ArrayLike<usize, Elem = f64>;
//! }
//!
//! struct MyStruct;
//!
//! impl MyTrait for MyStruct {
//!     type Weights = [f64; 10];
//! }
//! ```
//!
//! This module also implements various utilities for serializing arbitrary length arrays.
//! The current version of `serde` hasn't been able to do this as it breaks backwards
//! compatibility for zero length arrays.

use serde::de::{Deserialize, Deserializer, Error as _};
use serde::ser::{Serialize, SerializeTuple as _, Serializer};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

/// A helper trait for array types which can be indexed and iterated, with
/// compile time known length. Use of this trait can be removed if `generic_const_exprs`
/// is ever stabilized.
pub trait ArrayLike<Idx>:
    Index<Idx, Output = Self::Elem> + IndexMut<Idx, Output = Self::Elem>
{
    /// Length of array, known at compile time.
    const LEN: usize;

    /// Type of elements in the array
    type Elem;

    /// Creates an array of the given length and type by repeatly calling the given function.
    fn from_fn<F: FnMut(Idx) -> Self::Elem>(cb: F) -> Self;
}

impl<T, const N: usize> ArrayLike<usize> for [T; N] {
    const LEN: usize = N;

    type Elem = T;

    fn from_fn<F: FnMut(usize) -> T>(cb: F) -> Self {
        core::array::from_fn::<T, N, F>(cb)
    }
}

/// A wrapper around an `ArrayLike` which implements several common traits depending on the elements of `I`.
/// This includes `Default`, `Clone`, `From`, and serialization assuming the element type also satisfies those traits.
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct ArrayWrap<T, const N: usize>(pub [T; N]);

impl<T: Serialize, const N: usize> Serialize for ArrayWrap<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_tuple(N)?;

        for i in 0..N {
            seq.serialize_element(&self.0[i])?;
        }

        seq.end()
    }
}

impl<'de, T, const N: usize> Deserialize<'de> for ArrayWrap<T, N>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        /// Visitor for deserializing ArrayWrap<T, N>
        struct Visitor<T, const N: usize>(PhantomData<[T; N]>);

        impl<'de, T: Deserialize<'de>, const N: usize> serde::de::Visitor<'de> for Visitor<T, N> {
            type Value = ArrayWrap<T, N>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_fmt(format_args!("Array of Length {}", N))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut arr = [const { MaybeUninit::<T>::uninit() }; N];

                let mut i = 0;

                let err = loop {
                    if i >= N {
                        break None;
                    }

                    let elem = seq.next_element::<T>();

                    match elem {
                        Ok(Some(val)) => arr[i] = MaybeUninit::new(val),
                        Ok(None) => {
                            break Some(A::Error::custom::<String>(String::from(
                                "Sequence length does not match array length",
                            )));
                        }
                        Err(e) => break Some(e),
                    }

                    i += 1;
                };

                if let Some(e) = err {
                    for item in arr.iter_mut().take(i) {
                        unsafe {
                            item.assume_init_drop();
                        }
                    }

                    return Err(e);
                }

                Ok(ArrayWrap(unsafe {
                    std::mem::transmute_copy::<_, [T; N]>(&arr)
                }))
            }
        }

        deserializer.deserialize_tuple(N, Visitor::<T, N>(PhantomData))
    }
}

pub fn serialize<S, T, const N: usize>(data: &[T; N], ser: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    let arr: &ArrayWrap<T, N> = unsafe { std::mem::transmute(data) };
    arr.serialize(ser)
}

/// Deserialize const generic or arbitrarily-large arrays
///
/// For any array up to length `usize::MAX`, this function will allow Serde to properly deserialize
/// it, provided the type `T` itself is deserializable.
///
/// This implementation is adapted from the [Serde documentation][deserialize_map].
///
/// [deserialize_map]: https://serde.rs/deserialize-map.html
pub fn deserialize<'de, D, T, const N: usize>(deserialize: D) -> Result<[T; N], D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    ArrayWrap::<T, N>::deserialize(deserialize).map(|val| val.0)
}

/// Contains methods for working with vecs of arrays.
pub mod vec {
    use super::ArrayWrap;
    use serde::{de, ser::SerializeSeq, Deserialize, Deserializer, Serialize, Serializer};
    use std::{fmt, marker::PhantomData};

    /// Serialize vectors of const generic arrays.
    pub fn serialize<S, T, const N: usize>(data: &Vec<[T; N]>, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        // See: https://serde.rs/impl-serialize.html#serializing-a-tuple
        let mut s = ser.serialize_seq(Some(data.len()))?;
        for array in data {
            let array = unsafe { std::mem::transmute::<&[T; N], &ArrayWrap<T, N>>(array) };
            s.serialize_element(array)?;
        }
        s.end()
    }

    /// Deserialize vectors of const generic arrays.
    ///
    /// For any array up to length `usize::MAX`, this function will allow Serde to properly deserialize
    /// it, provided the type `T` itself is deserializable.
    ///
    /// This implementation is adapted from the [Serde documentation][deserialize_map].
    ///
    /// [deserialize_map]: https://serde.rs/deserialize-map.html
    pub fn deserialize<'de, D, T, const N: usize>(deserialize: D) -> Result<Vec<[T; N]>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        /// A Serde Deserializer `Visitor` for Vec<[T; N]> arrays
        struct Visitor<T, const N: usize> {
            _marker: PhantomData<T>,
        }

        impl<'de, T, const N: usize> de::Visitor<'de> for Visitor<T, N>
        where
            T: Deserialize<'de>,
        {
            type Value = Vec<[T; N]>;

            /// Format a message stating we expect an array of size `N`
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "a vector of arrays of size {}", N)
            }

            /// Process a sequence into an array
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut arr: Vec<[T; N]> = Vec::new();

                if let Some(size) = seq.size_hint() {
                    arr.reserve(size);
                }

                loop {
                    match seq.next_element() {
                        Ok(Some(ArrayWrap(val))) => arr.push(val),
                        Ok(None) => break,
                        Err(e) => return Err(e),
                    }
                }

                Ok(arr)
            }
        }

        deserialize.deserialize_seq(Visitor::<T, N> {
            _marker: PhantomData,
        })
    }
}
