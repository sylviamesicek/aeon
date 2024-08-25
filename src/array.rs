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

pub mod serde_array {
    use serde::{de, ser::SerializeTuple, Deserialize, Deserializer, Serialize, Serializer};
    use std::mem::MaybeUninit;
    use std::{fmt, marker::PhantomData};

    pub fn serialize<S, T, const N: usize>(data: [T; N], ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        // See: https://serde.rs/impl-serialize.html#serializing-a-tuple
        let mut s = ser.serialize_tuple(N)?;
        for item in data {
            s.serialize_element(&item)?;
        }
        s.end()
    }

    /// A Serde Deserializer `Visitor` for [T; N] arrays
    struct ArrayVisitor<T, const N: usize> {
        // Literally nothing (a "phantom"), but stops Rust complaining about the "unused" T parameter
        _marker: PhantomData<T>,
    }

    impl<'de, T, const N: usize> de::Visitor<'de> for ArrayVisitor<T, N>
    where
        T: Deserialize<'de>,
    {
        type Value = [T; N];

        /// Format a message stating we expect an array of size `N`
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(formatter, "an array of size {}", N)
        }

        /// Process a sequence into an array
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            // Safety: `assume_init` is sound because the type we are claiming to have
            // initialized here is a bunch of `MaybeUninit`s, which do not require
            // initialization.
            let mut arr: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };

            // Iterate over the array and fill the elemenets with the ones obtained from
            // `seq`.
            let mut place_iter = arr.iter_mut();
            let mut cnt_filled = 0;
            let err = loop {
                match (seq.next_element(), place_iter.next()) {
                    (Ok(Some(val)), Some(place)) => *place = MaybeUninit::new(val),
                    // no error, we're done
                    (Ok(None), None) => break None,
                    // error from serde, propagate it
                    (Err(e), _) => break Some(e),
                    // lengths do not match, report invalid_length
                    (Ok(None), Some(_)) | (Ok(Some(_)), None) => {
                        break Some(de::Error::invalid_length(cnt_filled, &self))
                    }
                }
                cnt_filled += 1;
            };
            if let Some(err) = err {
                if std::mem::needs_drop::<T>() {
                    for elem in arr.into_iter().take(cnt_filled) {
                        // Safety: `assume_init()` is sound because we did initialize CNT_FILLED
                        // elements. We call it to drop the deserialized values.
                        unsafe {
                            elem.assume_init();
                        }
                    }
                }
                return Err(err);
            }

            // Safety: everything is initialized and we are ready to transmute to the
            // initialized array type.

            // See https://github.com/rust-lang/rust/issues/62875#issuecomment-513834029
            //let ret = unsafe { std::mem::transmute::<_, [T; N]>(arr) };

            let ret = unsafe { std::mem::transmute_copy(&arr) };
            std::mem::forget(arr);

            Ok(ret)
        }
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
        deserialize.deserialize_tuple(
            N,
            ArrayVisitor {
                _marker: PhantomData,
            },
        )
    }
}

pub mod serde_vec_of_arrays {
    use serde::{de, ser::SerializeSeq, Deserialize, Deserializer, Serialize, Serializer};
    use std::{array, fmt, marker::PhantomData, mem::ManuallyDrop};

    fn serialize<S, T, const N: usize>(data: &Vec<[T; N]>, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        // See: https://serde.rs/impl-serialize.html#serializing-a-tuple
        let mut s = ser.serialize_seq(Some(data.len() * N))?;
        for array in data {
            for item in array {
                s.serialize_element(item)?;
            }
        }
        s.end()
    }

    /// A Serde Deserializer `Visitor` for Vec<[T; N]> arrays
    struct ArrayVisitor<T, const N: usize> {
        // Literally nothing (a "phantom"), but stops Rust complaining about the "unused" T parameter
        _marker: PhantomData<T>,
    }

    impl<'de, T, const N: usize> de::Visitor<'de> for ArrayVisitor<T, N>
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
            let mut arr: Vec<ManuallyDrop<T>> = Vec::new();

            if let Some(size) = seq.size_hint() {
                arr.reserve(size);
            }

            let err = loop {
                match seq.next_element() {
                    Ok(Some(val)) => arr.push(ManuallyDrop::new(val)),
                    Ok(None) => break None,
                    Err(e) => break Some(e),
                }
            };

            // If we had an error drop all elements.
            if let Some(e) = err {
                if std::mem::needs_drop::<T>() {
                    for elem in arr.into_iter() {
                        std::mem::drop(ManuallyDrop::<T>::into_inner(elem));
                    }
                }
                return Err(e);
            }

            if arr.len() % N != 0 {
                return Err(de::Error::invalid_length(
                    arr.len(),
                    &"Length must be divisable by N",
                ));
            }

            Ok(arr
                .chunks_mut(N)
                .map(|chunk| array::from_fn(|i| unsafe { ManuallyDrop::take(&mut chunk[i]) }))
                .collect())
        }
    }

    /// Deserialize const generic or arbitrarily-large arrays
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
        deserialize.deserialize_seq(ArrayVisitor::<T, N> {
            _marker: PhantomData,
        })
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

pub mod serialize {
    use serde::{
        de::{self, Deserialize, Deserializer, SeqAccess, Visitor},
        ser::{Serialize, SerializeSeq, SerializeTuple, Serializer},
    };
    use std::{fmt, marker::PhantomData, mem::MaybeUninit};

    pub struct ArrayWrap<'a, T: Serialize, const N: usize> {
        inner: &'a [T; N],
    }

    impl<'a, T: Serialize, const N: usize> ArrayWrap<'a, T, N> {
        pub fn new(array: &'a [T; N]) -> ArrayWrap<'a, T, N> {
            ArrayWrap { inner: array }
        }
    }

    impl<'a, T: Serialize, const N: usize> Serialize for ArrayWrap<'a, T, N> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize(self.inner, serializer)
        }
    }

    /// Trait for types serializable using `serde_arrays`
    ///
    /// In order to serialize data using this crate, the type needs to implement this trait. While this
    /// approach has limitations in what can be supported (namely it limits support to only those types
    /// this trait is explicitly implemented on), the trade off is a significant increase in ergonomics.
    ///
    /// If the greater flexibility lost by this approach is needed, see [`serde_with`][serde_with].
    ///
    /// [serde_with]: https://crates.io/crates/serde_with/
    pub trait Serializable<T: Serialize, const N: usize> {
        fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer;
    }

    impl<T: Serialize, const N: usize, const M: usize> Serializable<T, N> for [[T; N]; M] {
        fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            // Fixed-length structures, including arrays, are supported in Serde as tuples
            // See: https://serde.rs/impl-serialize.html#serializing-a-tuple
            let mut s = ser.serialize_tuple(N)?;
            for item in self {
                let wrapped = ArrayWrap::new(item);
                s.serialize_element(&wrapped)?;
            }
            s.end()
        }
    }

    impl<T: Serialize, const N: usize> Serializable<T, N> for Vec<[T; N]> {
        fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut s = ser.serialize_seq(Some(self.len()))?;
            for item in self {
                let wrapped = ArrayWrap::new(item);
                s.serialize_element(&wrapped)?;
            }
            s.end()
        }
    }

    impl<T: Serialize, const N: usize> Serializable<T, N> for [T; N] {
        fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize_as_tuple(self, ser)
        }
    }

    /// Serialize an array
    ///
    /// In Serde arrays (and other fixed-length structures) are supported as tuples
    fn serialize_as_tuple<S, T, const N: usize>(data: &[T; N], ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        // See: https://serde.rs/impl-serialize.html#serializing-a-tuple
        let mut s = ser.serialize_tuple(N)?;
        for item in data {
            s.serialize_element(item)?;
        }
        s.end()
    }

    /// Serialize const generic or arbitrarily-large arrays
    ///
    /// Types must implement the [`Serializable`] trait; while this requirement sharply limits how
    /// composable the final result is, the simple ergonomics make up for it.
    ///
    /// For greater flexibility see [`serde_with`][serde_with].
    ///
    /// [serde_with]: https://crates.io/crates/serde_with/
    pub fn serialize<A, S, T, const N: usize>(data: &A, ser: S) -> Result<S::Ok, S::Error>
    where
        A: Serializable<T, N>,
        S: Serializer,
        T: Serialize,
    {
        data.serialize(ser)
    }

    /// A Serde Deserializer `Visitor` for [T; N] arrays
    struct ArrayVisitor<T, const N: usize> {
        // Literally nothing (a "phantom"), but stops Rust complaining about the "unused" T parameter
        _marker: PhantomData<T>,
    }

    impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
    where
        T: Deserialize<'de>,
    {
        type Value = [T; N];

        /// Format a message stating we expect an array of size `N`
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(formatter, "an array of size {}", N)
        }

        /// Process a sequence into an array
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // Safety: `assume_init` is sound because the type we are claiming to have
            // initialized here is a bunch of `MaybeUninit`s, which do not require
            // initialization.
            let mut arr: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };

            // Iterate over the array and fill the elemenets with the ones obtained from
            // `seq`.
            let mut place_iter = arr.iter_mut();
            let mut cnt_filled = 0;
            let err = loop {
                match (seq.next_element(), place_iter.next()) {
                    (Ok(Some(val)), Some(place)) => *place = MaybeUninit::new(val),
                    // no error, we're done
                    (Ok(None), None) => break None,
                    // error from serde, propagate it
                    (Err(e), _) => break Some(e),
                    // lengths do not match, report invalid_length
                    (Ok(None), Some(_)) | (Ok(Some(_)), None) => {
                        break Some(de::Error::invalid_length(cnt_filled, &self))
                    }
                }
                cnt_filled += 1;
            };
            if let Some(err) = err {
                if std::mem::needs_drop::<T>() {
                    for elem in arr.into_iter().take(cnt_filled) {
                        // Safety: `assume_init()` is sound because we did initialize CNT_FILLED
                        // elements. We call it to drop the deserialized values.
                        unsafe {
                            elem.assume_init();
                        }
                    }
                }
                return Err(err);
            }

            // Safety: everything is initialized and we are ready to transmute to the
            // initialized array type.

            // See https://github.com/rust-lang/rust/issues/62875#issuecomment-513834029
            //let ret = unsafe { std::mem::transmute::<_, [T; N]>(arr) };

            let ret = unsafe { std::mem::transmute_copy(&arr) };
            std::mem::forget(arr);

            Ok(ret)
        }
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
        deserialize.deserialize_tuple(
            N,
            ArrayVisitor {
                _marker: PhantomData,
            },
        )
    }
}
