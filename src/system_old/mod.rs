//! Utilities and classes for working with `System`s.
//!
//! Systems are collections of multiple scalar fields that all have the same length, but different data. T
//! his is used to represent coupled PDEs and ODEs. Each system is defined by a `SystemLabel`, which can be
//! implemented by hand or using the provided procedural macro.

use aeon_array::ArrayLike;

use std::array;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

mod prim;
mod vec;

pub use prim::{Empty, Pair, Scalar};
pub use vec::{SystemFields, SystemFieldsMut, SystemRange, SystemSlice, SystemSliceMut, SystemVec};

/// Custom derive macro for the `SystemLabel` trait.
pub use aeon_macros::SystemLabel;

/// This trait is used to define systems of fields.
pub trait SystemLabel: Sized + Clone + Send + Sync + 'static {
    /// Name of the system (used for debugging and when serializing a system).
    const SYSTEM_NAME: &'static str;

    /// Retrieves the name of an individual field.
    fn name(&self) -> String;

    /// Retrieves the index of an individual field.
    fn index(&self) -> usize;

    /// Array type with same length as number of fields
    type Array<T>: ArrayLike<Self, Elem = T>;

    /// Returns an array of all possible system labels.
    fn fields() -> impl Iterator<Item = Self>;

    /// Creates a field from an index.
    fn field_from_index(index: usize) -> Self;
}

/// Number of fields in a given system.
pub const fn field_count<Label: SystemLabel>() -> usize {
    Label::Array::<()>::LEN
}

/// A wrapper around [T; L] to allow indexing them by a system label.
pub struct SystemArray<T, const L: usize>(pub [T; L]);

impl<T, const L: usize> From<[T; L]> for SystemArray<T, L> {
    fn from(value: [T; L]) -> Self {
        SystemArray(value)
    }
}

impl<T, const L: usize, S: SystemLabel<Array<T> = SystemArray<T, L>>> Index<S>
    for SystemArray<T, L>
{
    type Output = T;
    fn index(&self, index: S) -> &Self::Output {
        let index = index.index();
        self.0.index(index)
    }
}

impl<T, const L: usize, S: SystemLabel<Array<T> = SystemArray<T, L>>> IndexMut<S>
    for SystemArray<T, L>
{
    fn index_mut(&mut self, index: S) -> &mut Self::Output {
        let index = index.index();
        self.0.index_mut(index)
    }
}

impl<T, const L: usize, S: SystemLabel<Array<T> = SystemArray<T, L>>> ArrayLike<S>
    for SystemArray<T, L>
{
    const LEN: usize = L;
    type Elem = T;

    fn from_fn<F: FnMut(S) -> Self::Elem>(mut cb: F) -> Self {
        Self(array::from_fn(|index| cb(S::field_from_index(index))))
    }
}

/// Represents the values of a coupled system at a single point.
#[derive(Debug, Clone)]
pub struct SystemValue<Label: SystemLabel>(Label::Array<f64>);

impl<Label: SystemLabel> SystemValue<Label> {
    /// Constructs a new system value by wrapping an array of values.
    pub fn new(values: impl Into<Label::Array<f64>>) -> Self {
        Self(values.into())
    }

    /// Constructs the SystemVal by calling a function for each field
    pub fn from_fn<F: FnMut(Label) -> f64>(f: F) -> Self {
        Self(Label::Array::<f64>::from_fn(f))
    }

    /// Retrieves the value of the given field
    pub fn field(&self, label: Label) -> f64 {
        self.0[label]
    }

    /// Sets the value of the given field.
    pub fn set_field(&mut self, label: Label, v: f64) {
        self.0[label] = v
    }
}

impl<Label: SystemLabel> Default for SystemValue<Label> {
    fn default() -> Self {
        Self::from_fn(|_| f64::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, PartialEq, Eq, Debug, SystemLabel)]
    pub enum MySystem {
        First,
        Second,
        Third,
    }

    #[test]
    fn systems() {
        assert_eq!(MySystem::SYSTEM_NAME, "MySystem");
        for (a, b) in MySystem::fields().zip([MySystem::First, MySystem::Second, MySystem::Third]) {
            assert_eq!(a, b)
        }
        assert_eq!(MySystem::First.index(), 0);
        assert_eq!(MySystem::Second.index(), 1);
        assert_eq!(MySystem::Third.index(), 2);
        assert_eq!(MySystem::First.name(), "First");
        assert_eq!(MySystem::Second.name(), "Second");
        assert_eq!(MySystem::Third.name(), "Third");

        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let owned = SystemSlice::<MySystem>::from_contiguous(data.as_ref()).to_vec();

        assert_eq!(owned.len(), 3);
        assert_eq!(owned.field(MySystem::First), &[0.0, 1.0, 2.0]);
        assert_eq!(owned.field(MySystem::Second), &[3.0, 4.0, 5.0]);
        assert_eq!(owned.field(MySystem::Third), &[6.0, 7.0, 8.0]);

        let slice = owned.slice(1..3);
        let slice_cast = SystemSlice::<MySystem>::from_contiguous(&data);
        assert_eq!(
            slice.field(MySystem::First),
            &slice_cast.field(MySystem::First)[1..3]
        );
        assert_eq!(
            slice.field(MySystem::Second),
            &slice_cast.field(MySystem::Second)[1..3]
        );
        assert_eq!(
            slice.field(MySystem::Third),
            &slice_cast.field(MySystem::Third)[1..3]
        );

        let fields = owned.as_slice().fields();
        assert_eq!(fields.field(MySystem::First), &[0.0, 1.0, 2.0]);
        assert_eq!(fields.field(MySystem::Second), &[3.0, 4.0, 5.0]);
        assert_eq!(fields.field(MySystem::Third), &[6.0, 7.0, 8.0]);
    }
}
