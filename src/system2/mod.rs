#![allow(clippy::len_without_is_empty)]

use std::ops::Range;

mod dynamic;
mod label;
mod r#static;
mod tuple;

pub use dynamic::*;
pub use label::*;
pub use r#static::*;
pub use tuple::*;

pub trait System {
    /// Label used to index the various fields in the system.
    type Label;

    /// The number of DoFs in this system
    fn len(&self) -> usize;

    /// Enumerates over fields in the system.
    fn enumerate(&self) -> impl Iterator<Item = Self::Label>;
}

/// Represents a collection of slices of the same length that together form a system. This data is stored in SoA style and thus more cache friendly. These
/// can be used to store system vectors, tensor fields, etc.
pub trait SystemSlice<'a>: System {
    /// Result of taking a subslice of this slice.
    type Slice<'b>: SystemSlice<'b, Label = Self::Label>
    where
        Self: 'b;

    /// Takes an immutable subslice of the current slice.
    fn slice(&self, range: Range<usize>) -> Self::Slice<'_>;

    /// Returns a reference to a
    fn field(&self, label: Self::Label) -> &[f64];
}

/// A collection of mutable slices of the same length that together form a system.
pub trait SystemSliceMut<'a>: SystemSlice<'a> {
    /// Result of taking a mutable slice of the system.
    type SliceMut<'b>: SystemSliceMut<'b, Label = Self::Label>
    where
        Self: 'b;

    /// Result of transforming the system into a shared slice.
    type Shared: SystemSliceShared<'a, Label = Self::Label>;

    /// Takes a mutable slice of the current slice.
    fn slice_mut(&mut self, range: Range<usize>) -> Self::SliceMut<'_>;

    /// Returns a mutable reference to the given field of the system.
    fn field_mut(&mut self, label: Self::Label) -> &mut [f64];

    /// Converts a mutable reference to a slice into a shared slice that can be
    /// disjointly borrowed.
    fn into_shared(self) -> Self::Shared;
}

pub unsafe trait SystemSliceShared<'a>: System {
    type Slice<'b>: SystemSlice<'b>
    where
        Self: 'b;

    type SliceMut<'b>: SystemSliceMut<'b>
    where
        Self: 'b;

    unsafe fn slice_unsafe(&self, range: Range<usize>) -> Self::Slice<'_>;
    unsafe fn slice_unsafe_mut(&self, range: Range<usize>) -> Self::SliceMut<'_>;
}

// /// Converts genetic range to a concrete range type.
// fn bounds_to_range<R>(total: usize, range: R) -> Range<usize>
// where
//     R: RangeBounds<usize>,
// {
//     let start_inc = match range.start_bound() {
//         Bound::Included(&i) => i,
//         Bound::Excluded(&i) => i + 1,
//         Bound::Unbounded => 0,
//     };

//     let end_exc = match range.end_bound() {
//         Bound::Included(&i) => i + 1,
//         Bound::Excluded(&i) => i,
//         Bound::Unbounded => total,
//     };

//     start_inc..end_exc
// }
