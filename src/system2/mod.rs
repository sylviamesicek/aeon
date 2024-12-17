use std::ops::Range;

mod dynamic;
mod r#static;

pub use dynamic::*;
pub use r#static::*;

/// Represents a collection of slices of the same length that together form a system. This data is stored in SoA style and thus more cache friendly. These
/// can be used to store system vectors, tensor fields, etc.
pub trait SystemSlice<'a> {
    /// Label used to identify a given field in the system.
    type Label;

    /// Result of taking a subslice of this slice.
    type SubSlice<'b>: SystemSlice<'b, Label = Self::Label>
    where
        Self: 'b;

    /// Result of converting this slice into a system reference.
    type SystemRef<'b>: SystemRef<'b, Label = Self::Label>
    where
        Self: 'b;

    /// The number of DoFs in this system.
    fn len(&self) -> usize;

    /// Returns true if the system contains no DoFs.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Takes an immutable subslice of the current slice.
    fn subslice(&self, range: Range<usize>) -> Self::SubSlice<'_>;

    /// Converts a slice into a system ref.
    fn as_system_ref(&self) -> Self::SystemRef<'_>;
}

/// A collection of mutable slices of the same length that together form a system.
pub trait SystemSliceMut<'a>: SystemSlice<'a> {
    /// Result of taking a mutable slice of the system.
    type SubSliceMut<'b>: SystemSliceMut<'b, Label = Self::Label>
    where
        Self: 'b;

    /// Result of converting this slice to a mutable system reference.
    type SystemMut<'b>: SystemMut<'b, Label = Self::Label>
    where
        Self: 'b;

    /// Takes a mutable subslice of the current slice.
    fn subslice_mut(&mut self, range: Range<usize>) -> Self::SubSliceMut<'_>;

    /// Takes a mutable slice of the system using an immutable reference. This is
    /// safe and sound so long as each simultaneuously existing mutable slice is disjoint.
    unsafe fn subslice_unsafe(&self, range: Range<usize>) -> Self::SubSliceMut<'_>;

    /// Converts a mutable slice into a system mut.
    fn as_system_mut(&mut self) -> Self::SystemMut<'_>;
}

pub trait SystemRef<'a> {
    /// Label used to identify fields in the system.
    type Label;

    /// Number of DoFs in the system.
    fn len(&self) -> usize;

    /// Returns true if the system has no DoFs.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to a
    fn field(&self, label: Self::Label) -> &[f64];
}

pub trait SystemMut<'a>: SystemRef<'a> {
    fn field_mut(&mut self, label: Self::Label) -> &mut [f64];
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
