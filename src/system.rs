//! Utilities and classes for working with `System`s.
//!
//! Systems are collections of multiple scalar fields that all have the same length, but different data. T
//! his is used to represent coupled PDEs and ODEs. Each system is defined by a `SystemLabel`, which can be
//! implemented by hand or using the provided procedural macro.

use crate::array::{Array, ArrayLike};

use core::slice;
use reborrow::{Reborrow, ReborrowMut};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Bound, Range, RangeBounds};
use std::slice::SliceIndex;

/// Custom derive macro for the `SystemLabel` trait.
pub use aeon_macros::SystemLabel;

/// This trait is used to define systems of fields.
pub trait SystemLabel: Sized + Clone + Send + Sync + 'static {
    /// Name of the system (used for debugging and when serializing a system).
    const NAME: &'static str;

    /// Array type with same length as number of fields
    type FieldLike<T>: ArrayLike<Elem = T>;

    /// Returns an array of all possible system labels.
    fn fields() -> Array<Self::FieldLike<Self>>;

    /// Retrieves the index of an individual field.
    fn field_index(&self) -> usize;

    /// Retrieves the name of an individual field.
    fn field_name(&self) -> String;
}

/// Helper alias for building an array with the same number of elements as different fields.
pub type FieldArray<Label, T> = Array<<Label as SystemLabel>::FieldLike<T>>;

/// Number of fields in a given system.
pub const fn field_count<Label: SystemLabel>() -> usize {
    Label::FieldLike::<()>::LEN
}

/// Represents the values of a coupled system at a single point.
#[derive(Debug, Clone)]
pub struct SystemValue<Label: SystemLabel>(FieldArray<Label, f64>);

impl<Label: SystemLabel> SystemValue<Label> {
    /// Constructs a new system value by wrapping an array of values.
    pub fn new(values: Label::FieldLike<f64>) -> Self {
        Self(values.into())
    }

    /// Constructs the SystemVal by calling a function for each field
    pub fn from_fn<F: FnMut(Label) -> f64>(mut f: F) -> Self {
        let mut values = Array::default();

        for field in Label::fields() {
            values[field.field_index()] = f(field.clone())
        }

        Self(values)
    }

    /// Retrieves the value of the given field
    pub fn field(&self, label: Label) -> f64 {
        self.0[label.field_index()]
    }

    /// Sets the value of the given field.
    pub fn set_field(&mut self, label: Label, v: f64) {
        self.0[label.field_index()] = v
    }
}

impl<Label: SystemLabel> Default for SystemValue<Label> {
    fn default() -> Self {
        Self::from_fn(|_| f64::default())
    }
}

/// Stores a system in memory as an structure of field vectors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemVec<Label: SystemLabel> {
    data: Vec<f64>,
    _marker: PhantomData<Label>,
}

impl<Label: SystemLabel> SystemVec<Label> {
    /// Constructs a new system with the given length.
    pub fn with_length(length: usize) -> Self {
        Self {
            data: vec![0.0; length * field_count::<Label>()],
            _marker: PhantomData,
        }
    }

    /// Constructs a new system vector from the given data. `data.len()` must be divisible by `field_count::<Label>()`.
    pub fn from_contiguous(data: Vec<f64>) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    /// Transforms a system vector back into a linear vector
    pub fn into_contiguous(self) -> Vec<f64> {
        self.data
    }

    pub fn contigious(&self) -> &[f64] {
        &self.data
    }

    pub fn contigious_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Returns the length of the system.
    pub fn len(&self) -> usize {
        if field_count::<Label>() == 0 {
            0
        } else {
            self.data.len() / field_count::<Label>()
        }
    }

    /// Caches pointers to each field in the system.
    pub fn fields(&self) -> SystemFields<Label> {
        self.as_slice().fields()
    }

    /// Caches mutable pointers to each field in the system.
    pub fn fields_mut(&mut self) -> SystemFieldsMut<Label> {
        self.as_mut_slice().fields_mut()
    }

    /// Retrieves an immutable reference to a field located at the given index.
    pub fn field(&self, label: Label) -> &[f64] {
        let length = self.data.len() / field_count::<Label>();
        &self.data[length * label.field_index()..length * (label.field_index() + 1)]
    }

    /// Retrieves a mutable reference to a field located at the given index.
    pub fn field_mut(&mut self, label: Label) -> &mut [f64] {
        let length = self.data.len() / field_count::<Label>();
        &mut self.data[length * label.field_index()..length * (label.field_index() + 1)]
    }

    pub fn as_range(&self) -> SystemRange<Label> {
        SystemRange {
            ptr: self.data.as_ptr() as *mut f64,
            total: self.data.len(),
            offset: 0,
            length: self.len(),
            _marker: PhantomData,
        }
    }

    /// Borrows the vector as a system slice.
    pub fn as_slice<'s>(&'s self) -> SystemSlice<'s, Label> {
        self.slice(..)
    }

    /// Borrows the vector as a mutable system slice.
    pub fn as_mut_slice<'s>(&'s mut self) -> SystemSliceMut<'s, Label> {
        self.slice_mut(..)
    }

    /// Takes a slice of a vector.
    pub fn slice<R>(&self, range: R) -> SystemSlice<'_, Label>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let range = bounds_to_range(self.len(), range);

        SystemSlice {
            data: &self.data,
            offset: range.start,
            length: range.end - range.start,
            _marker: PhantomData,
        }
    }

    /// Takes a mutable slice of a vector.
    pub fn slice_mut<R>(&mut self, range: R) -> SystemSliceMut<'_, Label>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let range = bounds_to_range(self.len(), range);

        SystemSliceMut {
            data: &mut self.data,
            offset: range.start,
            length: range.end - range.start,
            _marker: PhantomData,
        }
    }
}

impl SystemVec<Empty> {
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            _marker: PhantomData,
        }
    }
}

/// Represents a subslice of an owned system vector.
pub struct SystemSlice<'a, Label: SystemLabel> {
    data: &'a [f64],
    offset: usize,
    length: usize,
    _marker: PhantomData<Label>,
}

impl<'a, Label: SystemLabel> SystemSlice<'a, Label> {
    /// Builds a system slice from a contiguous chunk of data.
    pub fn from_contiguous(data: &'a [f64]) -> Self {
        let mut length = 0;

        if field_count::<Label>() != 0 {
            assert!(data.len() % field_count::<Label>() == 0);
            length = data.len() / field_count::<Label>();
        }

        Self {
            data,
            offset: 0,
            length,
            _marker: PhantomData,
        }
    }

    /// Converts a system slice to a vector.
    pub fn to_vec(&self) -> SystemVec<Label> {
        let mut data = Vec::with_capacity(self.length * field_count::<Label>());

        for field in Label::fields() {
            data.extend_from_slice(self.field(field));
        }

        SystemVec::from_contiguous(data)
    }

    /// Returns the size of the system slice.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Caches immutable pointers to each field in the slice.
    pub fn fields(self) -> SystemFields<'a, Label> {
        if field_count::<Label>() == 0 {
            SystemFields {
                length: 0,
                fields: Label::FieldLike::from_fn(|_| std::ptr::null()).into(),
                _marker: PhantomData,
            }
        } else {
            let length = self.data.len() / field_count::<Label>();
            let fields = Label::FieldLike::from_fn(|index| unsafe {
                self.data.as_ptr().add(index * length + self.offset)
            })
            .into();

            SystemFields {
                length,
                fields,
                _marker: PhantomData,
            }
        }
    }

    /// Gets an immutable reference to the given field.
    pub fn field(&self, label: Label) -> &[f64] {
        let length = self.data.len() / field_count::<Label>();

        &self.data[length * label.field_index()..length * (label.field_index() + 1)]
            [self.offset..self.offset + self.length]
    }

    pub fn as_range(&self) -> SystemRange<Label> {
        SystemRange {
            ptr: self.data.as_ptr() as *mut f64,
            total: self.data.len(),
            offset: self.offset,
            length: self.length,
            _marker: PhantomData,
        }
    }

    /// Takes a subslice of the existing slice.
    pub fn slice<'s, R>(&'s self, range: R) -> SystemSlice<'s, Label>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSlice {
            data: &*self.data,
            offset: self.offset + bounds.start,
            length,
            _marker: PhantomData,
        }
    }
}

impl<'a> SystemSlice<'a, Empty> {
    pub fn empty() -> Self {
        Self {
            data: &[],
            offset: 0,
            length: 0,
            _marker: PhantomData,
        }
    }
}

impl<'long, 'short, Label: SystemLabel> Reborrow<'short> for SystemSlice<'long, Label> {
    type Target = SystemSlice<'short, Label>;

    fn rb(&'short self) -> Self::Target {
        SystemSlice {
            data: self.data,
            offset: self.offset,
            length: self.length,
            _marker: PhantomData,
        }
    }
}

impl<'a> From<&'a [f64]> for SystemSlice<'a, Scalar> {
    fn from(value: &'a [f64]) -> Self {
        SystemSlice::from_contiguous(value)
    }
}

impl<'a> From<&'a mut [f64]> for SystemSlice<'a, Scalar> {
    fn from(value: &'a mut [f64]) -> Self {
        SystemSlice::from_contiguous(value)
    }
}

/// A mutable reference to an owned system.
pub struct SystemSliceMut<'a, Label: SystemLabel> {
    data: &'a mut [f64],
    offset: usize,
    length: usize,
    _marker: PhantomData<Label>,
}

impl<'a, Label: SystemLabel> SystemSliceMut<'a, Label> {
    /// Builds a mutable system slice from contiguous data.
    pub fn from_contiguous(data: &'a mut [f64]) -> Self {
        let mut length = 0;

        if field_count::<Label>() != 0 {
            assert!(data.len() % field_count::<Label>() == 0);
            length = data.len() / field_count::<Label>();
        }

        Self {
            data,
            offset: 0,
            length,
            _marker: PhantomData,
        }
    }

    pub fn to_vec(&self) -> SystemVec<Label> {
        let mut data = Vec::with_capacity(self.length * field_count::<Label>());

        for field in Label::fields() {
            data.extend_from_slice(self.field(field));
        }

        SystemVec::from_contiguous(data)
    }

    /// The length of the slice.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if `self.len() == 0`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Caches immutable pointers to the constituent fields in the system.
    pub fn fields(self) -> SystemFields<'a, Label> {
        if field_count::<Label>() == 0 {
            SystemFields {
                length: 0,
                fields: Label::FieldLike::from_fn(|_| std::ptr::null()).into(),
                _marker: PhantomData,
            }
        } else {
            let length = self.data.len() / field_count::<Label>();
            let fields = Label::FieldLike::from_fn(|index| unsafe {
                self.data.as_ptr().add(index * length + self.offset)
            })
            .into();

            SystemFields {
                length,
                fields,
                _marker: PhantomData,
            }
        }
    }

    /// Caches mutable pointers to the constituent fields in the system.
    pub fn fields_mut(self) -> SystemFieldsMut<'a, Label> {
        if field_count::<Label>() == 0 {
            SystemFieldsMut {
                length: 0,
                fields: Label::FieldLike::from_fn(|_| std::ptr::null_mut()).into(),
                _marker: PhantomData,
            }
        } else {
            let length = self.data.len() / field_count::<Label>();
            let fields = Label::FieldLike::from_fn(|index| unsafe {
                self.data.as_mut_ptr().add(index * length + self.offset)
            })
            .into();

            SystemFieldsMut {
                length,
                fields,
                _marker: PhantomData,
            }
        }
    }

    /// Retrieves an immutable slice to the given field.
    pub fn field(&self, label: Label) -> &[f64] {
        let length = self.data.len() / field_count::<Label>();
        &self.data[length * label.field_index()..length * (label.field_index() + 1)]
            [self.offset..self.offset + self.length]
    }

    /// Retrieves a mutable slice of the given field.
    pub fn field_mut(&mut self, label: Label) -> &mut [f64] {
        let length = self.data.len() / field_count::<Label>();
        &mut self.data[length * label.field_index()..length * (label.field_index() + 1)]
            [self.offset..self.offset + self.length]
    }

    pub fn as_range(&self) -> SystemRange<Label> {
        SystemRange {
            ptr: self.data.as_ptr() as *mut f64,
            total: self.data.len(),
            offset: self.offset,
            length: self.length,
            _marker: PhantomData,
        }
    }

    /// Takes a subslice of this slice.
    pub fn slice<'s, R>(&'s self, range: R) -> SystemSlice<'s, Label>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSlice {
            data: &*self.data,
            offset: self.offset + bounds.start,
            length,
            _marker: PhantomData,
        }
    }

    /// Takes a mutable subslice of this slice.
    pub fn slice_mut<'s, R>(&'s mut self, range: R) -> SystemSliceMut<'s, Label>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSliceMut {
            data: self.data,
            offset: self.offset + bounds.start,
            length,
            _marker: PhantomData,
        }
    }
}

impl<'a> SystemSliceMut<'a, Empty> {
    pub fn empty() -> Self {
        Self {
            data: &mut [],
            offset: 0,
            length: 0,
            _marker: PhantomData,
        }
    }
}

impl<'long, 'short, Label: SystemLabel> Reborrow<'short> for SystemSliceMut<'long, Label> {
    type Target = SystemSlice<'short, Label>;

    fn rb(&'short self) -> Self::Target {
        SystemSlice {
            data: self.data,
            offset: self.offset,
            length: self.length,
            _marker: PhantomData,
        }
    }
}

impl<'long, 'short, Label: SystemLabel> ReborrowMut<'short> for SystemSliceMut<'long, Label> {
    type Target = SystemSliceMut<'short, Label>;

    fn rb_mut(&'short mut self) -> Self::Target {
        SystemSliceMut {
            data: self.data,
            offset: self.offset,
            length: self.length,
            _marker: PhantomData,
        }
    }
}

impl<'a> From<&'a mut [f64]> for SystemSliceMut<'a, Scalar> {
    fn from(value: &'a mut [f64]) -> Self {
        SystemSliceMut::from_contiguous(value)
    }
}

#[derive(Debug, Clone)]
pub struct SystemRange<Label: SystemLabel> {
    ptr: *mut f64,
    total: usize,
    offset: usize,
    length: usize,
    _marker: PhantomData<Label>,
}

impl<Label: SystemLabel> SystemRange<Label> {
    pub unsafe fn slice<R>(&self, range: R) -> SystemSlice<'_, Label>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSlice {
            data: unsafe { slice::from_raw_parts(self.ptr, self.total) },
            offset: self.offset + bounds.start,
            length,
            _marker: PhantomData,
        }
    }

    pub unsafe fn slice_mut<R>(&self, range: R) -> SystemSliceMut<'_, Label>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSliceMut {
            data: unsafe { slice::from_raw_parts_mut(self.ptr, self.total) },
            offset: self.offset + bounds.start,
            length,
            _marker: PhantomData,
        }
    }
}

unsafe impl<Label: SystemLabel> Send for SystemRange<Label> {}
unsafe impl<Label: SystemLabel> Sync for SystemRange<Label> {}

/// Converts genetic range to a concrete range type.
fn bounds_to_range<R>(total: usize, range: R) -> Range<usize>
where
    R: RangeBounds<usize>,
{
    let start_inc = match range.start_bound() {
        Bound::Included(&i) => i,
        Bound::Excluded(&i) => i + 1,
        Bound::Unbounded => 0,
    };

    let end_exc = match range.end_bound() {
        Bound::Included(&i) => i + 1,
        Bound::Excluded(&i) => i,
        Bound::Unbounded => total,
    };

    start_inc..end_exc
}

/// A cache of immutable pointers to underlying system data.
pub struct SystemFields<'a, Label: SystemLabel> {
    length: usize,
    fields: FieldArray<Label, *const f64>,
    _marker: PhantomData<&'a [f64]>,
}

impl<'a, Label: SystemLabel> SystemFields<'a, Label> {
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_fields<'s>(&'s self) -> SystemFields<'s, Label> {
        SystemFields {
            fields: self.fields.clone(),
            length: self.length,
            _marker: PhantomData,
        }
    }

    pub fn field(&self, label: Label) -> &'a [f64] {
        unsafe { std::slice::from_raw_parts(self.fields[label.field_index()], self.length) }
    }
}

/// A cache of mutable pointers to underlying system data.
pub struct SystemFieldsMut<'a, Label: SystemLabel> {
    length: usize,
    fields: FieldArray<Label, *mut f64>,
    _marker: PhantomData<&'a [f64]>,
}

impl<'a, Label: SystemLabel> SystemFieldsMut<'a, Label> {
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_fields<'s>(&'s self) -> SystemFields<'s, Label> {
        SystemFields {
            fields: Label::FieldLike::from_fn(|index| self.fields[index] as *const f64).into(),
            length: self.length,
            _marker: PhantomData,
        }
    }

    pub fn field(&self, label: Label) -> &'a [f64] {
        unsafe { std::slice::from_raw_parts(self.fields[label.field_index()], self.length) }
    }

    pub fn field_mut(&mut self, label: Label) -> &'a mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.fields[label.field_index()], self.length) }
    }
}

// ****************************
// Builtin Labels *************
// ****************************

#[derive(Clone)]
pub enum Empty {}

impl SystemLabel for Empty {
    const NAME: &'static str = "Empty";

    type FieldLike<T> = [T; 0];

    fn fields() -> Array<Self::FieldLike<Self>> {
        Array::from([])
    }

    fn field_index(&self) -> usize {
        unreachable!()
    }

    fn field_name(&self) -> String {
        unreachable!()
    }
}

#[derive(Clone)]
pub struct Scalar;

impl SystemLabel for Scalar {
    const NAME: &'static str = "Scalar";

    type FieldLike<T> = [T; 1];

    fn fields() -> Array<Self::FieldLike<Self>> {
        [Scalar].into()
    }

    fn field_index(&self) -> usize {
        0
    }

    fn field_name(&self) -> String {
        "scalar".to_string()
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
        assert_eq!(MySystem::NAME, "MySystem");
        assert_eq!(
            MySystem::fields().inner(),
            [MySystem::First, MySystem::Second, MySystem::Third],
        );
        assert_eq!(MySystem::First.field_index(), 0);
        assert_eq!(MySystem::Second.field_index(), 1);
        assert_eq!(MySystem::Third.field_index(), 2);
        assert_eq!(MySystem::First.field_name(), "First");
        assert_eq!(MySystem::Second.field_name(), "Second");
        assert_eq!(MySystem::Third.field_name(), "Third");

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
