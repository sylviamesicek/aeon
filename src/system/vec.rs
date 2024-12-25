use super::*;

use reborrow::{Reborrow, ReborrowMut};
use serde::{Deserialize, Serialize};
use std::ops::{Bound, Range, RangeBounds};
use std::slice::SliceIndex;

/// Stores a system in memory as an structure of field vectors.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SystemVec<S> {
    pub(crate) data: Vec<f64>,
    pub(crate) system: S,
}

impl<S: System> SystemVec<S> {
    pub fn new(system: S) -> Self {
        Self {
            data: Vec::new(),
            system,
        }
    }

    /// Constructs a new system with the given length.
    pub fn with_length(length: usize, system: S) -> Self {
        Self {
            data: vec![0.0; length * system.count()],
            system,
        }
    }

    pub fn system(&self) -> &S {
        &self.system
    }

    /// Resizes the system to store the given number of nodes.
    pub fn resize(&mut self, length: usize) {
        self.data.resize(length * self.system.count(), 0.0);
    }

    /// Constructs a new system vector from the given data. `data.len()` must be divisible by `field_count::<Label>()`.
    pub fn from_contiguous(data: Vec<f64>, system: S) -> Self {
        debug_assert!(data.len() % system.count() == 0);
        Self { data, system }
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
        if self.system.count() == 0 {
            0
        } else {
            self.data.len() / self.system.count()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieves an immutable reference to a field located at the given index.
    pub fn field(&self, label: S::Label) -> &[f64] {
        let stride = self.data.len() / self.system.count();
        let index = self.system.label_index(label);
        &self.data[stride * index..stride * (index + 1)]
    }

    /// Retrieves a mutable reference to a field located at the given index.
    pub fn field_mut(&mut self, label: S::Label) -> &mut [f64] {
        let stride = self.data.len() / self.system.count();
        let index = self.system.label_index(label);

        &mut self.data[stride * index..stride * (index + 1)]
    }

    /// Borrows the vector as a system slice.
    pub fn as_slice(&self) -> SystemSlice<'_, S> {
        self.slice(..)
    }

    /// Borrows the vector as a mutable system slice.
    pub fn as_mut_slice(&mut self) -> SystemSliceMut<'_, S> {
        self.slice_mut(..)
    }

    /// Takes a slice of a vector.
    pub fn slice<R>(&self, range: R) -> SystemSlice<'_, S>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let range = bounds_to_range(self.len(), range);
        SystemSlice {
            ptr: self.data.as_ptr(),
            total: self.data.len(),
            offset: range.start,
            length: range.end - range.start,
            system: &self.system,
        }
    }

    /// Takes a mutable slice of a vector.
    pub fn slice_mut<R>(&mut self, range: R) -> SystemSliceMut<'_, S>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let range = bounds_to_range(self.len(), range);

        SystemSliceMut {
            ptr: self.data.as_mut_ptr(),
            total: self.data.len(),
            offset: range.start,
            length: range.end - range.start,
            system: &self.system,
        }
    }
}

#[derive(Clone)]
/// Represents a subslice of an owned system vector.
pub struct SystemSlice<'a, S> {
    pub(crate) ptr: *const f64,
    pub(crate) total: usize,
    pub(crate) offset: usize,
    pub(crate) length: usize,
    pub(crate) system: &'a S,
}

impl<'a> SystemSlice<'a, Empty> {
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            total: 0,
            offset: 0,
            length: 0,
            system: &Empty,
        }
    }
}

impl<'a> SystemSlice<'a, Scalar> {
    pub fn from_scalar(data: &'a [f64]) -> Self {
        Self::from_contiguous(data, &Scalar)
    }

    pub fn into_scalar(self) -> &'a [f64] {
        unsafe { std::slice::from_raw_parts(self.ptr.add(self.offset), self.length) }
    }
}

impl<'a> From<&'a [f64]> for SystemSlice<'a, Scalar> {
    fn from(value: &'a [f64]) -> Self {
        Self::from_scalar(value)
    }
}

impl<'a, S: System> SystemSlice<'a, S> {
    /// Builds a system slice from a contiguous chunk of data.
    pub fn from_contiguous(data: &'a [f64], system: &'a S) -> Self {
        let mut length = 0;

        if system.count() != 0 {
            assert!(data.len() % system.count() == 0);
            length = data.len() / system.count();
        }

        Self {
            ptr: data.as_ptr(),
            total: data.len(),
            offset: 0,
            length,
            system,
        }
    }

    /// Returns the size of the system slice.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn system(&self) -> &'a S {
        self.system
    }

    fn stride(&self) -> usize {
        debug_assert!(self.system.count() >= 1);
        self.total / self.system.count()
    }

    /// Gets an immutable reference to the given field.
    pub fn field(&self, label: S::Label) -> &[f64] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr
                    .add(self.stride() * self.system.label_index(label) + self.offset),
                self.length,
            )
        }
    }

    /// Takes a subslice of the existing slice.
    pub fn slice<R>(&self, range: R) -> SystemSlice<'_, S>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSlice {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset + bounds.start,
            length,
            system: &self.system,
        }
    }
}

impl<'a, S: System + Clone> SystemSlice<'a, S> {
    /// Converts a system slice to a vector.
    pub fn to_vec(&self) -> SystemVec<S> {
        let mut data = Vec::with_capacity(self.length * self.system.count());

        for field in self.system.enumerate() {
            data.extend_from_slice(self.field(field));
        }

        SystemVec::from_contiguous(data, self.system.clone())
    }
}

impl<'long, 'short, S> Reborrow<'short> for SystemSlice<'long, S> {
    type Target = SystemSlice<'short, S>;

    fn rb(&'short self) -> Self::Target {
        SystemSlice {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset,
            length: self.length,
            system: self.system,
        }
    }
}

unsafe impl<'a, S> Send for SystemSlice<'a, S> {}
unsafe impl<'a, S> Sync for SystemSlice<'a, S> {}

/// A mutable reference to an owned system.
pub struct SystemSliceMut<'a, S> {
    pub(crate) ptr: *mut f64,
    pub(crate) total: usize,
    pub(crate) offset: usize,
    pub(crate) length: usize,
    pub(crate) system: &'a S,
}

impl<'a> SystemSliceMut<'a, Scalar> {
    pub fn from_scalar(data: &'a mut [f64]) -> Self {
        Self::from_contiguous(data, &Scalar)
    }

    pub fn into_scalar(self) -> &'a mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(self.offset), self.length) }
    }
}

impl<'a> From<&'a mut [f64]> for SystemSliceMut<'a, Scalar> {
    fn from(value: &'a mut [f64]) -> Self {
        Self::from_scalar(value)
    }
}

impl<'a, S: System> SystemSliceMut<'a, S> {
    /// Builds a mutable system slice from contiguous data.
    pub fn from_contiguous(data: &'a mut [f64], system: &'a S) -> Self {
        let mut length = 0;

        if system.count() != 0 {
            assert!(data.len() % system.count() == 0);
            length = data.len() / system.count();
        }

        Self {
            ptr: data.as_mut_ptr(),
            total: data.len(),
            offset: 0,
            length,
            system,
        }
    }

    /// Number of dofs in the slice.
    pub fn total_dofs(&self) -> usize {
        self.total
    }

    /// The length of the slice.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if `self.len() == 0`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn stride(&self) -> usize {
        debug_assert!(self.system.count() >= 1);
        self.total / self.system.count()
    }

    pub fn system(&self) -> &'a S {
        self.system
    }

    /// Retrieves an immutable slice to the given field.
    pub fn field(&self, label: S::Label) -> &[f64] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr
                    .add(self.stride() * self.system.label_index(label) + self.offset),
                self.length,
            )
        }
    }

    /// Retrieves a mutable slice of the given field.
    pub fn field_mut(&mut self, label: S::Label) -> &mut [f64] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr
                    .add(self.stride() * self.system.label_index(label) + self.offset),
                self.length,
            )
        }
    }

    /// Takes a subslice of this slice.
    pub fn slice<R>(&self, range: R) -> SystemSlice<'_, S>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSlice {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset + bounds.start,
            length,
            system: &self.system,
        }
    }

    /// Takes a mutable subslice of this slice.
    pub fn slice_mut<R>(&mut self, range: R) -> SystemSliceMut<'_, S>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSliceMut {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset + bounds.start,
            length,
            system: &self.system,
        }
    }

    pub fn into_shared(self) -> SystemSliceShared<'a, S> {
        SystemSliceShared {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset,
            length: self.length,
            system: self.system,
        }
    }
}

impl<'a, S: System + Clone> SystemSliceMut<'a, S> {
    /// Converts a system slice to a vector.
    pub fn to_vec(&self) -> SystemVec<S> {
        let mut data = Vec::with_capacity(self.length * self.system.count());

        for field in self.system.enumerate() {
            data.extend_from_slice(self.field(field));
        }

        SystemVec::from_contiguous(data, self.system.clone())
    }
}

impl<'long, 'short, S> Reborrow<'short> for SystemSliceMut<'long, S> {
    type Target = SystemSlice<'short, S>;

    fn rb(&'short self) -> Self::Target {
        SystemSlice {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset,
            length: self.length,
            system: &self.system,
        }
    }
}

impl<'long, 'short, S> ReborrowMut<'short> for SystemSliceMut<'long, S> {
    type Target = SystemSliceMut<'short, S>;

    fn rb_mut(&'short mut self) -> Self::Target {
        SystemSliceMut {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset,
            length: self.length,
            system: &self.system,
        }
    }
}

unsafe impl<'a, S> Send for SystemSliceMut<'a, S> {}
unsafe impl<'a, S> Sync for SystemSliceMut<'a, S> {}

/// An unsafe pointer to a range of a system.
#[derive(Debug, Clone)]
pub struct SystemSliceShared<'a, S> {
    ptr: *mut f64,
    total: usize,
    offset: usize,
    length: usize,
    system: &'a S,
}

impl<'a, S: System> SystemSliceShared<'a, S> {
    pub fn system(&self) -> &S {
        self.system
    }

    /// Retrieves an immutable reference to a slice of the system.
    ///
    /// # Safety
    /// No other mutable refernces may refer to any element of this slice while it is alive.
    pub unsafe fn slice<R>(&self, range: R) -> SystemSlice<'_, S>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSlice {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset + bounds.start,
            length,
            system: &self.system,
        }
    }

    /// Retrieves a mutable reference to a slice of the system.
    ///
    /// # Safety
    /// No other refernces may refer to any element of this slice while it is alive.
    pub unsafe fn slice_mut<R>(&self, range: R) -> SystemSliceMut<'_, S>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSliceMut {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset + bounds.start,
            length,
            system: &self.system,
        }
    }
}

unsafe impl<'a, S> Send for SystemSliceShared<'a, S> {}
unsafe impl<'a, S> Sync for SystemSliceShared<'a, S> {}

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
