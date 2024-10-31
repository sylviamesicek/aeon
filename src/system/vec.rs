use crate::system::{field_count, SystemLabel};
use aeon_array::ArrayLike;
use reborrow::{Reborrow, ReborrowMut};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::{Bound, Range, RangeBounds};
use std::slice::{self, SliceIndex};

/// Stores a system in memory as an structure of field vectors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemVec<Label: SystemLabel> {
    pub(crate) data: Vec<f64>,
    pub(crate) _marker: PhantomData<Label>,
}

impl<Label: SystemLabel> Default for SystemVec<Label> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Label: SystemLabel> SystemVec<Label> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Constructs a new system with the given length.
    pub fn with_length(length: usize) -> Self {
        Self {
            data: vec![0.0; length * field_count::<Label>()],
            _marker: PhantomData,
        }
    }

    /// Resizes the system to store the given number of nodes.
    pub fn resize(&mut self, length: usize) {
        self.data.resize(length * field_count::<Label>(), 0.0);
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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        &self.data[length * label.index()..length * (label.index() + 1)]
    }

    /// Retrieves a mutable reference to a field located at the given index.
    pub fn field_mut(&mut self, label: Label) -> &mut [f64] {
        let length = self.data.len() / field_count::<Label>();
        &mut self.data[length * label.index()..length * (label.index() + 1)]
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
    pub fn as_slice(&self) -> SystemSlice<'_, Label> {
        self.slice(..)
    }

    /// Borrows the vector as a mutable system slice.
    pub fn as_mut_slice(&mut self) -> SystemSliceMut<'_, Label> {
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

/// Represents a subslice of an owned system vector.
pub struct SystemSlice<'a, Label: SystemLabel> {
    pub(crate) data: &'a [f64],
    pub(crate) offset: usize,
    pub(crate) length: usize,
    pub(crate) _marker: PhantomData<Label>,
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

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Caches immutable pointers to each field in the slice.
    pub fn fields(self) -> SystemFields<'a, Label> {
        if field_count::<Label>() == 0 {
            SystemFields {
                length: 0,
                fields: Label::Array::from_fn(|_| std::ptr::null()),
                _marker: PhantomData,
            }
        } else {
            let length = self.data.len() / field_count::<Label>();
            let fields = Label::Array::from_fn(|field| unsafe {
                self.data.as_ptr().add(field.index() * length + self.offset)
            });

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

        &self.data[length * label.index()..length * (label.index() + 1)]
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
    pub fn slice<R>(&self, range: R) -> SystemSlice<'_, Label>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        SystemSlice {
            data: self.data,
            offset: self.offset + bounds.start,
            length,
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

/// A mutable reference to an owned system.
pub struct SystemSliceMut<'a, Label: SystemLabel> {
    pub(crate) data: &'a mut [f64],
    pub(crate) offset: usize,
    pub(crate) length: usize,
    pub(crate) _marker: PhantomData<Label>,
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
                fields: Label::Array::from_fn(|_| std::ptr::null()),
                _marker: PhantomData,
            }
        } else {
            let length = self.data.len() / field_count::<Label>();
            let fields = Label::Array::from_fn(|field| unsafe {
                self.data.as_ptr().add(field.index() * length + self.offset)
            });

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
                fields: Label::Array::from_fn(|_| std::ptr::null_mut()),
                _marker: PhantomData,
            }
        } else {
            let length = self.data.len() / field_count::<Label>();
            let fields = Label::Array::from_fn(|field| unsafe {
                self.data
                    .as_mut_ptr()
                    .add(field.index() * length + self.offset)
            });

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
        &self.data[length * label.index()..length * (label.index() + 1)]
            [self.offset..self.offset + self.length]
    }

    /// Retrieves a mutable slice of the given field.
    pub fn field_mut(&mut self, label: Label) -> &mut [f64] {
        let length = self.data.len() / field_count::<Label>();
        &mut self.data[length * label.index()..length * (label.index() + 1)]
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
    pub fn slice<R>(&self, range: R) -> SystemSlice<'_, Label>
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
    pub fn slice_mut<R>(&mut self, range: R) -> SystemSliceMut<'_, Label>
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

/// An unsafe pointer to a range of a system.
#[derive(Debug, Clone)]
pub struct SystemRange<Label: SystemLabel> {
    ptr: *mut f64,
    total: usize,
    offset: usize,
    length: usize,
    _marker: PhantomData<Label>,
}

impl<Label: SystemLabel> SystemRange<Label> {
    /// Retrieves an immutable reference to a slice of the system.
    ///
    /// # Safety
    /// No other mutable refernces may refer to any element of this slice while it is alive.
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

    /// Retrieves a mutable reference to a slice of the system.
    ///
    /// # Safety
    /// No other refernces may refer to any element of this slice while it is alive.
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
    pub(crate) length: usize,
    pub(crate) fields: Label::Array<*const f64>,
    pub(crate) _marker: PhantomData<&'a [f64]>,
}

impl<'a, Label: SystemLabel> SystemFields<'a, Label> {
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn field(&self, label: Label) -> &'a [f64] {
        unsafe { std::slice::from_raw_parts(self.fields[label], self.length) }
    }
}

impl<'long, 'short, Label: SystemLabel> Reborrow<'short> for SystemFields<'long, Label> {
    type Target = SystemFields<'short, Label>;

    fn rb(&'short self) -> Self::Target {
        SystemFields {
            fields: Label::Array::from_fn(|i| self.fields[i]),
            length: self.length,
            _marker: PhantomData,
        }
    }
}

/// A cache of mutable pointers to underlying system data.
pub struct SystemFieldsMut<'a, Label: SystemLabel> {
    pub(crate) length: usize,
    pub(crate) fields: Label::Array<*mut f64>,
    pub(crate) _marker: PhantomData<&'a [f64]>,
}

impl<'a, Label: SystemLabel> SystemFieldsMut<'a, Label> {
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn field(&self, label: Label) -> &'a [f64] {
        unsafe { std::slice::from_raw_parts(self.fields[label], self.length) }
    }

    pub fn field_mut(&mut self, label: Label) -> &'a mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.fields[label], self.length) }
    }
}

impl<'long, 'short, Label: SystemLabel> Reborrow<'short> for SystemFieldsMut<'long, Label> {
    type Target = SystemFields<'short, Label>;

    fn rb(&'short self) -> Self::Target {
        SystemFields {
            fields: Label::Array::from_fn(|index| self.fields[index] as *const f64),
            length: self.length,
            _marker: PhantomData,
        }
    }
}

impl<'long, 'short, Label: SystemLabel> ReborrowMut<'short> for SystemFieldsMut<'long, Label> {
    type Target = SystemFieldsMut<'short, Label>;

    fn rb_mut(&'short mut self) -> Self::Target {
        SystemFieldsMut {
            fields: Label::Array::from_fn(|index| self.fields[index]),
            length: self.length,
            _marker: PhantomData,
        }
    }
}
