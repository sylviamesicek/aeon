use reborrow::{Reborrow, ReborrowMut};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::{Bound, Range, RangeBounds};
use std::slice::SliceIndex;

#[derive(Clone, Debug, Default, Serialize, Deserialize, datasize::DataSize)]
pub struct Image {
    data: Vec<f64>,
    channels: usize,
}

impl Image {
    pub fn new(channels: usize, nodes: usize) -> Self {
        Self {
            data: vec![0.0; channels * nodes],
            channels,
        }
    }

    pub fn reinit(&mut self, channels: usize, nodes: usize) {
        self.data.resize(channels * nodes, 0.0);
        self.channels = channels;
    }

    pub fn resize(&mut self, nodes: usize) {
        self.reinit(self.channels, nodes);
    }

    pub fn len(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }

        self.data.len() / self.channels
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Constructs a new system vector from the given data. `data.len()` must be divisible by `field_count::<Label>()`.
    pub fn from_storage(data: Vec<f64>, channels: usize) -> Self {
        debug_assert!(data.len() % channels == 0);
        Self { data, channels }
    }

    /// Transforms a system vector back into a linear vector
    pub fn into_storage(self) -> Vec<f64> {
        self.data
    }

    pub fn storage(&self) -> &[f64] {
        &self.data
    }

    pub fn storage_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    pub fn num_channels(&self) -> usize {
        self.channels
    }

    pub fn channels(&self) -> Range<usize> {
        0..self.channels
    }

    pub fn channel(&self, channel: usize) -> &[f64] {
        let stride = self.data.len() / self.channels;
        &self.data[stride * channel..stride * (channel + 1)]
    }

    pub fn channel_mut(&mut self, channel: usize) -> &mut [f64] {
        let stride = self.data.len() / self.channels;
        &mut self.data[stride * channel..stride * (channel + 1)]
    }
}

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

/// Represents a subslice of an owned system vector.
#[derive(Clone, Copy)]
pub struct ImageRef<'a> {
    ptr: *const f64,
    total: usize,
    offset: usize,
    length: usize,
    channels: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a> ImageRef<'a> {
    /// Builds a system slice from a contiguous chunk of data.
    pub fn from_storage(data: &'a [f64], channels: usize) -> Self {
        let mut length = 0;

        if channels != 0 {
            assert!(data.len() % channels == 0);
            length = data.len() / channels;
        }

        Self {
            ptr: data.as_ptr(),
            total: data.len(),
            offset: 0,
            length,
            channels,
            _marker: PhantomData,
        }
    }

    /// Returns the size of the system slice.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn num_channels(&self) -> usize {
        self.channels
    }

    pub fn channels(&self) -> Range<usize> {
        0..self.channels
    }

    fn stride(&self) -> usize {
        debug_assert!(self.channels >= 1);
        self.total / self.channels
    }

    /// Gets an immutable reference to the given field.
    pub fn channel(&self, channel: usize) -> &[f64] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr.add(self.stride() * channel + self.offset),
                self.length,
            )
        }
    }

    /// Takes a subslice of the existing slice.
    pub fn slice<R>(&self, range: R) -> ImageRef<'_>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        ImageRef {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset + bounds.start,
            length,
            channels: self.channels,
            _marker: PhantomData,
        }
    }

    pub fn to_owned(&self) -> Image {
        let mut data = Vec::with_capacity(self.length * self.channels);

        for channel in 0..self.channels {
            data.extend_from_slice(self.channel(channel));
        }

        Image::from_storage(data, self.channels)
    }
}

impl<'short> Reborrow<'short> for ImageRef<'_> {
    type Target = ImageRef<'short>;

    fn rb(&'short self) -> Self::Target {
        ImageRef {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset,
            length: self.length,
            channels: self.channels,
            _marker: PhantomData,
        }
    }
}

impl<'a> From<&'a [f64]> for ImageRef<'a> {
    fn from(value: &'a [f64]) -> Self {
        ImageRef {
            ptr: value.as_ptr(),
            total: value.len(),
            offset: 0,
            length: value.len(),
            channels: 1,
            _marker: PhantomData,
        }
    }
}

impl<'a> From<&'a mut [f64]> for ImageRef<'a> {
    fn from(value: &'a mut [f64]) -> Self {
        ImageRef {
            ptr: value.as_ptr(),
            total: value.len(),
            offset: 0,
            length: value.len(),
            channels: 1,
            _marker: PhantomData,
        }
    }
}

unsafe impl Send for ImageRef<'_> {}
unsafe impl Sync for ImageRef<'_> {}

/// A mutable reference to an owned system.
pub struct ImageMut<'a> {
    ptr: *mut f64,
    total: usize,
    offset: usize,
    length: usize,
    channels: usize,
    _marker: PhantomData<&'a mut ()>,
}

impl<'a> ImageMut<'a> {
    /// Builds a mutable system slice from contiguous data.
    pub fn from_storage(data: &'a mut [f64], channels: usize) -> Self {
        let mut length = 0;

        if channels != 0 {
            assert!(data.len() % channels == 0);
            length = data.len() / channels;
        }

        Self {
            ptr: data.as_mut_ptr(),
            total: data.len(),
            offset: 0,
            length,
            channels,
            _marker: PhantomData,
        }
    }

    /// Returns the size of the system slice.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn num_channels(&self) -> usize {
        self.channels
    }

    pub fn channels(&self) -> Range<usize> {
        0..self.channels
    }

    fn stride(&self) -> usize {
        debug_assert!(self.channels >= 1);
        self.total / self.channels
    }

    /// Gets an immutable reference to the given field.
    pub fn channel(&self, channel: usize) -> &[f64] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr.add(self.stride() * channel + self.offset),
                self.length,
            )
        }
    }

    /// Retrieves a mutable slice of the given field.
    pub fn channel_mut(&mut self, channel: usize) -> &mut [f64] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.add(self.stride() * channel + self.offset),
                self.length,
            )
        }
    }

    /// Takes a subslice of this slice.
    pub fn slice<R>(&self, range: R) -> ImageRef<'_>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        ImageRef {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset + bounds.start,
            length,
            channels: self.channels,
            _marker: PhantomData,
        }
    }

    /// Takes a mutable subslice of this slice.
    pub fn slice_mut<R>(&mut self, range: R) -> ImageMut<'_>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        ImageMut {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset + bounds.start,
            length,
            channels: self.channels,
            _marker: PhantomData,
        }
    }

    pub fn to_owned(&self) -> Image {
        let mut data = Vec::with_capacity(self.length * self.channels);

        for channel in 0..self.channels {
            data.extend_from_slice(self.channel(channel));
        }

        Image::from_storage(data, self.channels)
    }
}

impl<'a> From<&'a mut [f64]> for ImageMut<'a> {
    fn from(value: &'a mut [f64]) -> Self {
        ImageMut {
            ptr: value.as_mut_ptr(),
            total: value.len(),
            offset: 0,
            length: value.len(),
            channels: 1,
            _marker: PhantomData,
        }
    }
}

impl<'short> Reborrow<'short> for ImageMut<'_> {
    type Target = ImageRef<'short>;

    fn rb(&'short self) -> Self::Target {
        ImageRef {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset,
            length: self.length,
            channels: self.channels,
            _marker: PhantomData,
        }
    }
}

impl<'short> ReborrowMut<'short> for ImageMut<'_> {
    type Target = ImageMut<'short>;

    fn rb_mut(&'short mut self) -> Self::Target {
        ImageMut {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset,
            length: self.length,
            channels: self.channels,
            _marker: PhantomData,
        }
    }
}

unsafe impl Send for ImageMut<'_> {}
unsafe impl Sync for ImageMut<'_> {}

/// An unsafe pointer to a range of a system.
#[derive(Debug, Clone)]
pub struct ImageShared<'a> {
    ptr: *mut f64,
    total: usize,
    offset: usize,
    length: usize,
    channels: usize,
    _marker: PhantomData<&'a mut ()>,
}

impl ImageShared<'_> {
    /// Returns the size of the system slice.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn num_channels(&self) -> usize {
        self.channels
    }

    fn stride(&self) -> usize {
        debug_assert!(self.channels >= 1);
        self.total / self.channels
    }

    /// Retrieves an immutable slice to the given field.
    pub unsafe fn channel(&self, channel: usize) -> &[f64] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr.add(self.stride() * channel + self.offset),
                self.length,
            )
        }
    }

    /// Retrieves a mutable slice of the given field.
    pub unsafe fn channel_mut(&self, channel: usize) -> &mut [f64] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.add(self.stride() * channel + self.offset),
                self.length,
            )
        }
    }

    /// Retrieves an immutable reference to a slice of the system.
    ///
    /// # Safety
    /// No other mutable refernces may refer to any element of this slice while it is alive.
    pub unsafe fn slice<R>(&self, range: R) -> ImageRef<'_>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        ImageRef {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset + bounds.start,
            length,
            channels: self.channels,
            _marker: PhantomData,
        }
    }

    /// Retrieves a mutable reference to a slice of the system.
    ///
    /// # Safety
    /// No other refernces may refer to any element of this slice while it is alive.
    pub unsafe fn slice_mut<R>(&self, range: R) -> ImageMut<'_>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.length, range);
        let length = bounds.end - bounds.start;

        ImageMut {
            ptr: self.ptr,
            total: self.total,
            offset: self.offset + bounds.start,
            length,
            channels: self.channels,
            _marker: PhantomData,
        }
    }
}

impl<'a> From<ImageMut<'a>> for ImageShared<'a> {
    fn from(value: ImageMut<'a>) -> Self {
        ImageShared {
            ptr: value.ptr,
            total: value.total,
            offset: value.offset,
            length: value.length,
            channels: value.channels,
            _marker: PhantomData,
        }
    }
}

unsafe impl Send for ImageShared<'_> {}
unsafe impl Sync for ImageShared<'_> {}
