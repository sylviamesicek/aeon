use reborrow::{Reborrow, ReborrowMut};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::{Bound, Range, RangeBounds};
use std::slice::SliceIndex;

/// Several fields of data spread among
#[derive(Clone, Debug, Default, Serialize, Deserialize, datasize::DataSize)]
pub struct Image {
    data: Vec<f64>,
    channels: usize,
}

impl Image {
    /// Allocates a new image with the specified number of channels and nodes
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

    pub fn num_nodes(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }

        self.data.len() / self.channels
    }

    pub fn is_empty(&self) -> bool {
        self.num_nodes() == 0 || self.num_channels() == 0
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

    pub fn as_ref(&self) -> ImageRef {
        ImageRef::from_storage(&self.data, self.channels)
    }

    pub fn as_mut(&mut self) -> ImageMut {
        ImageMut::from_storage(&mut self.data, self.channels)
    }

    pub fn slice<R>(&self, range: R) -> ImageRef<'_>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.num_nodes(), range);
        let length = bounds.end - bounds.start;

        ImageRef {
            ptr: self.data.as_ptr(),
            total: self.data.len(),
            offset: bounds.start,
            length,
            channels: self.channels,
            _marker: PhantomData,
        }
    }

    pub fn slice_mut<R>(&mut self, range: R) -> ImageMut<'_>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let bounds = bounds_to_range(self.num_nodes(), range);
        let length = bounds.end - bounds.start;

        ImageMut {
            ptr: self.data.as_mut_ptr(),
            total: self.data.len(),
            offset: bounds.start,
            length,
            channels: self.channels,
            _marker: PhantomData,
        }
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
    pub fn empty() -> Self {
        Self::from_storage(&[], 0)
    }

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
    pub fn num_nodes(&self) -> usize {
        self.length
    }

    // pub fn len(&self) -> usize {
    //     self.num_nodes() * self.num_channels()
    // }

    pub fn is_empty(&self) -> bool {
        self.length == 0 || self.channels == 0
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

    pub fn split_channels(self, split: usize) -> (ImageRef<'a>, ImageRef<'a>) {
        assert!(split <= self.channels);

        let left_channels = split;
        let right_channels = self.channels - split;

        let ptr = self.ptr;
        let length = self.length;
        let offset = self.offset;

        let left_total = left_channels * self.stride();
        let right_total = right_channels * self.stride();

        debug_assert_eq!(left_total + right_total, self.total);

        let left_ptr = ptr;
        let right_ptr = unsafe { ptr.add(left_total) };

        (
            ImageRef {
                ptr: left_ptr,
                total: left_total,
                offset,
                length,
                channels: left_channels,
                _marker: PhantomData,
            },
            ImageRef {
                ptr: right_ptr,
                total: right_total,
                offset,
                length,
                channels: right_channels,
                _marker: PhantomData,
            },
        )
    }

    /// Gets an immutable reference to the given field.
    pub fn channel(&self, channel: usize) -> &[f64] {
        debug_assert!(channel < self.num_channels());

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

        debug_assert!(self.channels == 0 || length <= self.length);

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
    pub fn num_nodes(&self) -> usize {
        self.length
    }

    // pub fn len(&self) -> usize {
    //     self.length * self.channels
    // }

    pub fn is_empty(&self) -> bool {
        self.length == 0 || self.channels == 0
    }

    pub fn num_channels(&self) -> usize {
        self.channels
    }

    pub fn channels(&self) -> Range<usize> {
        0..self.channels
    }

    pub fn split_channels(self, split: usize) -> (ImageMut<'a>, ImageMut<'a>) {
        assert!(split < self.channels);
        let left_channels = split;
        let right_channels = self.channels - split;

        let ptr = self.ptr;
        let length = self.length;
        let offset = self.offset;

        let left_total = left_channels * self.stride();
        let right_total = right_channels * self.stride();

        debug_assert_eq!(left_total + right_total, self.total);

        let left_ptr = ptr;
        let right_ptr = unsafe { ptr.add(left_total) };

        (
            ImageMut {
                ptr: left_ptr,
                total: left_total,
                offset,
                length,
                channels: left_channels,
                _marker: PhantomData,
            },
            ImageMut {
                ptr: right_ptr,
                total: right_total,
                offset,
                length,
                channels: right_channels,
                _marker: PhantomData,
            },
        )
    }

    fn stride(&self) -> usize {
        debug_assert!(self.channels >= 1);
        self.total / self.channels
    }

    /// Gets an immutable reference to the given field.
    pub fn channel(&self, channel: usize) -> &[f64] {
        debug_assert!(channel < self.num_channels());

        unsafe {
            std::slice::from_raw_parts(
                self.ptr.add(self.stride() * channel + self.offset),
                self.length,
            )
        }
    }

    /// Retrieves a mutable slice of the given field.
    pub fn channel_mut(&mut self, channel: usize) -> &mut [f64] {
        debug_assert!(channel < self.num_channels());

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

        debug_assert!(self.channels == 0 || length <= self.length);

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

        debug_assert!(self.channels == 0 || length <= self.length);

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
    pub fn num_nodes(&self) -> usize {
        self.length
    }

    // pub fn len(&self) -> usize {
    //     self.length * self.channels
    // }

    pub fn is_empty(&self) -> bool {
        self.length == 0 || self.num_channels() == 0
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

    /// Retrieves an immutable slice to the given field.
    pub unsafe fn channel(&self, channel: usize) -> &[f64] {
        debug_assert!(channel < self.num_channels());

        unsafe {
            std::slice::from_raw_parts(
                self.ptr.add(self.stride() * channel + self.offset),
                self.length,
            )
        }
    }

    /// Retrieves a mutable slice of the given field.
    pub unsafe fn channel_mut(&self, channel: usize) -> &mut [f64] {
        debug_assert!(channel < self.num_channels());

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

        debug_assert!(self.channels == 0 || length <= self.length);

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

        debug_assert!(self.channels == 0 || length <= self.length);

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

#[cfg(test)]
mod tests {
    use super::*;

    const FIRST_CH: usize = 0;
    const SECOND_CH: usize = 1;
    const THIRD_CH: usize = 2;

    /// Test of basic system functionality.
    #[test]
    fn basic() {
        let mut fields = Image::new(3, 3);

        {
            let shared: ImageShared = fields.as_mut().into();
            let mut slice = unsafe { shared.slice_mut(1..2) };

            slice.channel_mut(FIRST_CH).fill(1.0);
            slice.channel_mut(SECOND_CH).fill(2.0);
            slice.channel_mut(THIRD_CH).fill(3.0);
        }

        let buffer = fields.storage();

        assert_eq!(buffer, &[0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0]);

        let empty = Image::new(0, 0);
        assert!(empty.is_empty());
    }

    /// A simple test of creating composite (pair) systems, splitting them, and taking various references.
    #[test]
    fn pair() {
        let mut data = Image::new(5, 10);

        data.channel_mut(0).fill(0.0);
        data.channel_mut(1).fill(1.0);
        data.channel_mut(2).fill(2.0);
        data.channel_mut(3).fill(3.0);
        data.channel_mut(4).fill(4.0);

        {
            let data = data.as_ref();
            let (left, right) = data.split_channels(3);
            assert_eq!(left.num_channels(), 3);
            assert_eq!(right.num_channels(), 2);

            assert!(left.channel(0).iter().all(|v| *v == 0.0));
            assert!(left.channel(1).iter().all(|v| *v == 1.0));
            assert!(left.channel(2).iter().all(|v| *v == 2.0));
            assert!(right.channel(0).iter().all(|v| *v == 3.0));
            assert!(right.channel(1).iter().all(|v| *v == 4.0));
        }

        {
            let slice: ImageMut<'_> = ImageMut::from_storage(data.storage_mut(), 5);
            let (left, right) = slice.split_channels(3);

            assert!(left.channel(0).iter().all(|v| *v == 0.0));
            assert!(left.channel(1).iter().all(|v| *v == 1.0));
            assert!(left.channel(2).iter().all(|v| *v == 2.0));
            assert!(right.channel(0).iter().all(|v| *v == 3.0));
            assert!(right.channel(1).iter().all(|v| *v == 4.0));
        }

        let data = (0..15).map(|i| i as f64).collect::<Vec<_>>();
        let image = Image::from_storage(data, 3);

        {
            let image = image.as_ref();
            let (left, right) = image.split_channels(2);

            assert_eq!(left.channel(0), &[0.0, 1.0, 2.0, 3.0, 4.0]);
            assert_eq!(left.channel(1), &[5.0, 6.0, 7.0, 8.0, 9.0]);
            assert_eq!(right.channel(0), &[10.0, 11.0, 12.0, 13.0, 14.0]);
        }
        {
            let image = image.as_ref();
            let (slice1, slice2) = image.slice(2..4).split_channels(2);
            assert_eq!(slice1.channel(0), &[2.0, 3.0]);
            assert_eq!(slice1.channel(1), &[7.0, 8.0]);
            assert_eq!(slice2.channel(0), &[12.0, 13.0]);
        }
    }
}
