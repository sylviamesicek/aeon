use std::ops::Range;

use reborrow::{Reborrow, ReborrowMut};

use crate::shared::SharedSlice;

use super::{System, SystemSlice, SystemSliceMut, SystemSliceShared};

/// Represents a system with a dynamical number of fields (determined at runtime).
#[derive(Clone, Debug)]
pub struct DynamicSystem {
    /// Data vector holding data for each field in SoA format.
    data: Vec<f64>,
    /// Number of fields in the system.
    count: usize,
}

impl DynamicSystem {
    /// Constructs a new empty system with `count` fields.
    pub fn new(count: usize) -> Self {
        Self {
            count,
            data: Vec::new(),
        }
    }

    /// Allocates a new system with the given length and number of fields.
    pub fn with_length(length: usize, count: usize) -> Self {
        Self {
            data: vec![0.0; count * length],
            count,
        }
    }

    /// Converts a dynamic system into a immutable reference.
    pub fn as_slice(&self) -> DynamicSystemSlice<'_> {
        DynamicSystemSlice::from_continguous(&self.data, self.count)
    }

    /// Converts a dynamic system into a mutable reference.
    pub fn as_slice_mut(&mut self) -> DynamicSystemSliceMut<'_> {
        DynamicSystemSliceMut::from_continguous(&mut self.data, self.count)
    }
}

impl System for DynamicSystem {
    type Label = usize;

    fn len(&self) -> usize {
        self.data.len() / self.count
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..self.count).into_iter()
    }
}

pub struct DynamicSystemSlice<'a> {
    data: &'a [f64],
    range: Range<usize>,
    stride: usize,
    count: usize,
}

impl<'a> DynamicSystemSlice<'a> {
    pub fn from_continguous(data: &'a [f64], count: usize) -> Self {
        debug_assert!(data.len() % count == 0);
        let stride = data.len() / count;

        Self {
            data,
            stride,
            range: 0..stride,
            count,
        }
    }
}

impl<'long, 'short> Reborrow<'short> for DynamicSystemSlice<'long> {
    type Target = DynamicSystemSlice<'short>;

    fn rb(&'short self) -> Self::Target {
        DynamicSystemSlice {
            data: self.data,
            stride: self.stride,
            range: self.range.clone(),
            count: self.count,
        }
    }
}

impl<'a> System for DynamicSystemSlice<'a> {
    type Label = usize;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..self.count).into_iter()
    }
}

impl<'a> SystemSlice<'a> for DynamicSystemSlice<'a> {
    type Slice<'b> = DynamicSystemSlice<'b> where Self: 'b;

    fn slice(&self, range: Range<usize>) -> Self::Slice<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data,
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            count: self.count,
        }
    }

    fn field(&self, label: Self::Label) -> &[f64] {
        &self.data[label * self.stride..][self.range.clone()]
    }
}

pub struct DynamicSystemSliceMut<'a> {
    data: SharedSlice<'a, f64>,
    stride: usize,
    range: Range<usize>,
    count: usize,
}

impl<'a> DynamicSystemSliceMut<'a> {
    pub fn from_continguous(data: &'a mut [f64], count: usize) -> Self {
        debug_assert!(data.len() % count == 0);
        let stride = data.len() / count;

        Self {
            data: SharedSlice::new(data),
            stride,
            range: 0..stride,
            count,
        }
    }
}

impl<'long, 'short> Reborrow<'short> for DynamicSystemSliceMut<'long> {
    type Target = DynamicSystemSliceMut<'short>;

    fn rb(&'short self) -> Self::Target {
        DynamicSystemSliceMut {
            data: self.data,
            stride: self.stride,
            range: self.range.clone(),
            count: self.count,
        }
    }
}

impl<'long, 'short> ReborrowMut<'short> for DynamicSystemSliceMut<'long> {
    type Target = DynamicSystemSliceMut<'short>;

    fn rb_mut(&'short mut self) -> Self::Target {
        DynamicSystemSliceMut {
            data: self.data,
            stride: self.stride,
            range: self.range.clone(),
            count: self.count,
        }
    }
}

impl<'a> System for DynamicSystemSliceMut<'a> {
    type Label = usize;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..self.count).into_iter()
    }
}

impl<'a> SystemSlice<'a> for DynamicSystemSliceMut<'a> {
    type Slice<'b> = DynamicSystemSliceMut<'b> where Self: 'b;

    fn slice(&self, range: Range<usize>) -> Self::Slice<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data,
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            count: self.count,
        }
    }

    fn field(&self, label: usize) -> &[f64] {
        unsafe {
            let ptr = self.data.get(label * self.stride + self.range.start) as *const f64;
            core::slice::from_raw_parts(ptr, self.range.len())
        }
    }
}

impl<'a> SystemSliceMut<'a> for DynamicSystemSliceMut<'a> {
    type SliceMut<'b> = DynamicSystemSliceMut<'b> where Self: 'b;
    type Shared = DynamicSystemSliceMut<'a>;

    fn slice_mut(&mut self, range: Range<usize>) -> Self::SliceMut<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data,
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            count: self.count,
        }
    }

    fn field_mut(&mut self, label: Self::Label) -> &mut [f64] {
        unsafe {
            let ptr = self.data.get_mut(label * self.stride + self.range.start) as *mut f64;
            core::slice::from_raw_parts_mut(ptr, self.range.len())
        }
    }

    fn into_shared(self) -> Self::Shared {
        self
    }
}

unsafe impl<'a> SystemSliceShared<'a> for DynamicSystemSliceMut<'a> {
    type Slice<'b> = DynamicSystemSliceMut<'b> where Self: 'b;
    type SliceMut<'b> = DynamicSystemSliceMut<'b> where Self: 'b;

    unsafe fn slice_unsafe(&self, range: Range<usize>) -> Self::Slice<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data.clone(),
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            count: self.count,
        }
    }

    unsafe fn slice_unsafe_mut(&self, range: Range<usize>) -> Self::SliceMut<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data.clone(),
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            count: self.count,
        }
    }
}
