use std::ops::Range;

use reborrow::{Reborrow, ReborrowMut};

use crate::shared::SharedSlice;

use super::{System, SystemSlice, SystemSliceMut, SystemSliceShared};

#[derive(Clone, Debug)]
pub struct StaticSystem<const COUNT: usize> {
    data: Vec<f64>,
}

impl<const COUNT: usize> StaticSystem<COUNT> {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn with_length(length: usize) -> Self {
        Self {
            data: vec![0.0; length * COUNT],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len() / COUNT
    }

    pub fn as_slice(&self) -> StaticSystemSlice<'_, COUNT> {
        StaticSystemSlice::from_continguous(&self.data)
    }

    pub fn as_slice_mut(&mut self) -> StaticSystemSliceMut<'_, COUNT> {
        StaticSystemSliceMut::from_continguous(&mut self.data)
    }
}

pub struct StaticSystemSlice<'a, const COUNT: usize> {
    data: &'a [f64],
    stride: usize,
    range: Range<usize>,
}

impl<'a, const COUNT: usize> StaticSystemSlice<'a, COUNT> {
    pub fn from_continguous(data: &'a [f64]) -> Self {
        debug_assert!(data.len() % COUNT == 0);
        let stride = data.len() / COUNT;

        Self {
            data: data,
            stride,
            range: 0..stride,
        }
    }
}

impl<'long, 'short, const COUNT: usize> Reborrow<'short> for StaticSystemSlice<'long, COUNT> {
    type Target = StaticSystemSlice<'short, COUNT>;

    fn rb(&'short self) -> Self::Target {
        StaticSystemSlice {
            data: self.data,
            stride: self.stride,
            range: self.range.clone(),
        }
    }
}

impl<'a, const COUNT: usize> System for StaticSystemSlice<'a, COUNT> {
    type Label = usize;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..COUNT).into_iter()
    }
}

impl<'a, const COUNT: usize> SystemSlice<'a> for StaticSystemSlice<'a, COUNT> {
    type Slice<'b> = StaticSystemSlice<'b, COUNT> where Self: 'b;

    fn slice(&self, range: Range<usize>) -> Self::Slice<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            stride: self.stride,
        }
    }

    fn field(&self, label: Self::Label) -> &[f64] {
        &self.data[label * self.stride..][self.range.clone()]
    }
}

pub struct StaticSystemSliceMut<'a, const COUNT: usize> {
    data: SharedSlice<'a, f64>,
    stride: usize,
    range: Range<usize>,
}

impl<'long, 'short, const COUNT: usize> Reborrow<'short> for StaticSystemSliceMut<'long, COUNT> {
    type Target = StaticSystemSliceMut<'short, COUNT>;

    fn rb(&'short self) -> Self::Target {
        StaticSystemSliceMut {
            data: self.data,
            stride: self.stride,
            range: self.range.clone(),
        }
    }
}

impl<'long, 'short, const COUNT: usize> ReborrowMut<'short> for StaticSystemSliceMut<'long, COUNT> {
    type Target = StaticSystemSliceMut<'short, COUNT>;

    fn rb_mut(&'short mut self) -> Self::Target {
        StaticSystemSliceMut {
            data: self.data,
            stride: self.stride,
            range: self.range.clone(),
        }
    }
}

impl<'a, const COUNT: usize> StaticSystemSliceMut<'a, COUNT> {
    pub fn from_continguous(data: &'a mut [f64]) -> Self {
        debug_assert!(data.len() % COUNT == 0);
        let stride = data.len() / COUNT;

        Self {
            range: 0..stride,
            stride,
            data: SharedSlice::new(data),
        }
    }
}

impl<'a, const COUNT: usize> System for StaticSystemSliceMut<'a, COUNT> {
    type Label = usize;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..COUNT).into_iter()
    }
}

impl<'a, const COUNT: usize> SystemSlice<'a> for StaticSystemSliceMut<'a, COUNT> {
    type Slice<'b> = StaticSystemSliceMut<'b, COUNT> where Self: 'b;

    fn slice(&self, range: Range<usize>) -> Self::Slice<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data.clone(),
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
        }
    }

    fn field(&self, label: usize) -> &[f64] {
        unsafe {
            let ptr = self.data.get(label * self.stride + self.range.start) as *const f64;
            core::slice::from_raw_parts(ptr, self.range.len())
        }
    }
}

impl<'a, const COUNT: usize> SystemSliceMut<'a> for StaticSystemSliceMut<'a, COUNT> {
    type SliceMut<'b> = StaticSystemSliceMut<'b, COUNT> where Self: 'b;
    type Shared = StaticSystemSliceMut<'a, COUNT>;

    fn slice_mut(&mut self, range: Range<usize>) -> Self::SliceMut<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data.clone(),
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
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

unsafe impl<'a, const COUNT: usize> SystemSliceShared<'a> for StaticSystemSliceMut<'a, COUNT> {
    type Slice<'b> = StaticSystemSliceMut<'b, COUNT> where Self: 'b;
    type SliceMut<'b> = StaticSystemSliceMut<'b, COUNT> where Self: 'b;

    unsafe fn slice_unsafe(&self, range: Range<usize>) -> Self::Slice<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data.clone(),
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
        }
    }

    unsafe fn slice_unsafe_mut(&self, range: Range<usize>) -> Self::SliceMut<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data.clone(),
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
        }
    }
}
