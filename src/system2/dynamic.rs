use std::ops::Range;

use crate::shared::SharedSlice;

use super::{SystemMut, SystemRef, SystemSlice, SystemSliceMut};

#[derive(Clone, Debug)]
pub struct DynamicSystem {
    data: Vec<f64>,
    count: usize,
}

impl DynamicSystem {
    pub fn new(count: usize) -> Self {
        Self {
            count,
            data: Vec::new(),
        }
    }

    pub fn with_length(length: usize, count: usize) -> Self {
        Self {
            data: vec![0.0; count * length],
            count,
        }
    }

    pub fn as_slice(&self) -> DynamicSystemSlice<'_> {
        DynamicSystemSlice::from_continguous(&self.data, self.count)
    }

    pub fn as_slice_mut(&mut self) -> DynamicSystemSliceMut<'_> {
        DynamicSystemSliceMut::from_continguous(&mut self.data, self.count)
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

impl<'a> SystemSlice<'a> for DynamicSystemSlice<'a> {
    type Label = usize;

    type SubSlice<'b> = DynamicSystemSlice<'b> where Self: 'b;
    type SystemRef<'b> = DynamicSystemSlice<'b> where Self: 'b;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn subslice(&self, range: Range<usize>) -> Self::SubSlice<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data,
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            count: self.count,
        }
    }

    fn as_system_ref(&self) -> Self::SystemRef<'_> {
        DynamicSystemSlice {
            data: &self.data,
            range: self.range.clone(),
            stride: self.stride,
            count: self.count,
        }
    }
}

impl<'a> SystemRef<'a> for DynamicSystemSlice<'a> {
    type Label = usize;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn field(&self, label: usize) -> &[f64] {
        &self.data[(label * self.stride)..][self.range.clone()]
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

impl<'a> SystemSlice<'a> for DynamicSystemSliceMut<'a> {
    type Label = usize;

    type SubSlice<'b> = DynamicSystemSliceMut<'b> where Self: 'b;
    type SystemRef<'b> = DynamicSystemSliceMut<'b> where Self: 'b;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn subslice(&self, range: Range<usize>) -> Self::SubSlice<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data,
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            count: self.count,
        }
    }

    fn as_system_ref(&self) -> Self::SystemRef<'_> {
        DynamicSystemSliceMut {
            data: self.data,
            range: self.range.clone(),
            stride: self.stride,
            count: self.count,
        }
    }
}

impl<'a> SystemSliceMut<'a> for DynamicSystemSliceMut<'a> {
    type SubSliceMut<'b> = DynamicSystemSliceMut<'b> where Self: 'b;
    type SystemMut<'b> = DynamicSystemSliceMut<'b> where Self: 'b;

    fn subslice_mut(&mut self, range: Range<usize>) -> Self::SubSliceMut<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data,
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            count: self.count,
        }
    }

    unsafe fn subslice_unsafe(&self, range: Range<usize>) -> Self::SubSliceMut<'_> {
        Self {
            data: self.data,
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            count: self.count,
        }
    }

    fn as_system_mut(&mut self) -> Self::SystemRef<'_> {
        DynamicSystemSliceMut {
            data: self.data,
            range: self.range.clone(),
            stride: self.stride,
            count: self.count,
        }
    }
}

impl<'a> SystemRef<'a> for DynamicSystemSliceMut<'a> {
    type Label = usize;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn field(&self, label: usize) -> &[f64] {
        unsafe {
            core::slice::from_raw_parts(
                self.data.get(label * self.stride + self.range.start),
                self.range.len(),
            )
        }
    }
}

impl<'a> SystemMut<'a> for DynamicSystemSliceMut<'a> {
    fn field_mut(&mut self, label: usize) -> &mut [f64] {
        unsafe {
            core::slice::from_raw_parts_mut(
                self.data.get_mut(label * self.stride + self.range.start),
                self.range.len(),
            )
        }
    }
}
