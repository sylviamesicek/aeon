use std::{array, marker::PhantomData, ops::Range};

use reborrow::{Reborrow, ReborrowMut};

use crate::shared::SharedSlice;

use super::{SystemMut, SystemRef, SystemSlice, SystemSliceMut};

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

impl<'a, const COUNT: usize> SystemSlice<'a> for StaticSystemSlice<'a, COUNT> {
    type Label = usize;

    type SubSlice<'b> = StaticSystemSlice<'b, COUNT> where Self: 'b;
    type SystemRef<'b> = StaticSystemRef<'b, COUNT> where Self: 'b;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn subslice(&self, range: Range<usize>) -> Self::SubSlice<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data,
            range: (self.range.start + range.start)..(self.range.start + range.end),
            stride: self.stride,
        }
    }

    fn as_system_ref(&self) -> Self::SystemRef<'_> {
        StaticSystemRef {
            length: self.range.len(),
            array: array::from_fn(|i| &self.data[i * self.stride] as *const f64),
            _marker: PhantomData,
        }
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

impl<'a, const COUNT: usize> SystemSlice<'a> for StaticSystemSliceMut<'a, COUNT> {
    type Label = usize;

    type SubSlice<'b> = StaticSystemSliceMut<'b, COUNT> where Self: 'b;
    type SystemRef<'b> = StaticSystemRef<'b, COUNT> where Self: 'b;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn subslice(&self, range: Range<usize>) -> Self::SubSlice<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data.clone(),
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
        }
    }

    fn as_system_ref(&self) -> Self::SystemRef<'_> {
        let stride = self.data.len() / COUNT;
        StaticSystemRef {
            length: self.range.len(),
            array: array::from_fn(|i| unsafe { self.data.get(i * stride) } as *const f64),
            _marker: PhantomData,
        }
    }
}

impl<'a, const COUNT: usize> SystemSliceMut<'a> for StaticSystemSliceMut<'a, COUNT> {
    type SubSliceMut<'b> = StaticSystemSliceMut<'b, COUNT> where Self: 'b;
    type SystemMut<'b> = StaticSystemMut<'b, COUNT> where Self: 'b;

    fn subslice_mut(&mut self, range: Range<usize>) -> Self::SubSliceMut<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data.clone(),
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
        }
    }

    unsafe fn subslice_unsafe(&self, range: Range<usize>) -> Self::SubSliceMut<'_> {
        debug_assert!(range.start <= self.range.len() && range.end <= self.range.len());

        Self {
            data: self.data.clone(),
            stride: self.stride,
            range: (self.range.start + range.start)..(self.range.start + range.end),
        }
    }

    fn as_system_mut(&mut self) -> Self::SystemMut<'_> {
        let stride = self.data.len() / COUNT;
        StaticSystemMut {
            length: self.range.len(),
            array: array::from_fn(|i| unsafe { self.data.get_mut(i * stride) } as *mut f64),
            _marker: PhantomData,
        }
    }
}

pub struct StaticSystemRef<'a, const COUNT: usize> {
    length: usize,
    array: [*const f64; COUNT],
    _marker: PhantomData<&'a [f64]>,
}

impl<'a, const COUNT: usize> SystemRef<'a> for StaticSystemRef<'a, COUNT> {
    type Label = usize;

    fn len(&self) -> usize {
        self.length
    }

    fn field(&self, label: Self::Label) -> &[f64] {
        unsafe { core::slice::from_raw_parts(self.array[label], self.length) }
    }
}

pub struct StaticSystemMut<'a, const COUNT: usize> {
    length: usize,
    array: [*mut f64; COUNT],
    _marker: PhantomData<&'a mut [f64]>,
}

impl<'a, const COUNT: usize> SystemRef<'a> for StaticSystemMut<'a, COUNT> {
    type Label = usize;

    fn len(&self) -> usize {
        self.length
    }

    fn field(&self, label: Self::Label) -> &[f64] {
        unsafe { core::slice::from_raw_parts(self.array[label], self.length) }
    }
}

impl<'a, const COUNT: usize> SystemMut<'a> for StaticSystemMut<'a, COUNT> {
    fn field_mut(&mut self, label: Self::Label) -> &mut [f64] {
        unsafe { core::slice::from_raw_parts_mut(self.array[label], self.length) }
    }
}
