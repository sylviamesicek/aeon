use std::{marker::PhantomData, ops::Range};

use reborrow::{Reborrow, ReborrowMut};

use super::{
    DynamicSystem, DynamicSystemSlice, DynamicSystemSliceMut, System, SystemSlice, SystemSliceMut,
    SystemSliceShared,
};

pub trait SystemLabel: Clone + Copy {
    /// Returns a human readable name for a given label.
    fn name(&self) -> String;
    /// Returns an index for a given label.
    fn index(&self) -> usize;
    /// Returns the total number of valid labels.
    fn count() -> usize;
    /// Converts an index into a system label.
    fn from_index(i: usize) -> Self;
}

pub struct LabelledSystem<L: SystemLabel> {
    inner: DynamicSystem,
    _marker: PhantomData<L>,
}

impl<L: SystemLabel> LabelledSystem<L> {
    pub fn new() -> Self {
        Self {
            inner: DynamicSystem::new(L::count()),
            _marker: PhantomData,
        }
    }

    pub fn with_length(length: usize) -> Self {
        Self {
            inner: DynamicSystem::with_length(length, L::count()),
            _marker: PhantomData,
        }
    }
}

impl<L: SystemLabel> System for LabelledSystem<L> {
    type Label = L;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..L::count()).map(L::from_index)
    }
}

pub struct LabelledSystemSlice<'a, L> {
    inner: DynamicSystemSlice<'a>,
    _marker: PhantomData<L>,
}

impl<'a, L: SystemLabel> LabelledSystemSlice<'a, L> {
    pub fn from_continguous(data: &'a [f64]) -> Self {
        Self {
            inner: DynamicSystemSlice::from_continguous(data, L::count()),
            _marker: PhantomData,
        }
    }
}

impl<'long, 'short, L> Reborrow<'short> for LabelledSystemSlice<'long, L> {
    type Target = LabelledSystemSlice<'short, L>;

    fn rb(&'short self) -> Self::Target {
        LabelledSystemSlice {
            inner: self.inner.rb(),
            _marker: PhantomData,
        }
    }
}

impl<'a, L: SystemLabel> System for LabelledSystemSlice<'a, L> {
    type Label = L;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..L::count()).map(L::from_index)
    }
}

impl<'a, L: SystemLabel> SystemSlice<'a> for LabelledSystemSlice<'a, L> {
    type Slice<'b> = LabelledSystemSlice<'b, L> where Self: 'b;

    fn slice(&self, range: Range<usize>) -> Self::Slice<'_> {
        LabelledSystemSlice {
            inner: self.inner.slice(range),
            _marker: PhantomData,
        }
    }

    fn field(&self, label: Self::Label) -> &[f64] {
        self.inner.field(label.index())
    }
}

pub struct LabelledSystemSliceMut<'a, L> {
    inner: DynamicSystemSliceMut<'a>,
    _marker: PhantomData<L>,
}

impl<'a, L: SystemLabel> LabelledSystemSliceMut<'a, L> {
    pub fn from_continguous(data: &'a mut [f64]) -> Self {
        Self {
            inner: DynamicSystemSliceMut::from_continguous(data, L::count()),
            _marker: PhantomData,
        }
    }
}

impl<'long, 'short, L: SystemLabel> Reborrow<'short> for LabelledSystemSliceMut<'long, L> {
    type Target = LabelledSystemSliceMut<'short, L>;

    fn rb(&'short self) -> Self::Target {
        LabelledSystemSliceMut {
            inner: self.inner.rb(),
            _marker: PhantomData,
        }
    }
}

impl<'long, 'short, L: SystemLabel> ReborrowMut<'short> for LabelledSystemSliceMut<'long, L> {
    type Target = LabelledSystemSliceMut<'short, L>;

    fn rb_mut(&'short mut self) -> Self::Target {
        LabelledSystemSliceMut {
            inner: self.inner.rb_mut(),
            _marker: PhantomData,
        }
    }
}

impl<'a, L: SystemLabel> System for LabelledSystemSliceMut<'a, L> {
    type Label = L;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..L::count()).map(L::from_index)
    }
}

impl<'a, L: SystemLabel> SystemSlice<'a> for LabelledSystemSliceMut<'a, L> {
    type Slice<'b> = LabelledSystemSliceMut<'b, L> where Self: 'b;

    fn slice(&self, range: Range<usize>) -> Self::Slice<'_> {
        LabelledSystemSliceMut {
            inner: self.inner.slice(range),
            _marker: PhantomData,
        }
    }

    fn field(&self, label: L) -> &[f64] {
        self.inner.field(label.index())
    }
}

impl<'a, L: SystemLabel> SystemSliceMut<'a> for LabelledSystemSliceMut<'a, L> {
    type SliceMut<'b> = LabelledSystemSliceMut<'b, L> where Self: 'b;
    type Shared = LabelledSystemSliceMut<'a, L>;

    fn slice_mut(&mut self, range: Range<usize>) -> Self::SliceMut<'_> {
        LabelledSystemSliceMut {
            inner: self.inner.slice_mut(range),
            _marker: PhantomData,
        }
    }

    fn field_mut(&mut self, label: Self::Label) -> &mut [f64] {
        self.inner.field_mut(label.index())
    }

    fn into_shared(self) -> Self::Shared {
        self
    }
}

unsafe impl<'a, L: SystemLabel> SystemSliceShared<'a> for LabelledSystemSliceMut<'a, L> {
    type Slice<'b> = LabelledSystemSliceMut<'b, L> where Self: 'b;
    type SliceMut<'b> = LabelledSystemSliceMut<'b, L> where Self: 'b;

    unsafe fn slice_unsafe(&self, range: Range<usize>) -> Self::Slice<'_> {
        LabelledSystemSliceMut {
            inner: self.inner.slice_unsafe(range),
            _marker: PhantomData,
        }
    }

    unsafe fn slice_unsafe_mut(&self, range: Range<usize>) -> Self::SliceMut<'_> {
        LabelledSystemSliceMut {
            inner: self.inner.slice_unsafe_mut(range),
            _marker: PhantomData,
        }
    }
}
