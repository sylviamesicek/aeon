use std::ops::Range;

use super::{System, SystemSlice, SystemSliceMut, SystemSliceShared};

pub enum Pair<T, U> {
    First(T),
    Second(U),
}

impl<T: System, U: System> System for (T, U) {
    type Label = Pair<T::Label, U::Label>;

    fn len(&self) -> usize {
        debug_assert!(self.0.len() == self.1.len());
        self.0.len()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        let first = self.0.enumerate().map(Pair::First);
        let second = self.1.enumerate().map(Pair::Second);

        first.chain(second)
    }
}

impl<'a, T: SystemSlice<'a>, U: SystemSlice<'a>> SystemSlice<'a> for (T, U) {
    type Slice<'b> = (T::Slice<'b>, U::Slice<'b>) where Self: 'b;

    fn slice(&self, range: Range<usize>) -> Self::Slice<'_> {
        (self.0.slice(range.clone()), self.1.slice(range.clone()))
    }

    fn field(&self, label: Self::Label) -> &[f64] {
        match label {
            Pair::First(v) => self.0.field(v),
            Pair::Second(v) => self.1.field(v),
        }
    }
}

impl<'a, T: SystemSliceMut<'a>, U: SystemSliceMut<'a>> SystemSliceMut<'a> for (T, U) {
    type SliceMut<'b> = (T::SliceMut<'b>, U::SliceMut<'b>) where Self: 'b;
    type Shared = (T::Shared, U::Shared);

    fn slice_mut(&mut self, range: Range<usize>) -> Self::SliceMut<'_> {
        (
            self.0.slice_mut(range.clone()),
            self.1.slice_mut(range.clone()),
        )
    }

    fn field_mut(&mut self, label: Self::Label) -> &mut [f64] {
        match label {
            Pair::First(v) => self.0.field_mut(v),
            Pair::Second(v) => self.1.field_mut(v),
        }
    }

    fn into_shared(self) -> Self::Shared {
        (self.0.into_shared(), self.1.into_shared())
    }
}

unsafe impl<'a, T: SystemSliceShared<'a>, U: SystemSliceShared<'a>> SystemSliceShared<'a>
    for (T, U)
{
    type Slice<'b> = (T::Slice<'b>, U::Slice<'b>) where Self: 'b;
    type SliceMut<'b> = (T::SliceMut<'b>, U::SliceMut<'b>) where Self: 'b;

    unsafe fn slice_unsafe(&self, range: Range<usize>) -> Self::Slice<'_> {
        (
            self.0.slice_unsafe(range.clone()),
            self.1.slice_unsafe(range.clone()),
        )
    }

    unsafe fn slice_unsafe_mut(&self, range: Range<usize>) -> Self::SliceMut<'_> {
        (
            self.0.slice_unsafe_mut(range.clone()),
            self.1.slice_unsafe_mut(range.clone()),
        )
    }
}
