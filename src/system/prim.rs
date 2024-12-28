use super::*;

// ****************************
// Builtin systems ************
// ****************************

use std::convert::Infallible;

/// A builtin label for systems with no fields (useful for code generation).
#[derive(Clone, Default)]
pub struct Empty;

impl System for Empty {
    const NAME: &'static str = "Empty";

    type Label = Infallible;

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        [].into_iter()
    }

    fn count(&self) -> usize {
        0
    }

    fn label_from_index(&self, _: usize) -> Self::Label {
        unreachable!()
    }

    fn label_index(&self, _: Self::Label) -> usize {
        unreachable!()
    }
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

impl<'a> SystemSliceMut<'a, Empty> {
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            total: 0,
            offset: 0,
            length: 0,
            system: &Empty,
        }
    }
}

/// A builtin label for simple scalar systems.
#[derive(Clone, Default)]
pub struct Scalar;

impl System for Scalar {
    const NAME: &'static str = "Scalar";

    type Label = ();

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        std::iter::once(())
    }

    fn count(&self) -> usize {
        1
    }

    fn label_index(&self, _: Self::Label) -> usize {
        0
    }

    fn label_from_index(&self, _: usize) -> Self::Label {
        ()
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

/// A label for a tuple of systems.
#[derive(Clone, Copy)]
pub enum Pair<A, B> {
    First(A),
    Second(B),
}

impl<A: System, B: System> System for (A, B) {
    type Label = Pair<A::Label, B::Label>;

    fn count(&self) -> usize {
        self.0.count() + self.1.count()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        self.0
            .enumerate()
            .map(Pair::First)
            .chain(self.1.enumerate().map(Pair::Second))
    }

    fn label_index(&self, label: Self::Label) -> usize {
        match label {
            Pair::First(a) => self.0.label_index(a),
            Pair::Second(b) => self.1.label_index(b),
        }
    }

    fn label_from_index(&self, index: usize) -> Self::Label {
        if index < self.0.count() {
            Pair::First(self.0.label_from_index(index))
        } else {
            Pair::Second(self.1.label_from_index(index - self.0.count()))
        }
    }
}

impl<'a, A: System, B: System> SystemSlice<'a, (A, B)> {
    pub fn split_pair(self) -> (SystemSlice<'a, A>, SystemSlice<'a, B>) {
        let stride = self.total / self.system.count();
        let total1 = stride * self.system.0.count();
        let total2 = stride * self.system.1.count();
        let ptr1 = self.ptr;
        let ptr2 = unsafe { self.ptr.add(total1) };

        (
            SystemSlice {
                total: total1,
                ptr: ptr1,
                offset: self.offset,
                length: self.length,
                system: &self.system.0,
            },
            SystemSlice {
                total: total2,

                ptr: ptr2,
                offset: self.offset,
                length: self.length,
                system: &self.system.1,
            },
        )
    }
}

impl<'a, A: System, B: System> SystemSliceMut<'a, (A, B)> {
    pub fn split_pair(self) -> (SystemSliceMut<'a, A>, SystemSliceMut<'a, B>) {
        let stride = self.total / self.system.count();
        let total1 = stride * self.system.0.count();
        let total2 = stride * self.system.1.count();
        let ptr1 = self.ptr;
        let ptr2 = unsafe { self.ptr.add(total1) };

        (
            SystemSliceMut {
                total: total1,
                ptr: ptr1,
                offset: self.offset,
                length: self.length,
                system: &self.system.0,
            },
            SystemSliceMut {
                total: total2,
                ptr: ptr2,
                offset: self.offset,
                length: self.length,
                system: &self.system.1,
            },
        )
    }
}

/// A system with a dynamical number of components choosen at runtime.
#[derive(Clone, Copy)]
pub struct Static<const N: usize>();

impl<const N: usize> System for Static<N> {
    type Label = usize;

    fn count(&self) -> usize {
        N
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..N).into_iter()
    }

    fn label_index(&self, label: Self::Label) -> usize {
        label
    }

    fn label_from_index(&self, index: usize) -> Self::Label {
        index
    }
}

/// A system with a dynamical number of components choosen at runtime.
#[derive(Clone, Copy)]
pub struct Dynamic(pub usize);

impl System for Dynamic {
    type Label = usize;

    fn count(&self) -> usize {
        self.0
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..self.0).into_iter()
    }

    fn label_index(&self, label: Self::Label) -> usize {
        label
    }

    fn label_from_index(&self, index: usize) -> Self::Label {
        index
    }
}
