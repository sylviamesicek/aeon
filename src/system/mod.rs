mod vec;

pub use vec::*;

pub trait System {
    const NAME: &'static str = "Unknown";

    type Label: Clone + Copy;

    fn enumerate(&self) -> impl Iterator<Item = Self::Label>;
    fn count(&self) -> usize {
        self.enumerate().count()
    }

    fn label_index(&self, label: Self::Label) -> usize;
    fn label_from_index(&self, index: usize) -> Self::Label;
    fn label_name(&self, _label: Self::Label) -> String {
        "Unknown".to_string()
    }
}

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

/// A builtin label for simple scalar systems.
#[derive(Clone, Default)]
pub struct Scalar;

impl System for Scalar {
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
