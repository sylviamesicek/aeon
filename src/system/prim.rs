use crate::system::{
    SystemArray, SystemFields, SystemFieldsMut, SystemLabel, SystemSlice, SystemSliceMut, SystemVec,
};
use aeon_array::ArrayLike;
use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

// ****************************
// Builtin Labels *************
// ****************************

#[derive(Clone)]
pub enum Empty {}

impl SystemLabel for Empty {
    const SYSTEM_NAME: &'static str = "Empty";

    fn name(&self) -> String {
        unreachable!()
    }

    fn index(&self) -> usize {
        unreachable!()
    }

    type Array<T> = SystemArray<T, 0>;

    fn fields() -> impl Iterator<Item = Self> {
        None::<Self>.into_iter()
    }

    fn field_from_index(_index: usize) -> Self {
        unreachable!()
    }
}

impl SystemVec<Empty> {
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<'a> SystemSlice<'a, Empty> {
    pub fn empty() -> Self {
        Self {
            data: &[],
            offset: 0,
            length: 0,
            _marker: PhantomData,
        }
    }
}

impl<'a> SystemSliceMut<'a, Empty> {
    pub fn empty() -> Self {
        Self {
            data: &mut [],
            offset: 0,
            length: 0,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct Scalar;

impl SystemLabel for Scalar {
    const SYSTEM_NAME: &'static str = "Scalar";

    fn name(&self) -> String {
        "scalar".to_string()
    }

    fn index(&self) -> usize {
        0
    }

    type Array<T> = SystemArray<T, 1>;

    fn fields() -> impl Iterator<Item = Self> {
        [Scalar].into_iter()
    }

    fn field_from_index(_index: usize) -> Self {
        Self
    }
}

impl<'a> From<&'a [f64]> for SystemSlice<'a, Scalar> {
    fn from(value: &'a [f64]) -> Self {
        SystemSlice::from_contiguous(value)
    }
}

impl<'a> From<&'a mut [f64]> for SystemSlice<'a, Scalar> {
    fn from(value: &'a mut [f64]) -> Self {
        SystemSlice::from_contiguous(value)
    }
}

impl<'a> From<&'a mut [f64]> for SystemSliceMut<'a, Scalar> {
    fn from(value: &'a mut [f64]) -> Self {
        SystemSliceMut::from_contiguous(value)
    }
}

#[derive(Clone)]
pub enum Pair<L, R> {
    Left(L),
    Right(R),
}

pub struct PairArray<T, L: SystemLabel, R: SystemLabel> {
    left: L::Array<T>,
    right: R::Array<T>,
}

impl<T, L: SystemLabel, R: SystemLabel> Index<Pair<L, R>> for PairArray<T, L, R> {
    type Output = T;

    fn index(&self, index: Pair<L, R>) -> &Self::Output {
        match index {
            Pair::Left(index) => &self.left[index],
            Pair::Right(index) => &self.right[index],
        }
    }
}

impl<T, L: SystemLabel, R: SystemLabel> IndexMut<Pair<L, R>> for PairArray<T, L, R> {
    fn index_mut(&mut self, index: Pair<L, R>) -> &mut Self::Output {
        match index {
            Pair::Left(index) => &mut self.left[index],
            Pair::Right(index) => &mut self.right[index],
        }
    }
}

impl<T, L: SystemLabel, R: SystemLabel> ArrayLike<Pair<L, R>> for PairArray<T, L, R> {
    type Elem = T;
    const LEN: usize = L::Array::<()>::LEN + R::Array::<()>::LEN;

    fn from_fn<F: FnMut(Pair<L, R>) -> Self::Elem>(mut f: F) -> Self {
        Self {
            left: L::Array::from_fn(|left| f(Pair::Left(left))),
            right: R::Array::from_fn(|right| f(Pair::Right(right))),
        }
    }
}

impl<L: SystemLabel, R: SystemLabel> SystemLabel for Pair<L, R> {
    const SYSTEM_NAME: &'static str = "Pair";

    fn name(&self) -> String {
        "Element of Pair".to_string()
    }

    fn index(&self) -> usize {
        match self {
            Pair::Left(left) => left.index(),
            Pair::Right(right) => right.index() + L::Array::<()>::LEN,
        }
    }

    type Array<T> = PairArray<T, L, R>;

    fn fields() -> impl Iterator<Item = Self> {
        L::fields()
            .map(Pair::Left)
            .chain(R::fields().map(Pair::Right))
    }

    fn field_from_index(index: usize) -> Self {
        if index < L::Array::<()>::LEN {
            Pair::Left(L::field_from_index(index))
        } else {
            Pair::Right(R::field_from_index(index - L::Array::<()>::LEN))
        }
    }
}

impl<'a, L: SystemLabel, R: SystemLabel> SystemFields<'a, Pair<L, R>> {
    pub fn join_pair(left: SystemFields<'a, L>, right: SystemFields<'a, R>) -> Self {
        assert_eq!(left.length, right.length);

        Self {
            length: left.length,
            fields: <Pair<L, R> as SystemLabel>::Array::from_fn(|pair| match pair {
                Pair::Left(l) => left.fields[l],
                Pair::Right(r) => right.fields[r],
            }),
            _marker: PhantomData,
        }
    }

    pub fn split_pair(self) -> (SystemFields<'a, L>, SystemFields<'a, R>) {
        let left = SystemFields {
            length: self.length,
            fields: L::Array::from_fn(|field| self.fields[Pair::Left(field)]),
            _marker: PhantomData,
        };

        let right = SystemFields {
            length: self.length,
            fields: R::Array::from_fn(|field| self.fields[Pair::Right(field)]),
            _marker: PhantomData,
        };

        (left, right)
    }
}

impl<'a, L: SystemLabel, R: SystemLabel> SystemFieldsMut<'a, Pair<L, R>> {
    pub fn join_pair(left: SystemFieldsMut<'a, L>, right: SystemFieldsMut<'a, R>) -> Self {
        assert_eq!(left.length, right.length);

        Self {
            length: left.length,
            fields: <Pair<L, R> as SystemLabel>::Array::from_fn(|pair| match pair {
                Pair::Left(l) => left.fields[l],
                Pair::Right(r) => right.fields[r],
            }),
            _marker: PhantomData,
        }
    }

    pub fn split_pair(self) -> (SystemFieldsMut<'a, L>, SystemFieldsMut<'a, R>) {
        let left = SystemFieldsMut {
            length: self.length,
            fields: L::Array::from_fn(|field| self.fields[Pair::Left(field)]),
            _marker: PhantomData,
        };

        let right = SystemFieldsMut {
            length: self.length,
            fields: R::Array::from_fn(|field| self.fields[Pair::Right(field)]),
            _marker: PhantomData,
        };

        (left, right)
    }
}
