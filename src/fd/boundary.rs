use crate::{
    geometry::{Face, FaceMask},
    system::{Empty, Pair, Scalar, SystemLabel},
};

use aeon_basis::{Boundary, BoundaryKind, Condition};

/// A transformer for overriding individual boundary faces with `BoundaryKind::Custom`. This
/// is primarily used internally to ensure interior boundaries between blocks are properly
/// used.
#[derive(Debug, Clone)]
pub struct BlockBoundary<const N: usize, I> {
    pub inner: I,
    pub flags: FaceMask<N>,
}

impl<const N: usize, I> BlockBoundary<N, I> {
    pub fn new(flags: FaceMask<N>, inner: I) -> Self {
        Self { inner, flags }
    }
}

impl<const N: usize, I: Boundary<N>> Boundary<N> for BlockBoundary<N, I> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        if self.flags.is_set(face) {
            self.inner.kind(face)
        } else {
            BoundaryKind::Custom
        }
    }
}

/// A generalization of `Condition<N>` for a coupled systems of scalar fields.
pub trait Conditions<const N: usize>: Clone {
    type System: SystemLabel;

    fn parity(&self, _field: Self::System, _face: Face<N>) -> bool {
        false
    }

    fn radiative(&self, _field: Self::System, _position: [f64; N]) -> f64 {
        0.0
    }
}

/// Transforms a single condition into a set of `Conditions<N>` where `Self::System = Scalar`.
#[derive(Clone)]
pub struct ScalarConditions<I>(pub I);

impl<I> ScalarConditions<I> {
    pub const fn new(inner: I) -> Self {
        Self(inner)
    }
}

impl<const N: usize, I: Condition<N>> Conditions<N> for ScalarConditions<I> {
    type System = Scalar;

    fn parity(&self, _field: Self::System, face: Face<N>) -> bool {
        self.0.parity(face)
    }

    fn radiative(&self, _field: Self::System, position: [f64; N]) -> f64 {
        self.0.radiative(position)
    }
}

#[derive(Clone)]
pub struct SystemCondition<S, C> {
    field: S,
    conditions: C,
}

impl<S, C> SystemCondition<S, C> {
    pub const fn new(field: S, conditions: C) -> Self {
        Self { field, conditions }
    }
}

impl<const N: usize, S: SystemLabel, C: Conditions<N, System = S>> Condition<N>
    for SystemCondition<S, C>
{
    fn parity(&self, face: Face<N>) -> bool {
        self.conditions.parity(self.field.clone(), face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.conditions.radiative(self.field.clone(), position)
    }
}

/// Combines a boundary with a set of conditions, for a specific field.
#[derive(Clone)]
pub struct SystemBC<S: SystemLabel, B, C> {
    field: S,
    boundary: B,
    conditions: C,
}

impl<S: SystemLabel, B, C> SystemBC<S, B, C> {
    pub const fn new(field: S, boundary: B, conditions: C) -> Self {
        Self {
            field,
            boundary,
            conditions,
        }
    }
}

impl<const N: usize, S: SystemLabel, B: Clone, C: Conditions<N, System = S>> Condition<N>
    for SystemBC<S, B, C>
{
    fn parity(&self, face: Face<N>) -> bool {
        self.conditions.parity(self.field.clone(), face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.conditions.radiative(self.field.clone(), position)
    }
}

impl<const N: usize, S: SystemLabel, B: Boundary<N>, C: Clone> Boundary<N> for SystemBC<S, B, C> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        self.boundary.kind(face)
    }
}

#[derive(Clone)]
pub struct PairConditions<L, R> {
    pub left: L,
    pub right: R,
}

impl<const N: usize, L: Conditions<N>, R: Conditions<N>> Conditions<N> for PairConditions<L, R> {
    type System = Pair<L::System, R::System>;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        match field {
            Pair::Left(left) => self.left.parity(left, face),
            Pair::Right(right) => self.right.parity(right, face),
        }
    }

    fn radiative(&self, field: Self::System, position: [f64; N]) -> f64 {
        match field {
            Pair::Left(left) => self.left.radiative(left, position),
            Pair::Right(right) => self.right.radiative(right, position),
        }
    }
}

#[derive(Clone)]
pub struct EmptyConditions;

impl<const N: usize> Conditions<N> for EmptyConditions {
    type System = Empty;

    fn parity(&self, _field: Self::System, _face: Face<N>) -> bool {
        unreachable!()
    }

    fn radiative(&self, _field: Self::System, _position: [f64; N]) -> f64 {
        unreachable!()
    }
}
