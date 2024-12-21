use crate::{
    geometry::{Face, FaceMask},
    system::{Empty, Pair, Scalar, System},
};

use aeon_basis::{Boundary, BoundaryKind, Condition, RadiativeParams};

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
    type System: System;

    fn parity(&self, _label: <Self::System as System>::Label, _face: Face<N>) -> bool {
        false
    }

    fn radiative(
        &self,
        _label: <Self::System as System>::Label,
        _position: [f64; N],
        _spacing: f64,
    ) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
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

    fn parity(&self, _field: (), face: Face<N>) -> bool {
        self.0.parity(face)
    }

    fn radiative(&self, _field: (), position: [f64; N], spacing: f64) -> RadiativeParams {
        self.0.radiative(position, spacing)
    }
}

#[derive(Clone)]
pub struct SystemCondition<const N: usize, C: Conditions<N>> {
    conditions: C,
    field: <C::System as System>::Label,
}

impl<const N: usize, C: Conditions<N>> SystemCondition<N, C> {
    pub const fn new(conditions: C, field: <C::System as System>::Label) -> Self {
        Self { field, conditions }
    }
}

impl<const N: usize, C: Conditions<N>> Condition<N> for SystemCondition<N, C>
where
    C::System: Clone,
{
    fn parity(&self, face: Face<N>) -> bool {
        self.conditions.parity(self.field.clone(), face)
    }

    fn radiative(&self, position: [f64; N], spacing: f64) -> RadiativeParams {
        self.conditions
            .radiative(self.field.clone(), position, spacing)
    }
}

/// Combines a boundary with a set of conditions, for a specific field.
pub struct SystemBC<const N: usize, B, C: Conditions<N>> {
    boundary: B,
    conditions: C,
    field: <C::System as System>::Label,
}

impl<const N: usize, B: Boundary<N>, C: Conditions<N>> Clone for SystemBC<N, B, C> {
    fn clone(&self) -> Self {
        Self {
            boundary: self.boundary.clone(),
            conditions: self.conditions.clone(),
            field: self.field,
        }
    }
}

impl<const N: usize, B: Boundary<N>, C: Conditions<N>> SystemBC<N, B, C> {
    pub const fn new(boundary: B, conditions: C, field: <C::System as System>::Label) -> Self {
        Self {
            field,
            boundary,
            conditions,
        }
    }
}

impl<const N: usize, B: Boundary<N>, C: Conditions<N>> Condition<N> for SystemBC<N, B, C> {
    fn parity(&self, face: Face<N>) -> bool {
        self.conditions.parity(self.field.clone(), face)
    }

    fn radiative(&self, position: [f64; N], spacing: f64) -> RadiativeParams {
        self.conditions
            .radiative(self.field.clone(), position, spacing)
    }
}

impl<const N: usize, B: Boundary<N>, C: Conditions<N>> Boundary<N> for SystemBC<N, B, C> {
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
    type System = (L::System, R::System);

    fn parity(&self, field: <Self::System as System>::Label, face: Face<N>) -> bool {
        match field {
            Pair::First(left) => self.left.parity(left, face),
            Pair::Second(right) => self.right.parity(right, face),
        }
    }

    fn radiative(
        &self,
        field: <Self::System as System>::Label,
        position: [f64; N],
        spacing: f64,
    ) -> RadiativeParams {
        match field {
            Pair::First(left) => self.left.radiative(left, position, spacing),
            Pair::Second(right) => self.right.radiative(right, position, spacing),
        }
    }
}

#[derive(Clone)]
pub struct EmptyConditions;

impl<const N: usize> Conditions<N> for EmptyConditions {
    type System = Empty;

    fn parity(&self, _field: <Self::System as System>::Label, _face: Face<N>) -> bool {
        unreachable!()
    }

    fn radiative(
        &self,
        _field: <Self::System as System>::Label,
        _position: [f64; N],
        _spacing: f64,
    ) -> RadiativeParams {
        unreachable!()
    }
}
