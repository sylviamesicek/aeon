use crate::kernel::{Condition, RadiativeParams};
use crate::system::{Empty, Pair, Scalar, System};
use aeon_geometry::Face;

/// A generalization of `Condition<N>` for a coupled systems of scalar fields.
pub trait SystemConditions<const N: usize>: Clone {
    type System: System;

    fn parity(&self, _label: <Self::System as System>::Label, _face: Face<N>) -> bool {
        false
    }

    fn radiative(
        &self,
        _label: <Self::System as System>::Label,
        _position: [f64; N],
    ) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
}

/// Transfers a set of `Conditions<N>` into a single `Condition<N>` by only applying the set of conditions
/// to a single field.
pub struct SystemCondition<const N: usize, C: SystemConditions<N>> {
    conditions: C,
    field: <C::System as System>::Label,
}

impl<const N: usize, C: SystemConditions<N>> SystemCondition<N, C> {
    pub const fn new(conditions: C, field: <C::System as System>::Label) -> Self {
        Self { field, conditions }
    }
}

impl<const N: usize, C: SystemConditions<N>> Clone for SystemCondition<N, C> {
    fn clone(&self) -> Self {
        Self {
            conditions: self.conditions.clone(),
            field: self.field,
        }
    }
}

impl<const N: usize, C: SystemConditions<N>> Condition<N> for SystemCondition<N, C> {
    fn parity(&self, face: Face<N>) -> bool {
        self.conditions.parity(self.field.clone(), face)
    }

    fn radiative(&self, position: [f64; N]) -> RadiativeParams {
        self.conditions.radiative(self.field.clone(), position)
    }
}

// ****************************
// Specializations ************
// ****************************

/// Transforms a single condition into a set of `Conditions<N>` where `Self::System = Scalar`.
#[derive(Clone)]
pub struct ScalarConditions<I>(pub I);

impl<I> ScalarConditions<I> {
    pub const fn new(inner: I) -> Self {
        Self(inner)
    }
}

impl<const N: usize, I: Condition<N>> SystemConditions<N> for ScalarConditions<I> {
    type System = Scalar;

    fn parity(&self, _field: (), face: Face<N>) -> bool {
        self.0.parity(face)
    }

    fn radiative(&self, _field: (), position: [f64; N]) -> RadiativeParams {
        self.0.radiative(position)
    }
}

#[derive(Clone)]
pub struct PairConditions<L, R> {
    pub left: L,
    pub right: R,
}

impl<const N: usize, L: SystemConditions<N>, R: SystemConditions<N>> SystemConditions<N>
    for PairConditions<L, R>
{
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
    ) -> RadiativeParams {
        match field {
            Pair::First(left) => self.left.radiative(left, position),
            Pair::Second(right) => self.right.radiative(right, position),
        }
    }
}

#[derive(Clone)]
pub struct EmptyConditions;

impl<const N: usize> SystemConditions<N> for EmptyConditions {
    type System = Empty;

    fn parity(&self, _field: <Self::System as System>::Label, _face: Face<N>) -> bool {
        unreachable!()
    }

    fn radiative(
        &self,
        _field: <Self::System as System>::Label,
        _position: [f64; N],
    ) -> RadiativeParams {
        unreachable!()
    }
}
