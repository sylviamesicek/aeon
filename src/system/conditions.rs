use crate::geometry::Face;
use crate::kernel::{Boundary, BoundaryKind, DirichletParams, RadiativeParams};
use crate::system::{Empty, Pair, Scalar, System};

/// A generalization of `Condition<N>` for a coupled systems of scalar fields.
pub trait SystemBoundaryConds<const N: usize>: Clone {
    type System: System;

    fn kind(&self, _label: <Self::System as System>::Label, _face: Face<N>) -> BoundaryKind;

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

    fn dirichlet(
        &self,
        _label: <Self::System as System>::Label,
        _position: [f64; N],
    ) -> DirichletParams {
        DirichletParams {
            target: 0.0,
            strength: 1.0,
        }
    }

    fn field(&self, field: <Self::System as System>::Label) -> FieldBoundaryConds<N, Self> {
        FieldBoundaryConds::new(self.clone(), field)
    }
}

/// Transfers a set of `Conditions<N>` into a single `Condition<N>` by only applying the set of conditions
/// to a single field.
pub struct FieldBoundaryConds<const N: usize, C: SystemBoundaryConds<N>> {
    conditions: C,
    field: <C::System as System>::Label,
}

impl<const N: usize, C: SystemBoundaryConds<N>> FieldBoundaryConds<N, C> {
    pub const fn new(conditions: C, field: <C::System as System>::Label) -> Self {
        Self { field, conditions }
    }
}

impl<const N: usize, C: SystemBoundaryConds<N>> Clone for FieldBoundaryConds<N, C> {
    fn clone(&self) -> Self {
        Self {
            conditions: self.conditions.clone(),
            field: self.field,
        }
    }
}

impl<const N: usize, C: SystemBoundaryConds<N>> Boundary<N> for FieldBoundaryConds<N, C> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        self.conditions.kind(self.field, face)
    }

    fn radiative(&self, position: [f64; N]) -> RadiativeParams {
        self.conditions.radiative(self.field, position)
    }

    fn dirichlet(&self, position: [f64; N]) -> DirichletParams {
        self.conditions.dirichlet(self.field, position)
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

impl<const N: usize, I: Boundary<N>> SystemBoundaryConds<N> for ScalarConditions<I> {
    type System = Scalar;

    fn kind(&self, _label: <Self::System as System>::Label, face: Face<N>) -> BoundaryKind {
        self.0.kind(face)
    }

    fn radiative(&self, _field: (), position: [f64; N]) -> RadiativeParams {
        self.0.radiative(position)
    }

    fn dirichlet(
        &self,
        _label: <Self::System as System>::Label,
        position: [f64; N],
    ) -> DirichletParams {
        self.0.dirichlet(position)
    }
}

#[derive(Clone)]
pub struct PairConditions<L, R> {
    pub left: L,
    pub right: R,
}

impl<const N: usize, L: SystemBoundaryConds<N>, R: SystemBoundaryConds<N>> SystemBoundaryConds<N>
    for PairConditions<L, R>
{
    type System = (L::System, R::System);

    fn kind(&self, field: <Self::System as System>::Label, face: Face<N>) -> BoundaryKind {
        match field {
            Pair::First(left) => self.left.kind(left, face),
            Pair::Second(right) => self.right.kind(right, face),
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

    fn dirichlet(
        &self,
        field: <Self::System as System>::Label,
        position: [f64; N],
    ) -> DirichletParams {
        match field {
            Pair::First(left) => self.left.dirichlet(left, position),
            Pair::Second(right) => self.right.dirichlet(right, position),
        }
    }
}

#[derive(Clone)]
pub struct EmptyConditions;

impl<const N: usize> SystemBoundaryConds<N> for EmptyConditions {
    type System = Empty;

    fn kind(&self, _label: <Self::System as System>::Label, _face: Face<N>) -> BoundaryKind {
        unreachable!()
    }

    fn radiative(
        &self,
        _field: <Self::System as System>::Label,
        _position: [f64; N],
    ) -> RadiativeParams {
        unreachable!()
    }

    fn dirichlet(
        &self,
        _label: <Self::System as System>::Label,
        _position: [f64; N],
    ) -> DirichletParams {
        unreachable!()
    }
}
