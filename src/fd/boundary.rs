//! Provides generalized interfaces for expressing various kinds of boundary condition.
//!
//! This module uses a combination of trait trickery and type transformers to
//! create an ergonomic API for working with boundaries.

use crate::{
    geometry::{Face, FaceMask},
    prelude::{Rectangle, SystemLabel},
    system::Scalar,
};

/// Indicates what type of boundary condition is used along a particualr
/// face of the domain. More specific boundary conditions are provided
/// by the `Condition` API, but for many funtions, `Boundary` provides
/// enough information to compute supports and apply stencils.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryKind {
    /// A (anti)symmetric boundary condition. True indicates an even function
    /// (so function values are reflected across the axis with the same sign)
    /// and false indicates an odd function.
    Parity,
    /// This boundary condition indicates that the ghost nodes have been filled manually via some external
    /// process. This can be used to implement custom boundary conditions, and is primarily used to
    /// fill inter-grid boundaries in the adaptive mesh refinement driver.
    Custom,
    Radiative,
    Free,
}

impl BoundaryKind {
    pub fn has_ghost(self) -> bool {
        matches!(self, Self::Custom)
    }
}

/// Provides information about what kind of boundary is used on each face.
pub trait Boundary<const N: usize>: Clone {
    fn kind(&self, face: Face<N>) -> BoundaryKind;
}

/// Provides information about the bounds of the space.
pub trait Domain<const N: usize>: Clone {
    fn bounds(&self) -> Rectangle<N>;
}

impl<const N: usize> Domain<N> for Rectangle<N> {
    fn bounds(&self) -> Rectangle<N> {
        self.clone()
    }
}

/// Provides specifics for enforcing boundary conditions for
/// a particular field.
pub trait Condition<const N: usize>: Clone {
    fn parity(&self, _face: Face<N>) -> bool {
        false
    }

    fn radiative(&self, _position: [f64; N]) -> f64 {
        0.0
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

/// Transforms a single condition into a set of `Conditions<N>` where `Self::System = ()`.
#[derive(Clone)]
pub struct UnitBC<I>(pub I);

impl<I> UnitBC<I> {
    pub fn new(inner: I) -> Self {
        Self(inner)
    }
}

impl<const N: usize, I: Boundary<N>> Boundary<N> for UnitBC<I> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        self.0.kind(face)
    }
}

impl<const N: usize, I: Condition<N>> Condition<N> for UnitBC<I> {
    fn parity(&self, face: Face<N>) -> bool {
        self.0.parity(face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.0.radiative(position)
    }
}

impl<const N: usize, I: Condition<N>> Conditions<N> for UnitBC<I> {
    type System = Scalar;

    fn parity(&self, _field: Self::System, face: Face<N>) -> bool {
        self.0.parity(face)
    }

    fn radiative(&self, _field: Self::System, position: [f64; N]) -> f64 {
        self.0.radiative(position)
    }
}

/// A transformer for extracting boundary information for a specific field in a system.
#[derive(Clone)]
pub struct SystemBC<S: SystemLabel, I> {
    field: S,
    inner: I,
}

impl<S: SystemLabel, C> SystemBC<S, C> {
    pub const fn new(field: S, inner: C) -> Self {
        Self { field, inner }
    }
}

impl<const N: usize, S: SystemLabel, C: Conditions<N, System = S>> Condition<N> for SystemBC<S, C> {
    fn parity(&self, face: Face<N>) -> bool {
        self.inner.parity(self.field.clone(), face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.inner.radiative(self.field.clone(), position)
    }
}

impl<const N: usize, S: SystemLabel, C: Conditions<N, System = S>> Conditions<N>
    for SystemBC<S, C>
{
    type System = Scalar;

    fn parity(&self, _field: Scalar, face: Face<N>) -> bool {
        self.inner.parity(self.field.clone(), face)
    }

    fn radiative(&self, _field: Scalar, position: [f64; N]) -> f64 {
        self.inner.radiative(self.field.clone(), position)
    }
}

impl<const N: usize, S: SystemLabel, B: Boundary<N>> Boundary<N> for SystemBC<S, B> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        self.inner.kind(face)
    }
}

impl<const N: usize, S: SystemLabel, B: Domain<N>> Domain<N> for SystemBC<S, B> {
    fn bounds(&self) -> Rectangle<N> {
        self.inner.bounds()
    }
}

/// A transformer for overriding individual boundary faces with `BoundaryKind::Custom`. This
/// is primarily used internally to ensure interior boundaries between blocks are properly
/// used.
#[derive(Debug, Clone)]
pub struct BlockBC<const N: usize, I> {
    pub inner: I,
    pub flags: FaceMask<N>,
}

impl<const N: usize, I> BlockBC<N, I> {
    pub fn new(flags: FaceMask<N>, inner: I) -> Self {
        Self { inner, flags }
    }
}

impl<const N: usize, I: Boundary<N>> Boundary<N> for BlockBC<N, I> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        if self.flags.is_set(face) {
            self.inner.kind(face)
        } else {
            BoundaryKind::Custom
        }
    }
}

impl<const N: usize, I: Domain<N>> Domain<N> for BlockBC<N, I> {
    fn bounds(&self) -> Rectangle<N> {
        self.inner.bounds()
    }
}

impl<const N: usize, I: Condition<N>> Condition<N> for BlockBC<N, I> {
    fn parity(&self, face: Face<N>) -> bool {
        self.inner.parity(face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.inner.radiative(position)
    }
}

impl<const N: usize, C: Conditions<N>> Conditions<N> for BlockBC<N, C> {
    type System = C::System;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        self.inner.parity(field, face)
    }

    fn radiative(&self, field: Self::System, position: [f64; N]) -> f64 {
        self.inner.radiative(field, position)
    }
}

/// Combines a `Boundary<N>` with a `Condition<N>` or `Conditions<N>`.
#[derive(Clone)]
pub struct BC<B, C> {
    pub boundary: B,
    pub condition: C,
}

impl<B, C> BC<B, C> {
    pub fn new(boundary: B, condition: C) -> Self {
        Self {
            boundary,
            condition,
        }
    }
}

impl<const N: usize, B: Boundary<N>, C: Clone> Boundary<N> for BC<B, C> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        self.boundary.kind(face)
    }
}

impl<const N: usize, B: Clone, C: Condition<N>> Condition<N> for BC<B, C> {
    fn parity(&self, face: Face<N>) -> bool {
        self.condition.parity(face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.condition.radiative(position)
    }
}

impl<const N: usize, B: Clone, C: Conditions<N>> Conditions<N> for BC<B, C> {
    type System = C::System;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        self.condition.parity(field, face)
    }

    fn radiative(&self, field: Self::System, position: [f64; N]) -> f64 {
        self.condition.radiative(field, position)
    }
}

/// Bundles a domain with a set of boundary conditions.
#[derive(Clone)]
pub struct DomainWithBC<D, I> {
    pub domain: D,
    pub inner: I,
}

impl<D, I> DomainWithBC<D, I> {
    pub fn new(domain: D, inner: I) -> Self {
        Self { domain, inner }
    }
}

impl<const N: usize, D: Domain<N>, I: Clone> Domain<N> for DomainWithBC<D, I> {
    fn bounds(&self) -> Rectangle<N> {
        self.domain.bounds()
    }
}

impl<const N: usize, I: Clone, BC: Boundary<N>> Boundary<N> for DomainWithBC<I, BC> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        self.inner.kind(face)
    }
}

impl<const N: usize, I: Clone, BC: Condition<N>> Condition<N> for DomainWithBC<I, BC> {
    fn parity(&self, face: Face<N>) -> bool {
        self.inner.parity(face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.inner.radiative(position)
    }
}

impl<const N: usize, B: Clone, C: Conditions<N>> Conditions<N> for DomainWithBC<B, C> {
    type System = C::System;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        self.inner.parity(field, face)
    }

    fn radiative(&self, field: Self::System, position: [f64; N]) -> f64 {
        self.inner.radiative(field, position)
    }
}
