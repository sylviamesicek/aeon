//! Provides generalized interfaces for expressing various kinds of boundary condition.
//!
//! This module uses a combination of trait trickery and type transformers to
//! create an ergonomic API for working with boundaries.

use aeon_geometry::Face;

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
        matches!(self, Self::Custom | Self::Parity)
    }
}

/// Provides information about what kind of boundary is used on each face.
pub trait Boundary<const N: usize>: Clone {
    fn kind(&self, face: Face<N>) -> BoundaryKind;
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

/// Combines a `Boundary<N>` with a `Condition<N>`.
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
