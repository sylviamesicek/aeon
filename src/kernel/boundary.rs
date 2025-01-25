//! Provides generalized interfaces for expressing various kinds of boundary condition.
//!
//! This module uses a combination of trait trickery and type transformers to
//! create an ergonomic API for working with boundaries.

use aeon_geometry::Face;

/// Indicates what type of boundary condition is used along a particualr
/// face of the domain. More specific boundary conditions are provided
/// by the `Condition` API, but for many funtions, `Boundary` provides
/// enough information to compute supports and apply stencils.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
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
    #[default]
    Free,
}

impl BoundaryKind {
    /// Are ghost nodes are used when enforcing this kind of boundary condition?
    pub fn has_ghost(self) -> bool {
        matches!(self, Self::Custom | Self::Parity)
    }
}

/// Describes a radiative boundary condition at a point on the boundary.
pub struct RadiativeParams {
    /// Target value for field.
    pub target: f64,
    /// Wavespeed of field at boundary.
    pub speed: f64,
}

impl RadiativeParams {
    /// Constructs a boundary condition for a wave asymptotically approaching a given value, travelling
    /// at the speed of light (c = 1).
    pub fn lightlike(target: f64) -> Self {
        Self { target, speed: 1.0 }
    }
}

/// Provides specifics for enforcing boundary conditions for
/// a particular field.
pub trait Condition<const N: usize>: Clone {
    fn parity(&self, _face: Face<N>) -> bool {
        false
    }

    fn radiative(&self, _position: [f64; N]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
}
