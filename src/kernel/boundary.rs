//! Provides generalized interfaces for expressing various kinds of boundary condition.
//!
//! This module uses a combination of trait trickery and type transformers to
//! create an ergonomic API for working with boundaries.

use crate::geometry::{faces, Face, FaceArray};

/// Indicates what type of boundary condition is used along a particualr
/// face of the domain. More specific boundary conditions are provided
/// by the `Condition` API, but for many funtions, `Boundary` provides
/// enough information to compute supports and apply stencils.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BoundaryKind {
    /// Symmetric Boundary condition. Function is even along axis
    /// (so function values are reflected across the axis with the same sign)
    Symmetric,
    /// Antisymmetric Boundary condition. Function is even along axis
    /// (so function values are reflected across the axis with the opposite sign,
    /// on axis set to zero)
    AntiSymmetric,
    /// This boundary condition indicates that the ghost nodes have been filled manually via some external
    /// process. This can be used to implement custom boundary conditions, and is primarily used to
    /// fill inter-grid boundaries in the adaptive mesh refinement driver.
    Custom,
    /// The boundary condition is implemented via
    Radiative,
    #[default]
    Free,
    /// Boundary is set to a given value, lopsided stencils are used near the boundary. This
    /// condition can either be strongly enforced (values of systems are set at boundary before
    /// application) or weakly enforced (condition is applied to time derivative of system).
    StrongDirichlet,
    WeakDirichlet,
}

impl BoundaryKind {
    /// Are ghost nodes are used when enforcing this kind of boundary condition?
    pub fn needs_ghost(self) -> bool {
        match self {
            BoundaryKind::AntiSymmetric | BoundaryKind::Symmetric | BoundaryKind::Custom => true,
            BoundaryKind::Radiative
            | BoundaryKind::Free
            | BoundaryKind::StrongDirichlet
            | BoundaryKind::WeakDirichlet => false,
        }
    }

    pub fn is_parity(self) -> bool {
        matches!(self, BoundaryKind::AntiSymmetric | BoundaryKind::Symmetric)
    }

    pub fn class(self) -> BoundaryClass {
        match self {
            BoundaryKind::AntiSymmetric | BoundaryKind::Symmetric | BoundaryKind::Custom => {
                BoundaryClass::Ghost
            }
            BoundaryKind::Radiative
            | BoundaryKind::StrongDirichlet
            | BoundaryKind::WeakDirichlet
            | BoundaryKind::Free => BoundaryClass::OneSided,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BoundaryClass {
    #[default]
    OneSided,
    Ghost,
    Periodic,
}

impl BoundaryClass {
    pub fn has_ghost(self) -> bool {
        matches!(self, BoundaryClass::Ghost | BoundaryClass::Periodic)
    }
}

/// Describes a radiative boundary condition at a point on the boundary.
#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
pub struct DirichletParams {
    pub target: f64,
    pub strength: f64,
}

/// Provides specifics for enforcing boundary conditions for
/// a particular field.
pub trait BoundaryConds<const N: usize>: Clone {
    fn kind(&self, _face: Face<N>) -> BoundaryKind;

    fn radiative(&self, _position: [f64; N]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }

    fn dirichlet(&self, _position: [f64; N]) -> DirichletParams {
        DirichletParams {
            target: 0.0,
            strength: 1.0,
        }
    }
}

/// Checks whether a set of boundary conditions are compatible with the given ghost flags.
pub fn is_boundary_compatible<const N: usize, B: BoundaryConds<N>>(
    boundary: &FaceArray<N, BoundaryClass>,
    conditions: &B,
) -> bool {
    faces::<N>()
        .map(|face| conditions.kind(face).class() == boundary[face])
        .all(|x| x)
}
