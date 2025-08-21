//! Provides generalized interfaces for expressing various kinds of boundary condition.
//!
//! This module uses a combination of trait trickery and type transformers to
//! create an ergonomic API for working with boundaries.

use crate::geometry::{Face, FaceArray};

// #[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
// pub enum Boundary {
//     Symmetric,
//     AntiSymmetric,
//     Dirichlet(f64),
//     #[default]
//     Free,
//     Custom,
// }

// impl Boundary {
//     pub fn is_one_sided(self) -> bool {
//         matches!(self, Self::Free)
//     }
// }

// pub trait BoundaryConds<const N: usize> {
//     fn boundary(&self, channel: usize, position: [f64; N]) -> Boundary;
// }

// #[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
// pub enum Penalty {
//     #[default]
//     Free,
//     Radiative {
//         target: f64,
//         strength: f64,
//     },
// }

// pub trait PenaltyTerms<const N: usize> {
//     fn penalty(&self, channel: usize, position: [f64; N]) -> Penalty;
// }

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
    Face::iterate()
        .map(|face| conditions.kind(face).class() == boundary[face])
        .all(|x| x)
}

/// A generalization of `Condition<N>` for a coupled systems of scalar fields.
pub trait SystemBoundaryConds<const N: usize>: Clone {
    fn kind(&self, channel: usize, _face: Face<N>) -> BoundaryKind;

    fn radiative(&self, _channel: usize, _position: [f64; N]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }

    fn dirichlet(&self, _channel: usize, _position: [f64; N]) -> DirichletParams {
        DirichletParams {
            target: 0.0,
            strength: 1.0,
        }
    }

    fn field(&self, channel: usize) -> FieldBoundaryConds<N, Self> {
        FieldBoundaryConds::new(self.clone(), channel)
    }
}

/// Transfers a set of `Conditions<N>` into a single `Condition<N>` by only applying the set of conditions
/// to a single field.
pub struct FieldBoundaryConds<const N: usize, C> {
    conditions: C,
    channel: usize,
}

impl<const N: usize, C: SystemBoundaryConds<N>> FieldBoundaryConds<N, C> {
    pub const fn new(conditions: C, channel: usize) -> Self {
        Self {
            channel,
            conditions,
        }
    }
}

impl<const N: usize, C: SystemBoundaryConds<N>> Clone for FieldBoundaryConds<N, C> {
    fn clone(&self) -> Self {
        Self {
            conditions: self.conditions.clone(),
            channel: self.channel,
        }
    }
}

impl<const N: usize, C: SystemBoundaryConds<N>> BoundaryConds<N> for FieldBoundaryConds<N, C> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        self.conditions.kind(self.channel, face)
    }

    fn radiative(&self, position: [f64; N]) -> RadiativeParams {
        self.conditions.radiative(self.channel, position)
    }

    fn dirichlet(&self, position: [f64; N]) -> DirichletParams {
        self.conditions.dirichlet(self.channel, position)
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

impl<const N: usize, I: BoundaryConds<N>> SystemBoundaryConds<N> for ScalarConditions<I> {
    fn kind(&self, channel: usize, face: Face<N>) -> BoundaryKind {
        debug_assert!(channel == 0);
        self.0.kind(face)
    }

    fn radiative(&self, channel: usize, position: [f64; N]) -> RadiativeParams {
        debug_assert!(channel == 0);
        self.0.radiative(position)
    }

    fn dirichlet(&self, channel: usize, position: [f64; N]) -> DirichletParams {
        debug_assert!(channel == 0);
        self.0.dirichlet(position)
    }
}

#[derive(Clone)]
pub struct EmptyConditions;

impl<const N: usize> SystemBoundaryConds<N> for EmptyConditions {
    fn kind(&self, _channel: usize, _face: Face<N>) -> BoundaryKind {
        unreachable!()
    }

    fn radiative(&self, _channel: usize, _position: [f64; N]) -> RadiativeParams {
        unreachable!()
    }

    fn dirichlet(&self, _channel: usize, _position: [f64; N]) -> DirichletParams {
        unreachable!()
    }
}
