//! Provides generalized interfaces for expressing various kinds of boundary condition.
//!
//! This module uses a combination of trait trickery and type transformers to
//! create an ergonomic API for working with boundaries.

use crate::{
    IRef,
    geometry::{Face, FaceArray},
};

#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum BoundaryKind {
    /// Symmetry boundary conditions, where function is either even or odd along face.
    Symmetric,
    AntiSymmetric,
    /// Strongly enforced dirichlet conditions, where function is set to a specific value along the face.
    Dirichlet,
    /// No boundary condition is used along this face, we simply switch to one-sided stencils.
    #[default]
    Free,
}

impl BoundaryKind {
    /// Returns true if this boundary condition requires one-sided stencils along the given face
    pub fn is_one_sided(self) -> bool {
        matches!(self, Self::Free | Self::Dirichlet)
    }
}

pub trait Boundary<const N: usize> {
    /// What boundary condition should we apply to the `index`th boundary, for the given channel,
    /// at the given position?
    fn kind(&self, index: usize, channel: usize) -> BoundaryKind;
    fn dirichlet(&self, channel: usize, position: [f64; N]) -> f64 {
        0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum PenaltyKind {
    #[default]
    Free,
    Radiative,
    Dirichlet,
}

#[derive(Debug, Clone, Copy)]
pub struct RadiativeParams {
    pub target: f64,
    pub speed: f64,
}

pub trait Penalty<const N: usize> {
    /// What penalty term should we apply to the `index`th boundary for the given channel,
    /// at the given position?
    fn kind(&self, index: usize, channel: usize) -> PenaltyKind;
    fn radiative(&self, channel: usize, position: [f64; N]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
    fn dirichlet(&self, channel: usize, position: [f64; N]) -> f64 {
        0.0
    }
}

impl<'a, const N: usize, B: Boundary<N>> Boundary<N> for IRef<'a, B> {
    fn kind(&self, index: usize, channel: usize) -> BoundaryKind {
        self.0.kind(index, channel)
    }

    fn dirichlet(&self, channel: usize, position: [f64; N]) -> f64 {
        self.0.dirichlet(channel, position)
    }
}

impl<'a, const N: usize, B: Penalty<N>> Penalty<N> for IRef<'a, B> {
    fn kind(&self, index: usize, channel: usize) -> PenaltyKind {
        self.0.kind(index, channel)
    }

    fn radiative(&self, channel: usize, position: [f64; N]) -> RadiativeParams {
        self.0.radiative(channel, position)
    }

    fn dirichlet(&self, channel: usize, position: [f64; N]) -> f64 {
        self.0.dirichlet(channel, position)
    }
}
