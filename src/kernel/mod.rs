//! A crate for approximating numerical operators on uniform rectangular meshes using finite differencing.

#![allow(clippy::needless_range_loop)]

mod boundary;
// mod convolution;
// mod element;
// mod node;
mod weights;

pub use boundary::{Boundary, BoundaryKind, Penalty, PenaltyKind, RadiativeParams};

use crate::IRef;

// *****************************
// Kernel **********************
// *****************************

/// Distance of a vertex from a boundary.
#[derive(Clone, Copy, Debug)]
pub enum Border {
    Negative(usize),
    Positive(usize),
}

impl Border {
    /// Returns false for negative borders and true for positive borders.
    pub fn side(self) -> bool {
        match self {
            Border::Negative(_) => false,
            Border::Positive(_) => true,
        }
    }
}

/// A vertex centered conventional stencil.
pub trait Kernel {
    /// How large is the centered stencil?
    fn border_width(&self) -> usize;
    /// Returns weights for a centered stencil.
    fn interior(&self) -> &[f64];
    /// Returns weights for the edge of a stencil
    fn free(&self, border: Border) -> &[f64];
    /// Additional scaling depending on the physical spacing of the mesh.
    fn scale(&self, spacing: f64) -> f64;
}

/// A cell-centered interpolating stencil
pub trait Interpolant {
    /// How large is the centered stencil?
    fn border_width(&self) -> usize;
    /// Returns weights for a centered stencil.
    fn interior(&self) -> &[f64];
    /// Returns weights for the edge of a stencil
    fn free(&self, border: Border) -> &[f64];
    /// Additional scaling of the stencil.
    fn scale(&self) -> f64;
}

impl<'a, B: Kernel> Kernel for IRef<'a, B> {
    fn border_width(&self) -> usize {
        self.0.border_width()
    }

    fn interior(&self) -> &[f64] {
        self.0.interior()
    }

    fn free(&self, border: Border) -> &[f64] {
        self.0.free(border)
    }

    fn scale(&self, spacing: f64) -> f64 {
        self.0.scale(spacing)
    }
}

impl<'a, B: Interpolant> Interpolant for IRef<'a, B> {
    fn border_width(&self) -> usize {
        self.0.border_width()
    }

    fn interior(&self) -> &[f64] {
        self.0.interior()
    }

    fn free(&self, border: Border) -> &[f64] {
        self.0.free(border)
    }

    fn scale(&self) -> f64 {
        self.0.scale()
    }
}
