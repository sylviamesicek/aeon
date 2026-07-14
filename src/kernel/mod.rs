//! A crate for approximating numerical operators on uniform rectangular meshes using finite differencing.

#![allow(clippy::needless_range_loop)]

mod boundary;
mod convolution;
mod element;
mod node;
mod weights;

pub use boundary::{
    Boundary, BoundaryClass, BoundaryKind, DirichletParams, EmptyBoundary, RadiativeParams,
    is_boundary_compatible,
};
pub use convolution::{Convolution, Gradient, Hessian};
pub use element::Element;
pub use node::{
    NodeCartesianIter, NodePlaneIter, NodeSpace, NodeWindow, node_from_vertex, vertex_from_node,
};
pub use weights::{Derivative, Dissipation, Interpolation, SecondDerivative, Unimplemented, Value};

use crate::IRef;

// **************************
// Border *******************
// **************************

#[derive(Debug, Clone, Copy)]
pub enum Border {
    Negative(usize),
    Positive(usize),
}

impl Border {
    /// Returns true if the border is near the positive edge and false otherwise.
    pub fn side(self) -> bool {
        match self {
            Border::Negative(_) => false,
            Border::Positive(_) => true,
        }
    }
}

// *****************************
// Kernel **********************
// *****************************

/// A weighted kernel that can be applied to a field, often to compute derivatives
/// or perform interpolation
pub trait Kernel {
    /// How far does the kernel's edge extend?
    fn border_width(&self) -> usize;
    /// Weights for the kernel on the interior of the domain, with equal supports on either side.
    fn interior(&self) -> &[f64];
    /// Weights for kernel near the boundary of domain.
    fn free(&self, border: Border) -> &[f64];
    /// A scale factor depending on the physical spacing of the support nodes of the field.
    fn scale(&self, spacing: f64) -> f64;
}

impl<'a, T: Kernel> Kernel for IRef<'a, T> {
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

pub trait Interpolant {
    fn border_width(&self) -> usize;
    fn interior(&self) -> &[f64];
    fn free(&self, border: Border) -> &[f64];
    fn scale(&self) -> f64;
}

impl<'a, T: Interpolant> Interpolant for IRef<'a, T> {
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
