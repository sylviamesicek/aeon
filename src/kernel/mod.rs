//! A crate for approximating numerical operators on uniform rectangular meshes using finite differencing.

#![allow(clippy::needless_range_loop)]

mod boundary;
mod convolution;
mod element;
mod node;
mod weights;

pub use boundary::{
    BoundaryClass, BoundaryConds, BoundaryKind, DirichletParams, EmptyConditions, RadiativeParams,
    ScalarConditions, SystemBoundaryConds, is_boundary_compatible,
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

pub trait Kernel {
    fn border_width(&self) -> usize;
    fn interior(&self) -> &[f64];
    fn free(&self, border: Border) -> &[f64];
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
