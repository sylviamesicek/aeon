//! A crate for approximating numerical operators on uniform rectangular meshes using finite differencing.

#![allow(clippy::needless_range_loop)]

mod boundary;
mod convolution;
mod element;
mod element2;
mod node;
mod weights;

pub use boundary::{
    BoundaryClass, BoundaryConds, BoundaryKind, DirichletParams, RadiativeParams,
    is_boundary_compatible,
};
pub use convolution::{Convolution, Gradient, Hessian};
pub use element::Element;
pub use node::{
    NodeCartesianIter, NodePlaneIter, NodeSpace, NodeWindow, node_from_vertex, vertex_from_node,
};
pub use weights::{
    Border, Derivative, Dissipation, Interpolation, SecondDerivative, Unimplemented, Value,
};

// *****************************
// Kernel **********************
// *****************************

pub trait Kernel: Clone {
    fn border_width(&self) -> usize;

    fn interior(&self) -> &[f64];
    fn free(&self, border: Border) -> &[f64];
}

pub trait VertexKernel: Kernel {
    fn scale(&self, spacing: f64) -> f64;
}

/// A kernel which is used for prolonging values between levels.
pub trait CellKernel: Kernel {
    fn scale(&self) -> f64;
}

// ************************************
// Order ******************************
// ************************************

#[derive(Clone, Copy, Default)]
pub struct Order<const ORDER: usize>;

mod private {
    pub trait Sealed {}
}

impl private::Sealed for Order<2> {}
impl private::Sealed for Order<4> {}
impl private::Sealed for Order<6> {}

/// Associates an order with a type. Commonly used to set the order of accuracy for certain
/// operators or boundary conditions.
pub trait Kernels: private::Sealed + Clone + Copy + Default + 'static {
    const ORDER: usize;
    const MAX_BORDER: usize;

    fn derivative() -> &'static impl VertexKernel;
    fn second_derivative() -> &'static impl VertexKernel;
    fn dissipation() -> &'static impl VertexKernel;
    fn interpolation() -> &'static impl CellKernel;
}

impl Kernels for Order<2> {
    const ORDER: usize = 2;
    const MAX_BORDER: usize = 1;

    fn derivative() -> &'static impl VertexKernel {
        &Derivative::<2>
    }

    fn second_derivative() -> &'static impl VertexKernel {
        &SecondDerivative::<2>
    }

    fn dissipation() -> &'static impl VertexKernel {
        &Unimplemented(2)
    }

    fn interpolation() -> &'static impl CellKernel {
        &Interpolation::<2>
    }
}

impl Kernels for Order<4> {
    const ORDER: usize = 4;
    const MAX_BORDER: usize = 2;

    fn derivative() -> &'static impl VertexKernel {
        &Derivative::<4>
    }

    fn second_derivative() -> &'static impl VertexKernel {
        &SecondDerivative::<4>
    }

    fn dissipation() -> &'static impl VertexKernel {
        &Dissipation::<4>
    }

    fn interpolation() -> &'static impl CellKernel {
        &Interpolation::<4>
    }
}

impl Kernels for Order<6> {
    const ORDER: usize = 6;
    const MAX_BORDER: usize = 3;

    fn derivative() -> &'static impl VertexKernel {
        &Derivative::<6>
    }

    fn second_derivative() -> &'static impl VertexKernel {
        &SecondDerivative::<6>
    }

    fn dissipation() -> &'static impl VertexKernel {
        &Dissipation::<6>
    }

    fn interpolation() -> &'static impl CellKernel {
        &Unimplemented(6)
    }
}
