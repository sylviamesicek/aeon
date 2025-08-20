//! A crate for approximating numerical operators on uniform rectangular meshes using finite differencing.

#![allow(clippy::needless_range_loop)]

mod boundary;
mod convolution;
mod element;
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

// // ************************************
// // Order ******************************
// // ************************************

// #[derive(Clone, Copy, Default)]
// pub struct Order<const ORDER: usize>;

// mod private {
//     pub trait Sealed {}
// }

// impl private::Sealed for Order<2> {}
// impl private::Sealed for Order<4> {}
// impl private::Sealed for Order<6> {}

// /// Associates an order with a type. Commonly used to set the order of accuracy for certain
// /// operators or boundary conditions.
// pub trait Kernels: private::Sealed + Clone + Copy + Default + 'static {
//     const ORDER: usize;
//     const MAX_BORDER: usize;

//     fn derivative() -> &'static impl Kernel;
//     fn second_derivative() -> &'static impl Kernel;
//     fn dissipation() -> &'static impl Kernel;
//     fn interpolation() -> &'static impl Interpolant;
// }

// impl Kernels for Order<2> {
//     const ORDER: usize = 2;
//     const MAX_BORDER: usize = 1;

//     fn derivative() -> &'static impl Kernel {
//         &Derivative::<2>
//     }

//     fn second_derivative() -> &'static impl Kernel {
//         &SecondDerivative::<2>
//     }

//     fn dissipation() -> &'static impl Kernel {
//         &Unimplemented(2)
//     }

//     fn interpolation() -> &'static impl Interpolant {
//         &Interpolation::<2>
//     }
// }

// impl Kernels for Order<4> {
//     const ORDER: usize = 4;
//     const MAX_BORDER: usize = 2;

//     fn derivative() -> &'static impl Kernel {
//         &Derivative::<4>
//     }

//     fn second_derivative() -> &'static impl Kernel {
//         &SecondDerivative::<4>
//     }

//     fn dissipation() -> &'static impl Kernel {
//         &Dissipation::<4>
//     }

//     fn interpolation() -> &'static impl Interpolant {
//         &Interpolation::<4>
//     }
// }

// impl Kernels for Order<6> {
//     const ORDER: usize = 6;
//     const MAX_BORDER: usize = 3;

//     fn derivative() -> &'static impl Kernel {
//         &Derivative::<6>
//     }

//     fn second_derivative() -> &'static impl Kernel {
//         &SecondDerivative::<6>
//     }

//     fn dissipation() -> &'static impl Kernel {
//         &Dissipation::<6>
//     }

//     fn interpolation() -> &'static impl Interpolant {
//         &Unimplemented(6)
//     }
// }
