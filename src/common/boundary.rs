use super::stencils::{BoundaryStencil, DirectedStencil};
use aeon_stencils::boundary_derivative;

pub trait Boundary {
    const EXTENT: usize;

    type Stencil: DirectedStencil;

    fn stencil(self: &Self, extent: usize) -> Self::Stencil;
}

#[derive(Debug, Clone, Copy)]
pub struct FreeBoundary;

impl Boundary for FreeBoundary {
    const EXTENT: usize = 0;

    type Stencil = BoundaryStencil<0>;

    fn stencil(self: &Self, _: usize) -> Self::Stencil {
        BoundaryStencil([])
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SymmetricBoundary<const EXTENT: usize>;

impl<const EXTENT: usize> Boundary for SymmetricBoundary<EXTENT> {
    const EXTENT: usize = EXTENT;

    type Stencil = BoundaryStencil<EXTENT>;

    fn stencil(self: &Self, extent: usize) -> Self::Stencil {
        let mut result = [0.0; EXTENT];

        result[extent] = 1.0;

        BoundaryStencil(result)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AntiSymmetricBoundary<const ORDER: usize>;

impl<const EXTENT: usize> Boundary for AntiSymmetricBoundary<EXTENT> {
    const EXTENT: usize = EXTENT;

    type Stencil = BoundaryStencil<EXTENT>;

    fn stencil(self: &Self, extent: usize) -> Self::Stencil {
        let mut result = [0.0; EXTENT];

        result[extent] = -1.0;

        BoundaryStencil(result)
    }
}

pub struct RobinBoundary<const SUPPORT: usize> {
    pub spacing: f64,
    pub coef: f64,
}

// macro_rules! robin_boundary_impl {
//     ($n:expr, $t:expr, $($ts:expr),*) => {
//         impl Boundary for RobinBoundary<$n> {
//             const EXTENT: usize = 1;

//             type Stencil = BoundaryStencil<$n>;

//             fn stencil(self: &Self, _: usize) -> Self::Stencil {
//                 let mut derivative = boundary_derivative!(($n - 1), 0);

//                 for i in 0..derivative.len() {
//                     derivative[i] /= self.spacing;
//                 }

//                 let mut result = [0.0; $n];

//                 $(
//                     result[$ts] = -derivative[$ts];
//                 )*

//                 result[0] += self.coef;

//                 for i in 0..result.len() {
//                     result[i] /= derivative[$t];
//                 }

//                 BoundaryStencil(result)
//             }
//         }

//         robin_boundary_impl! { ($n - 1), $($ts)* }
//     };
//     ($n: expr) => {}
// }

// robin_boundary_impl!(5, 4, 3, 2, 1, 0);

impl Boundary for RobinBoundary<1> {
    const EXTENT: usize = 1;

    type Stencil = BoundaryStencil<1>;

    fn stencil(self: &Self, _: usize) -> Self::Stencil {
        let mut derivative = boundary_derivative!(2, 0);

        for i in 0..derivative.len() {
            derivative[i] /= self.spacing;
        }

        BoundaryStencil([1.0 / derivative[1] * (self.coef - derivative[0])])
    }
}

impl Boundary for RobinBoundary<2> {
    const EXTENT: usize = 1;

    type Stencil = BoundaryStencil<2>;

    fn stencil(self: &Self, _: usize) -> Self::Stencil {
        let mut derivative = boundary_derivative!(3, 0);

        for i in 0..derivative.len() {
            derivative[i] /= self.spacing;
        }

        let mut result = [0.0; 2];

        result[0] = -derivative[0];
        result[1] = -derivative[1];

        result[0] += self.coef;

        for i in 0..result.len() {
            result[i] /= derivative[2];
        }

        BoundaryStencil(result)
    }
}

impl Boundary for RobinBoundary<5> {
    const EXTENT: usize = 1;

    type Stencil = BoundaryStencil<5>;

    fn stencil(self: &Self, _: usize) -> Self::Stencil {
        let mut derivative = boundary_derivative!(6, 0);

        for i in 0..derivative.len() {
            derivative[i] /= self.spacing;
        }

        let mut result = [0.0; 5];

        result[0] = -derivative[0];
        result[1] = -derivative[1];
        result[2] = -derivative[2];
        result[3] = -derivative[3];
        result[4] = -derivative[4];

        result[0] += self.coef;

        for i in 0..result.len() {
            result[i] /= derivative[5];
        }

        BoundaryStencil(result)
    }
}
