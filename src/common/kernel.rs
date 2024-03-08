use super::stencils::*;
use aeon_macros::{derivative, second_derivative};

/// A seperable kernal used for approximating a derivative or numerical operator
/// to some order of accuracy.
pub trait Kernel {
    type InteriorStencil: VertexStencil;
    type BoundaryStencil: DirectedStencil;

    /// Stencil weights for the interior of the domain.
    fn interior() -> Self::InteriorStencil;

    /// Stencil weights for the negative edge of the domain.
    fn negative(left: usize) -> Self::BoundaryStencil;

    /// Stencil weights for the positive edge of the domain.
    fn positive(right: usize) -> Self::BoundaryStencil;

    /// Scale factor given spacing.
    fn scale(spacing: f64) -> f64;
}

pub struct FDDerivative<const ORDER: usize>;

impl Kernel for FDDerivative<2> {
    type InteriorStencil = CenteredSecondOrder;
    type BoundaryStencil = BoundaryStencil<3>;

    fn interior() -> Self::InteriorStencil {
        CenteredSecondOrder(derivative!(1, 1, 0))
    }

    fn negative(_: usize) -> Self::BoundaryStencil {
        BoundaryStencil(derivative!(1, 1, -1))
    }

    fn positive(_: usize) -> Self::BoundaryStencil {
        let mut result = derivative!(1, 1, 1);

        result.reverse();

        BoundaryStencil(result)
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / spacing
    }
}

impl Kernel for FDDerivative<4> {
    type InteriorStencil = CenteredFourthOrder;
    type BoundaryStencil = BoundaryStencil<5>;

    fn interior() -> Self::InteriorStencil {
        CenteredFourthOrder(derivative!(2, 2, 0))
    }

    fn negative(left: usize) -> Self::BoundaryStencil {
        if left == 0 {
            BoundaryStencil(derivative!(0, 4, 0))
        } else {
            BoundaryStencil(derivative!(0, 4, 1))
        }
    }

    fn positive(right: usize) -> Self::BoundaryStencil {
        let mut result = if right == 0 {
            derivative!(4, 0, 0)
        } else {
            derivative!(4, 0, -1)
        };

        result.reverse();

        BoundaryStencil(result)
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / spacing
    }
}

pub struct FDSecondDerivative<const ORDER: usize>;

impl Kernel for FDSecondDerivative<2> {
    type InteriorStencil = CenteredSecondOrder;
    type BoundaryStencil = BoundaryStencil<4>;

    fn interior() -> Self::InteriorStencil {
        CenteredSecondOrder(second_derivative!(1, 1, 0))
    }

    fn negative(_: usize) -> Self::BoundaryStencil {
        BoundaryStencil(second_derivative!(0, 3, 0))
    }

    fn positive(_: usize) -> Self::BoundaryStencil {
        let mut result = second_derivative!(3, 0, 0);
        result.reverse();
        BoundaryStencil(result)
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

impl Kernel for FDSecondDerivative<4> {
    type InteriorStencil = CenteredFourthOrder;
    type BoundaryStencil = BoundaryStencil<6>;

    fn interior() -> Self::InteriorStencil {
        CenteredFourthOrder(second_derivative!(2, 2, 0))
    }

    fn negative(left: usize) -> Self::BoundaryStencil {
        if left == 0 {
            BoundaryStencil(second_derivative!(0, 5, 0))
        } else {
            BoundaryStencil(second_derivative!(0, 5, 1))
        }
    }

    fn positive(right: usize) -> Self::BoundaryStencil {
        let mut result = if right == 0 {
            second_derivative!(5, 0, 0)
        } else {
            second_derivative!(5, 0, -1)
        };

        result.reverse();

        BoundaryStencil(result)
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

// pub trait Transfer {
//     type InteriorStencil: CellStencil;
//     type BoundaryStencil: DirectedStencil;

//     fn interior() -> Self::InteriorStencil;
//     fn negative(left: usize) -> Self::BoundaryStencil;
//     fn positive(right: usize) -> Self::BoundaryStencil;
// }

// pub struct FDProlong<const ORDER: usize>;

// impl Transfer for FDProlong<2> {
//     type InteriorStencil = ProlongSecondOrder;
//     type BoundaryStencil = BoundaryStencil<2>;

//     fn interior() -> Self::InteriorStencil {
//         ProlongSecondOrder(prolong!(1, 1, 0))
//     }

//     fn negative(left: usize) -> Self::BoundaryStencil {
//         panic!("2nd Order Stencil has no boundary support")
//     }

//     fn positive(right: usize) -> Self::BoundaryStencil {
//         panic!("2nd Order Stencil has no boundary support")
//     }
// }

// impl Transfer for FDProlong<4> {
//     type InteriorStencil = ProlongFourthOrder;
//     type BoundaryStencil = BoundaryStencil<4>;

//     fn interior() -> Self::InteriorStencil {
//         ProlongFourthOrder(prolong!(2, 2, 0))
//     }

//     fn negative(left: usize) -> Self::BoundaryStencil {
//         assert!(left == 1);

//         BoundaryStencil(prolong!(1, 3, 0))
//     }

//     fn positive(right: usize) -> Self::BoundaryStencil {
//         assert!(right == 1);

//         BoundaryStencil(prolong!(3, 1, 0))
//     }
// }
