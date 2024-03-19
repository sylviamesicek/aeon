use crate::array::Array;
use aeon_macros::{derivative, second_derivative};

/// A seperable kernal used for approximating a derivative or numerical operator
/// to some order of accuracy. All kernel weights are applied negative to positive.
pub trait Kernel {
    type InteriorStencil: Array<f64>;
    type BoundaryStencil: Array<f64>;

    const POSITIVE_SUPPORT: usize;
    const NEGATIVE_SUPPORT: usize;

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
    type InteriorStencil = [f64; 3];
    type BoundaryStencil = [f64; 3];

    const POSITIVE_SUPPORT: usize = 1;
    const NEGATIVE_SUPPORT: usize = 1;

    fn interior() -> Self::InteriorStencil {
        derivative!(1, 1, 0)
    }

    fn negative(_: usize) -> Self::BoundaryStencil {
        derivative!(1, 1, -1)
    }

    fn positive(_: usize) -> Self::BoundaryStencil {
        derivative!(1, 1, 1)
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / spacing
    }
}

impl Kernel for FDDerivative<4> {
    type InteriorStencil = [f64; 5];
    type BoundaryStencil = [f64; 5];

    const POSITIVE_SUPPORT: usize = 2;
    const NEGATIVE_SUPPORT: usize = 2;

    fn interior() -> Self::InteriorStencil {
        derivative!(2, 2, 0)
    }

    fn negative(left: usize) -> Self::BoundaryStencil {
        if left == 0 {
            derivative!(0, 4, 0)
        } else {
            derivative!(0, 4, 1)
        }
    }

    fn positive(right: usize) -> Self::BoundaryStencil {
        if right == 0 {
            derivative!(4, 0, 0)
        } else {
            derivative!(4, 0, -1)
        }
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / spacing
    }
}

pub struct FDSecondDerivative<const ORDER: usize>;

impl Kernel for FDSecondDerivative<2> {
    type InteriorStencil = [f64; 3];
    type BoundaryStencil = [f64; 4];

    const NEGATIVE_SUPPORT: usize = 1;
    const POSITIVE_SUPPORT: usize = 1;

    fn interior() -> Self::InteriorStencil {
        second_derivative!(1, 1, 0)
    }

    fn negative(_: usize) -> Self::BoundaryStencil {
        second_derivative!(0, 3, 0)
    }

    fn positive(_: usize) -> Self::BoundaryStencil {
        second_derivative!(3, 0, 0)
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

impl Kernel for FDSecondDerivative<4> {
    type InteriorStencil = [f64; 5];
    type BoundaryStencil = [f64; 6];

    const NEGATIVE_SUPPORT: usize = 2;
    const POSITIVE_SUPPORT: usize = 2;

    fn interior() -> Self::InteriorStencil {
        second_derivative!(2, 2, 0)
    }

    fn negative(left: usize) -> Self::BoundaryStencil {
        if left == 0 {
            second_derivative!(0, 5, 0)
        } else {
            second_derivative!(0, 5, 1)
        }
    }

    fn positive(right: usize) -> Self::BoundaryStencil {
        if right == 0 {
            second_derivative!(5, 0, 0)
        } else {
            second_derivative!(5, 0, -1)
        }
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

pub struct FDDissipation<const ORDER: usize>;

impl Kernel for FDDissipation<2> {
    type InteriorStencil = [f64; 5];
    type BoundaryStencil = [f64; 5];

    const NEGATIVE_SUPPORT: usize = 2;
    const POSITIVE_SUPPORT: usize = 2;

    fn interior() -> Self::InteriorStencil {
        [1.0, -4.0, 6.0, -4.0, 1.0]
    }

    fn negative(_: usize) -> Self::BoundaryStencil {
        [1.0, -4.0, 6.0, -4.0, 1.0]
    }

    fn positive(_: usize) -> Self::BoundaryStencil {
        [1.0, -4.0, 6.0, -4.0, 1.0]
    }

    fn scale(_: f64) -> f64 {
        -1.0 / 16.0
    }
}

impl Kernel for FDDissipation<4> {
    type InteriorStencil = [f64; 7];
    type BoundaryStencil = [f64; 7];

    const NEGATIVE_SUPPORT: usize = 3;
    const POSITIVE_SUPPORT: usize = 3;

    fn interior() -> Self::InteriorStencil {
        [1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]
    }

    fn negative(_: usize) -> Self::BoundaryStencil {
        [1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]
    }

    fn positive(_: usize) -> Self::BoundaryStencil {
        [1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]
    }

    fn scale(_: f64) -> f64 {
        1.0 / 64.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_kernel_weights() {
        assert_eq!(FDDerivative::<2>::negative(0), [-1.5, 2.0, -0.5]);
        assert_eq!(FDDerivative::<2>::interior(), [-0.5, 0.0, 0.5]);
        assert_eq!(FDDerivative::<2>::positive(0), [0.5, -2.0, 1.5]);

        assert_eq!(
            FDDerivative::<4>::negative(0),
            [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0]
        );
        assert_eq!(
            FDDerivative::<4>::interior(),
            [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]
        );
        assert_eq!(
            FDDerivative::<4>::positive(0),
            [1.0 / 4.0, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0]
        );

        assert_eq!(FDSecondDerivative::<2>::negative(0), [2.0, -5.0, 4.0, -1.0]);
        assert_eq!(FDSecondDerivative::<2>::interior(), [1.0, -2.0, 1.0]);
        assert_eq!(FDSecondDerivative::<2>::positive(0), [-1.0, 4.0, -5.0, 2.0]);
    }
}
