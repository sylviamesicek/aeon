/// Represents a stencil used for a kernel. This provides
/// the number of support points on each side of the kernel,
/// as well as a type representing an array of weights.
pub trait Stencil {
    const POSITIVE: usize;
    const NEGATIVE: usize;

    const EDGE: usize;

    type NegWeights: IntoIterator<Item = f64>;
    type IntWeights: IntoIterator<Item = f64>;
    type PosWeights: IntoIterator<Item = f64>;
}

pub struct FDStencil<const ORDER: usize>;

impl Stencil for FDStencil<1> {
    const POSITIVE: usize = 1;
    const NEGATIVE: usize = 1;
    const EDGE: usize = 4;

    type NegWeights = [f64; 4];
    type IntWeights = [f64; 3];
    type PosWeights = [f64; 4];
}

impl Stencil for FDStencil<2> {
    const POSITIVE: usize = 2;
    const NEGATIVE: usize = 2;
    const EDGE: usize = 6;

    type NegWeights = [f64; 6];
    type IntWeights = [f64; 5];
    type PosWeights = [f64; 6];
}

/// A seperable kernal used for approximating a derivative or numerical operator
/// to some order of accuracy.
pub trait Kernel {
    type Stencil: Stencil;

    /// Stencil weights for the interior of the domain.
    fn interior() -> <Self::Stencil as Stencil>::IntWeights;

    /// Stencil weights for the negative edge of the domain.
    fn negative(left: usize) -> <Self::Stencil as Stencil>::NegWeights;

    /// Stencil weights for the positive edge of the domain.
    fn positive(right: usize) -> <Self::Stencil as Stencil>::PosWeights;

    /// Scale factor given spacing.
    fn scale(spacing: f64) -> f64;
}

pub struct FDDerivative<const ORDER: usize, const RANK: usize>;

impl Kernel for FDDerivative<1, 1> {
    type Stencil = FDStencil<1>;

    fn interior() -> <Self::Stencil as Stencil>::IntWeights {
        [-0.5, 0.0, 0.5]
    }

    fn negative(_: usize) -> <Self::Stencil as Stencil>::NegWeights {
        [-1.5, 2.0, -0.5, 0.0]
    }

    fn positive(_: usize) -> <Self::Stencil as Stencil>::PosWeights {
        [0.0, 0.5, -2.0, 1.5]
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / spacing
    }
}

impl Kernel for FDDerivative<1, 2> {
    type Stencil = FDStencil<1>;

    fn interior() -> <Self::Stencil as Stencil>::IntWeights {
        [1.0, -2.0, 1.0]
    }

    fn negative(_: usize) -> <Self::Stencil as Stencil>::NegWeights {
        [2.0, -5.0, 4.0, -1.0]
    }

    fn positive(_: usize) -> <Self::Stencil as Stencil>::PosWeights {
        [-1.0, 4.0, -5.0, 2.0]
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

impl Kernel for FDDerivative<2, 1> {
    type Stencil = FDStencil<2>;

    fn interior() -> <Self::Stencil as Stencil>::IntWeights {
        [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]
    }

    fn negative(left: usize) -> <Self::Stencil as Stencil>::NegWeights {
        if left == 0 {
            [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0, 0.0]
        } else {
            [
                -1.0 / 4.0,
                -5.0 / 6.0,
                3.0 / 2.0,
                -1.0 / 2.0,
                1.0 / 12.0,
                0.0,
            ]
        }
    }

    fn positive(right: usize) -> <Self::Stencil as Stencil>::PosWeights {
        if right == 0 {
            [0.0, 1.0 / 4.0, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0]
        } else {
            [
                0.0,
                -1.0 / 12.0,
                1.0 / 2.0,
                -3.0 / 2.0,
                5.0 / 6.0,
                1.0 / 4.0,
            ]
        }
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / spacing
    }
}

impl Kernel for FDDerivative<2, 2> {
    type Stencil = FDStencil<2>;

    fn interior() -> <Self::Stencil as Stencil>::IntWeights {
        [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
    }

    fn negative(left: usize) -> <Self::Stencil as Stencil>::NegWeights {
        if left == 0 {
            [
                15.0 / 4.0,
                -77.0 / 6.0,
                107.0 / 6.0,
                -13.0,
                61.0 / 12.0,
                -5.0 / 6.0,
            ]
        } else {
            [
                5.0 / 6.0,
                -5.0 / 4.0,
                -1.0 / 3.0,
                7.0 / 6.0,
                -1.0 / 2.0,
                1.0 / 12.0,
            ]
        }
    }

    fn positive(right: usize) -> <Self::Stencil as Stencil>::PosWeights {
        if right == 0 {
            [
                -5.0 / 6.0,
                61.0 / 12.0,
                -13.0,
                107.0 / 6.0,
                -77.0 / 6.0,
                15.0 / 4.0,
            ]
        } else {
            [
                1.0 / 12.0,
                -1.0 / 2.0,
                7.0 / 6.0,
                -1.0 / 3.0,
                -5.0 / 4.0,
                5.0 / 6.0,
            ]
        }
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}
