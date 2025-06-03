use crate::kernel::{CellKernel, Kernel, VertexKernel};
use aeon_macros::{derivative, second_derivative};

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

// *********************************
// Kernel **************************
// *********************************

/// Value operation.
#[derive(Clone)]
pub struct Value;

impl Kernel for Value {
    fn border_width(&self) -> usize {
        0
    }

    fn interior(&self) -> &[f64] {
        &[1.0]
    }

    fn free(&self, _border: Border) -> &[f64] {
        &[1.0]
    }
}

impl VertexKernel for Value {
    fn scale(&self, _spacing: f64) -> f64 {
        1.0
    }
}

#[derive(Clone)]
pub struct Unimplemented(pub usize);

impl Kernel for Unimplemented {
    fn border_width(&self) -> usize {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }

    fn interior(&self) -> &[f64] {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }

    fn free(&self, _border: Border) -> &[f64] {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }
}

impl VertexKernel for Unimplemented {
    fn scale(&self, _spacing: f64) -> f64 {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }
}

impl CellKernel for Unimplemented {
    fn scale(&self) -> f64 {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }
}

/// Derivative operation of a given order.
#[derive(Clone)]
pub struct Derivative<const ORDER: usize>;

impl Kernel for Derivative<2> {
    fn border_width(&self) -> usize {
        1
    }

    fn interior(&self) -> &[f64] {
        &derivative!(1, 1, 0)
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(_) => &derivative!(0, 2, 0),
            Border::Positive(_) => &derivative!(2, 0, 0),
        }
    }
}

impl Kernel for Derivative<4> {
    fn border_width(&self) -> usize {
        2
    }

    fn interior(&self) -> &[f64] {
        &derivative!(2, 2, 0)
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &derivative!(0, 4, 0),
            Border::Negative(_) => &derivative!(0, 4, 1),
            Border::Positive(0) => &derivative!(4, 0, 0),
            Border::Positive(_) => &derivative!(4, 0, -1),
        }
    }
}

impl Kernel for Derivative<6> {
    fn border_width(&self) -> usize {
        3
    }

    fn interior(&self) -> &[f64] {
        &derivative!(3, 3, 0)
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &derivative!(0, 6, 0),
            Border::Negative(1) => &derivative!(0, 6, 1),
            Border::Negative(_) => &derivative!(0, 6, 2),
            Border::Positive(0) => &derivative!(6, 0, 0),
            Border::Positive(1) => &derivative!(6, 0, -1),
            Border::Positive(_) => &derivative!(6, 0, -2),
        }
    }
}

impl<const ORDER: usize> VertexKernel for Derivative<ORDER>
where
    Derivative<ORDER>: Kernel,
{
    fn scale(&self, spacing: f64) -> f64 {
        1.0 / spacing
    }
}

/// Second derivative operator of a given order.
#[derive(Clone)]
pub struct SecondDerivative<const ORDER: usize>;

impl Kernel for SecondDerivative<2> {
    fn border_width(&self) -> usize {
        1
    }

    fn interior(&self) -> &[f64] {
        &second_derivative!(1, 1, 0)
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(_) => &second_derivative!(0, 3, 0),
            Border::Positive(_) => &second_derivative!(3, 0, 0),
        }
    }
}

impl Kernel for SecondDerivative<4> {
    fn border_width(&self) -> usize {
        2
    }

    fn interior(&self) -> &[f64] {
        &second_derivative!(2, 2, 0)
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &second_derivative!(0, 5, 0),
            Border::Negative(_) => &second_derivative!(0, 5, 1),
            Border::Positive(0) => &second_derivative!(5, 0, 0),
            Border::Positive(_) => &second_derivative!(5, 0, -1),
        }
    }
}

impl Kernel for SecondDerivative<6> {
    fn border_width(&self) -> usize {
        3
    }

    fn interior(&self) -> &[f64] {
        &second_derivative!(3, 3, 0)
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &second_derivative!(0, 7, 0),
            Border::Negative(1) => &second_derivative!(0, 7, 1),
            Border::Negative(_) => &second_derivative!(0, 7, 2),
            Border::Positive(0) => &second_derivative!(7, 0, 0),
            Border::Positive(1) => &second_derivative!(7, 0, -1),
            Border::Positive(_) => &second_derivative!(7, 0, -2),
        }
    }
}

impl<const ORDER: usize> VertexKernel for SecondDerivative<ORDER>
where
    SecondDerivative<ORDER>: Kernel,
{
    fn scale(&self, spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

/// Kriss Olgier dissipation of the given order.
#[derive(Clone)]
pub struct Dissipation<const ORDER: usize>;

impl Kernel for Dissipation<4> {
    fn border_width(&self) -> usize {
        2
    }

    fn interior(&self) -> &[f64] {
        &[1.0, -4.0, 6.0, -4.0, 1.0]
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &[3.0, -14.0, 26.0, -24.0, 11.0, -2.0],
            Border::Negative(_) => &[2.0, -9.0, 16.0, -14.0, 6.0, -1.0],
            Border::Positive(0) => &[-2.0, 11.0, -24.0, 26.0, -14.0, 3.0],
            Border::Positive(_) => &[-1.0, 6.0, -14.0, 16.0, -9.0, 2.0],
        }
    }
}

impl VertexKernel for Dissipation<4> {
    fn scale(&self, _spacing: f64) -> f64 {
        -1.0 / 16.0
    }
}

impl Kernel for Dissipation<6> {
    fn border_width(&self) -> usize {
        3
    }

    fn interior(&self) -> &[f64] {
        &[1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &[4.0, -27.0, 78.0, -125.0, 120.0, -69.0, 22.0, -3.0],
            Border::Negative(1) => &[3.0, -20.0, 57.0, -90.0, 85.0, -48.0, 15.0, -2.0],
            Border::Negative(_) => &[2.0, -13.0, 36.0, -55.0, 50.0, -27.0, 8.0, -1.0],
            Border::Positive(0) => &[-3.0, 22.0, -69.0, 120.0, -125.0, 78.0, -27.0, 4.0],
            Border::Positive(1) => &[-2.0, 15.0, -48.0, 85.0, -90.0, 57.0, -20.0, 3.0],
            Border::Positive(_) => &[-1.0, 8.0, -27.0, 50.0, -55.0, 36.0, -13.0, 2.0],
        }
    }
}

impl VertexKernel for Dissipation<6> {
    fn scale(&self, _spacing: f64) -> f64 {
        1.0 / 64.0
    }
}

#[derive(Clone)]
pub struct Interpolation<const ORDER: usize>;

impl Kernel for Interpolation<2> {
    fn border_width(&self) -> usize {
        1
    }

    fn interior(&self) -> &[f64] {
        &[-1.0, 9.0, 9.0, -1.0]
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Positive(_) => &[1.0, -5.0, 15.0, 5.0],
            Border::Negative(_) => &[5.0, 15.0, -5.0, 1.0],
        }
    }

    // fn symmetric(&self, border: Border) -> &[f64] {
    //     match border {
    //         Border::Negative(_) => &[9.0, 8.0, -1.0],
    //         Border::Positive(_) => &[-1.0, 8.0, 9.0],
    //     }
    // }

    // fn antisymmetric(&self, border: Border) -> &[f64] {
    //     match border {
    //         Border::Negative(_) => &[9.0, 10.0, -1.0],
    //         Border::Positive(_) => &[-1.0, 10.0, 9.0],
    //     }
    // }
}

impl CellKernel for Interpolation<2> {
    fn scale(&self) -> f64 {
        1.0 / 16.0
    }
}

impl Kernel for Interpolation<4> {
    fn border_width(&self) -> usize {
        2
    }

    fn interior(&self) -> &[f64] {
        &[3.0, -25.0, 150.0, 150.0, -25.0, 3.0]
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &[63.0, 315.0, -210.0, 126.0, -45.0, 7.0],
            Border::Negative(_) => &[-7.0, 105.0, 210.0, -70.0, 21.0, -3.0],
            Border::Positive(0) => &[7.0, -45.0, 126.0, -210.0, 315.0, 63.0],
            Border::Positive(_) => &[-3.0, 21.0, -70.0, 210.0, 105.0, -7.0],
        }
    }

    // fn symmetric(&self, border: Border) -> &[f64] {
    //     match border {
    //         Border::Negative(0) => &[150.0, 125.0, -22.0, 3.0],
    //         Border::Negative(_) => &[-25.0, 153.0, 150.0, -25.0, 3.0],
    //         Border::Positive(0) => &[3.0, -22.0, 125.0, 150.0],
    //         Border::Positive(_) => &[3.0, -25.0, 150.0, 153.0, -25.0],
    //     }
    // }

    // fn antisymmetric(&self, border: Border) -> &[f64] {
    //     match border {
    //         Border::Negative(0) => &[150.0, 175.0, -28.0, 3.0],
    //         Border::Negative(_) => &[-25.0, 147.0, 150.0, -25.0, 3.0],
    //         Border::Positive(0) => &[3.0, -28.0, 175.0, 150.0],
    //         Border::Positive(_) => &[3.0, -25.0, 150.0, 147.0, -28.0],
    //     }
    // }
}

impl CellKernel for Interpolation<4> {
    fn scale(&self) -> f64 {
        1.0 / 256.0
    }
}
