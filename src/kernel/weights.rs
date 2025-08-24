use crate::kernel::{Border, Interpolant, Kernel};
use aeon_macros::{derivative, second_derivative};

/// Unimplemented Kernel (used for debugging)
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

    fn scale(&self, _spacing: f64) -> f64 {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }
}

impl Interpolant for Unimplemented {
    fn border_width(&self) -> usize {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }

    fn interior(&self) -> &[f64] {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }

    fn free(&self, _border: Border) -> &[f64] {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }

    fn scale(&self) -> f64 {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }
}

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

    fn scale(&self, _spacing: f64) -> f64 {
        1.0
    }
}

/// Derivative operation of a given order.
#[derive(Clone)]
pub struct Derivative<const ORDER: usize>;

impl<const ORDER: usize> Kernel for Derivative<ORDER> {
    fn border_width(&self) -> usize {
        ORDER / 2
    }

    fn interior(&self) -> &[f64] {
        match ORDER {
            2 => &derivative!(1, 1, 0),
            4 => &derivative!(2, 2, 0),
            6 => &derivative!(3, 3, 0),
            _ => unimplemented!("Kernel is unimplemented for order {}", ORDER),
        }
    }

    fn free(&self, border: Border) -> &[f64] {
        match ORDER {
            2 => match border {
                Border::Negative(_) => &derivative!(0, 2, 0),
                Border::Positive(_) => &derivative!(2, 0, 0),
            },
            4 => match border {
                Border::Negative(0) => &derivative!(0, 4, 0),
                Border::Negative(_) => &derivative!(0, 4, 1),
                Border::Positive(0) => &derivative!(4, 0, 0),
                Border::Positive(_) => &derivative!(4, 0, -1),
            },
            6 => match border {
                Border::Negative(0) => &derivative!(0, 6, 0),
                Border::Negative(1) => &derivative!(0, 6, 1),
                Border::Negative(_) => &derivative!(0, 6, 2),
                Border::Positive(0) => &derivative!(6, 0, 0),
                Border::Positive(1) => &derivative!(6, 0, -1),
                Border::Positive(_) => &derivative!(6, 0, -2),
            },
            _ => unimplemented!("Kernel is unimplemented for order {}", ORDER),
        }
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / spacing
    }
}

/// Second derivative operator of a given order.
#[derive(Clone)]
pub struct SecondDerivative<const ORDER: usize>;

impl<const ORDER: usize> Kernel for SecondDerivative<ORDER> {
    fn border_width(&self) -> usize {
        ORDER / 2
    }

    fn interior(&self) -> &[f64] {
        match ORDER {
            2 => &second_derivative!(1, 1, 0),
            4 => &second_derivative!(2, 2, 0),
            6 => &second_derivative!(3, 3, 0),
            _ => unimplemented!("Kernel is unimplemented for order {}", ORDER),
        }
    }

    fn free(&self, border: Border) -> &[f64] {
        match ORDER {
            2 => match border {
                Border::Negative(_) => &second_derivative!(0, 3, 0),
                Border::Positive(_) => &second_derivative!(3, 0, 0),
            },
            4 => match border {
                Border::Negative(0) => &second_derivative!(0, 5, 0),
                Border::Negative(_) => &second_derivative!(0, 5, 1),
                Border::Positive(0) => &second_derivative!(5, 0, 0),
                Border::Positive(_) => &second_derivative!(5, 0, -1),
            },
            6 => match border {
                Border::Negative(0) => &second_derivative!(0, 7, 0),
                Border::Negative(1) => &second_derivative!(0, 7, 1),
                Border::Negative(_) => &second_derivative!(0, 7, 2),
                Border::Positive(0) => &second_derivative!(7, 0, 0),
                Border::Positive(1) => &second_derivative!(7, 0, -1),
                Border::Positive(_) => &second_derivative!(7, 0, -2),
            },
            _ => unimplemented!("Kernel is unimplemented for order {}", ORDER),
        }
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

/// Kriss Olgier dissipation of the given order.
#[derive(Clone)]
pub struct Dissipation<const ORDER: usize>;

impl<const ORDER: usize> Kernel for Dissipation<ORDER> {
    fn border_width(&self) -> usize {
        ORDER / 2
    }

    fn interior(&self) -> &[f64] {
        match ORDER {
            4 => &[1.0, -4.0, 6.0, -4.0, 1.0],
            6 => &[1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0],
            _ => unimplemented!("Kernel is unimplemented for order {}", ORDER),
        }
    }

    fn free(&self, border: Border) -> &[f64] {
        match ORDER {
            4 => match border {
                Border::Negative(0) => &[3.0, -14.0, 26.0, -24.0, 11.0, -2.0],
                Border::Negative(_) => &[2.0, -9.0, 16.0, -14.0, 6.0, -1.0],
                Border::Positive(0) => &[-2.0, 11.0, -24.0, 26.0, -14.0, 3.0],
                Border::Positive(_) => &[-1.0, 6.0, -14.0, 16.0, -9.0, 2.0],
            },
            6 => match border {
                Border::Negative(0) => &[4.0, -27.0, 78.0, -125.0, 120.0, -69.0, 22.0, -3.0],
                Border::Negative(1) => &[3.0, -20.0, 57.0, -90.0, 85.0, -48.0, 15.0, -2.0],
                Border::Negative(_) => &[2.0, -13.0, 36.0, -55.0, 50.0, -27.0, 8.0, -1.0],
                Border::Positive(0) => &[-3.0, 22.0, -69.0, 120.0, -125.0, 78.0, -27.0, 4.0],
                Border::Positive(1) => &[-2.0, 15.0, -48.0, 85.0, -90.0, 57.0, -20.0, 3.0],
                Border::Positive(_) => &[-1.0, 8.0, -27.0, 50.0, -55.0, 36.0, -13.0, 2.0],
            },
            _ => unimplemented!("Kernel is unimplemented for order {}", ORDER),
        }
    }

    fn scale(&self, _spacing: f64) -> f64 {
        match ORDER {
            4 => -1.0 / 16.0,
            6 => 1.0 / 64.0,
            _ => unimplemented!("Kernel is unimplemented for order {}", ORDER),
        }
    }
}

#[derive(Clone)]
pub struct Interpolation<const ORDER: usize>;

impl<const ORDER: usize> Interpolant for Interpolation<ORDER> {
    fn border_width(&self) -> usize {
        ORDER / 2
    }

    fn interior(&self) -> &[f64] {
        match ORDER {
            2 => &[-1.0, 9.0, 9.0, -1.0],
            4 => &[3.0, -25.0, 150.0, 150.0, -25.0, 3.0],
            _ => unimplemented!("Kernel is unimplemented for order {}", ORDER),
        }
    }

    fn free(&self, border: Border) -> &[f64] {
        match ORDER {
            2 => match border {
                Border::Positive(_) => &[1.0, -5.0, 15.0, 5.0],
                Border::Negative(_) => &[5.0, 15.0, -5.0, 1.0],
            },
            4 => match border {
                Border::Negative(0) => &[63.0, 315.0, -210.0, 126.0, -45.0, 7.0],
                Border::Negative(_) => &[-7.0, 105.0, 210.0, -70.0, 21.0, -3.0],
                Border::Positive(0) => &[7.0, -45.0, 126.0, -210.0, 315.0, 63.0],
                Border::Positive(_) => &[-3.0, 21.0, -70.0, 210.0, 105.0, -7.0],
            },
            _ => unimplemented!("Kernel is unimplemented for order {}", ORDER),
        }
    }

    fn scale(&self) -> f64 {
        match ORDER {
            2 => 1.0 / 16.0,
            4 => 1.0 / 256.0,
            _ => unimplemented!("Kernel is unimplemented for order {}", ORDER),
        }
    }
}
