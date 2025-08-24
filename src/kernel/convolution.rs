use crate::kernel::{Border, Derivative, Kernel, SecondDerivative, Value};

// ************************************
// Convolution ************************
// ************************************

/// A N-dimensional tensor product of several seperable kernels.
pub trait Convolution<const N: usize> {
    fn border_width(&self, axis: usize) -> usize;
    fn interior(&self, axis: usize) -> &[f64];
    fn free(&self, border: Border, axis: usize) -> &[f64];
    fn scale(&self, spacing: [f64; N]) -> f64;
}

macro_rules! impl_convolution_for_tuples {
    ($($N:literal => $($T:ident $i:tt)*)*) => {
        $(
            impl<$($T: Kernel,)*> Convolution<$N> for ($($T,)*) {
                fn border_width(&self, axis: usize) -> usize {
                    match axis {
                        $($i => self.$i.border_width(),)*
                        _ => panic!("Invalid Axis")
                    }
                }

                fn interior(&self, axis: usize) -> &[f64] {
                    match axis {
                        $($i => self.$i.interior(),)*
                        _ => panic!("Invalid Axis")
                    }
                }

                fn free(&self, border: Border, axis: usize) -> &[f64] {
                    match axis {
                        $($i => self.$i.free(border),)*
                        _ => panic!("Invalid Axis")
                    }
                }

                fn scale(&self, spacing: [f64; $N]) -> f64 {
                    let mut result = 1.0;

                    $(
                        result *= self.$i.scale(spacing[$i]);
                    )*

                    result
                }
            }
        )*

    };
}

impl_convolution_for_tuples! {
   1 => K0 0
   2 => K0 0 K1 1
   3 => K0 0 K1 1 K2 2
   4 => K0 0 K1 1 K2 2 K3 3
}

/// Computes the gradient along the given axis.
#[derive(Clone)]
pub struct Gradient<const ORDER: usize>(pub usize);

impl<const N: usize, const ORDER: usize> Convolution<N> for Gradient<ORDER> {
    fn border_width(&self, axis: usize) -> usize {
        if axis == self.0 {
            Derivative::<ORDER>.border_width()
        } else {
            Value.border_width()
        }
    }

    fn interior(&self, axis: usize) -> &[f64] {
        if axis == self.0 {
            Derivative::<ORDER>.interior()
        } else {
            Value.interior()
        }
    }

    fn free(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            Derivative::<ORDER>.free(border)
        } else {
            Value.free(border)
        }
    }

    fn scale(&self, spacing: [f64; N]) -> f64 {
        1.0 / spacing[self.0]
    }
}

/// Computes the mixed derivative of the given axes.
#[derive(Debug, Clone, Copy)]
pub struct Hessian<const ORDER: usize>(pub usize, pub usize);

impl<const ORDER: usize> Hessian<ORDER> {
    /// Constructs a convolution which computes the given entry of the hessian matrix.
    pub const fn new(i: usize, j: usize) -> Self {
        Self(i, j)
    }
}

impl<const ORDER: usize> Hessian<ORDER> {
    fn is_second(&self, axis: usize) -> bool {
        self.0 == self.1 && axis == self.0
    }

    fn is_first(&self, axis: usize) -> bool {
        self.0 == axis || self.1 == axis
    }
}

impl<const N: usize, const ORDER: usize> Convolution<N> for Hessian<ORDER> {
    fn border_width(&self, axis: usize) -> usize {
        if self.is_second(axis) {
            SecondDerivative::<ORDER>.border_width()
        } else if self.is_first(axis) {
            Derivative::<ORDER>.border_width()
        } else {
            Value.border_width()
        }
    }

    fn interior(&self, axis: usize) -> &[f64] {
        if self.is_second(axis) {
            SecondDerivative::<ORDER>.interior()
        } else if self.is_first(axis) {
            Derivative::<ORDER>.interior()
        } else {
            Value.interior()
        }
    }

    fn free(&self, border: Border, axis: usize) -> &[f64] {
        if self.is_second(axis) {
            SecondDerivative::<ORDER>.free(border)
        } else if self.is_first(axis) {
            Derivative::<ORDER>.free(border)
        } else {
            Value.free(border)
        }
    }
    fn scale(&self, spacing: [f64; N]) -> f64 {
        1.0 / (spacing[self.0] * spacing[self.1])
    }
}
