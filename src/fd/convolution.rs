use crate::fd::kernel::{Border, Kernel, Kernels, Value, VertexKernel};
use std::marker::PhantomData;

// ************************************
// Convolution ************************
// ************************************

/// A N-dimensional tensor product of several seperable kernels.
pub trait Convolution<const N: usize> {
    fn border_width(&self, axis: usize) -> usize;

    fn interior(&self, axis: usize) -> &[f64];
    fn free(&self, border: Border, axis: usize) -> &[f64];
    fn symmetric(&self, border: Border, axis: usize) -> &[f64];
    fn antisymmetric(&self, border: Border, axis: usize) -> &[f64];

    fn scale(&self, spacing: [f64; N]) -> f64;
}

macro_rules! impl_convolution_for_tuples {
    ($($N:literal => $($T:ident $i:tt)*)*) => {
        $(
            impl<$($T: VertexKernel,)*> Convolution<$N> for ($($T,)*) {
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

                fn symmetric(&self, border: Border, axis: usize) -> &[f64] {
                    match axis {
                        $($i => self.$i.symmetric(border),)*
                        _ => panic!("Invalid Axis")
                    }
                }


                fn antisymmetric(&self, border: Border, axis: usize) -> &[f64] {
                    match axis {
                        $($i => self.$i.antisymmetric(border),)*
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
pub struct Gradient<O: Kernels>(usize, PhantomData<O>);

impl<O: Kernels> Gradient<O> {
    /// Constructs a convolution which computes the derivative along the given axis.
    pub const fn new(axis: usize) -> Self {
        Self(axis, PhantomData)
    }
}

impl<O: Kernels> Clone for Gradient<O> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<const N: usize, O: Kernels> Convolution<N> for Gradient<O> {
    fn border_width(&self, axis: usize) -> usize {
        if axis == self.0 {
            O::derivative().border_width()
        } else {
            Value.border_width()
        }
    }

    fn interior(&self, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::derivative().interior()
        } else {
            Value.interior()
        }
    }

    fn free(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::derivative().free(border)
        } else {
            Value.free(border)
        }
    }

    fn symmetric(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::derivative().symmetric(border)
        } else {
            Value.symmetric(border)
        }
    }

    fn antisymmetric(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::derivative().antisymmetric(border)
        } else {
            Value.antisymmetric(border)
        }
    }

    fn scale(&self, spacing: [f64; N]) -> f64 {
        1.0 / spacing[self.0]
    }
}

/// Computes the mixed derivative of the given axes.
pub struct Hessian<O: Kernels>(usize, usize, PhantomData<O>);

impl<O: Kernels> Hessian<O> {
    /// Constructs a convolution which computes the given entry of the hessian matrix.
    pub const fn new(i: usize, j: usize) -> Self {
        Self(i, j, PhantomData)
    }
}

impl<O: Kernels> Hessian<O> {
    fn is_second(&self, axis: usize) -> bool {
        self.0 == self.1 && axis == self.0
    }

    fn is_first(&self, axis: usize) -> bool {
        self.0 == axis || self.1 == axis
    }
}

impl<const N: usize, O: Kernels> Convolution<N> for Hessian<O> {
    fn border_width(&self, axis: usize) -> usize {
        if self.is_second(axis) {
            O::second_derivative().border_width()
        } else if self.is_first(axis) {
            O::derivative().border_width()
        } else {
            Value.border_width()
        }
    }

    fn interior(&self, axis: usize) -> &[f64] {
        if self.is_second(axis) {
            O::second_derivative().interior()
        } else if self.is_first(axis) {
            O::derivative().interior()
        } else {
            Value.interior()
        }
    }

    fn free(&self, border: Border, axis: usize) -> &[f64] {
        if self.is_second(axis) {
            O::second_derivative().free(border)
        } else if self.is_first(axis) {
            O::derivative().free(border)
        } else {
            Value.free(border)
        }
    }

    fn symmetric(&self, border: Border, axis: usize) -> &[f64] {
        if self.is_second(axis) {
            O::second_derivative().symmetric(border)
        } else if self.is_first(axis) {
            O::derivative().symmetric(border)
        } else {
            Value.symmetric(border)
        }
    }

    fn antisymmetric(&self, border: Border, axis: usize) -> &[f64] {
        if self.is_second(axis) {
            O::second_derivative().antisymmetric(border)
        } else if self.is_first(axis) {
            O::derivative().antisymmetric(border)
        } else {
            Value.antisymmetric(border)
        }
    }

    fn scale(&self, spacing: [f64; N]) -> f64 {
        1.0 / (spacing[self.0] * spacing[self.1])
    }
}

/// Computes the gradient along the given axis.
pub struct Dissipation<O: Kernels>(usize, PhantomData<O>);

impl<O: Kernels> Dissipation<O> {
    /// Constructs a convolution which computes the derivative along the given axis.
    pub const fn new(axis: usize) -> Self {
        Self(axis, PhantomData)
    }
}

impl<O: Kernels> Clone for Dissipation<O> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<const N: usize, O: Kernels> Convolution<N> for Dissipation<O> {
    fn border_width(&self, axis: usize) -> usize {
        if axis == self.0 {
            O::dissipation().border_width()
        } else {
            Value.border_width()
        }
    }

    fn interior(&self, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::dissipation().interior()
        } else {
            Value.interior()
        }
    }

    fn free(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::dissipation().free(border)
        } else {
            Value.free(border)
        }
    }

    fn symmetric(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::dissipation().symmetric(border)
        } else {
            Value.symmetric(border)
        }
    }

    fn antisymmetric(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::dissipation().antisymmetric(border)
        } else {
            Value.antisymmetric(border)
        }
    }

    fn scale(&self, spacing: [f64; N]) -> f64 {
        O::dissipation().scale(spacing[self.0])
    }
}
