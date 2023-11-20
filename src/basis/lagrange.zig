const std = @import("std");
const assert = std.debug.assert;

/// Provides routines for computing langrange polynomial based stencils for interpolation
/// and higher order derivatives.
pub fn Lagrange(comptime Real: type) type {
    return struct {
        /// This builds a stencil that for a given grid will interpolate the value of a function
        /// at a given point. The `grid` and `stencil` arguments must have the same length.
        /// The contents of `stencil` are overriden with the new stencil values.
        pub fn value(grid: []const Real, point: Real, stencil: []Real) void {
            assert(grid.len == stencil.len);

            for (0..grid.len) |i| {
                stencil[i] = @as(Real, 1.0);

                for (0..grid.len) |j| {
                    if (i != j) {
                        stencil[i] *= (point - grid[j]) / (grid[i] - grid[j]);
                    }
                }
            }
        }

        /// This builds a stencil that for a given grid will calculate the derivative of a function
        /// at a given point. The `grid` and `stencil` arguments must have the same length.
        /// The contents of `stencil` are overriden with the new stencil values.
        pub fn derivative(grid: []const Real, point: Real, stencil: []Real) void {
            assert(grid.len == stencil.len);

            for (0..grid.len) |i| {
                stencil[i] = @as(Real, 0.0);

                for (0..grid.len) |j| {
                    if (i != j) {
                        var result: Real = @as(Real, 1.0);

                        for (0..grid.len) |k| {
                            if (i != k and j != k) {
                                result *= (point - grid[k]) / (grid[i] - grid[k]);
                            }
                        }

                        stencil[i] += result / (grid[i] - grid[j]);
                    }
                }
            }
        }

        /// This builds a stencil that for a given grid will calculate the second derivative of a function
        /// at a given point. The `grid` and `stencil` arguments must have the same length.
        /// The contents of `stencil` are overriden with the new stencil values.
        pub fn secondDerivative(grid: []const Real, point: Real, stencil: []Real) void {
            assert(grid.len == stencil.len);

            // This function generates many branches if called during comptime.
            @setEvalBranchQuota(10000);

            for (0..grid.len) |i| {
                stencil[i] = @as(Real, 0.0);

                for (0..grid.len) |j| {
                    if (i != j) {
                        var result1: Real = @as(Real, 0.0);

                        for (0..grid.len) |k| {
                            if (k != i and k != j) {
                                var result2: Real = @as(Real, 1.0);

                                for (0..grid.len) |l| {
                                    if (l != i and l != j and l != k) {
                                        result2 *= (point - grid[l]) / (grid[i] - grid[l]);
                                    }
                                }

                                result1 += result2 / (grid[i] - grid[k]);
                            }
                        }

                        stencil[i] += result1 / (grid[i] - grid[j]);
                    }
                }
            }
        }
    };
}

// /// Computes the interpolation stencil given a grid and a point.
// pub fn valueStencil(comptime L: usize, grid: [L]f64, point: f64) [L]f64 {
//     var stencil: [L]f64 = undefined;

//     for (0..L) |i| {
//         stencil[i] = 1.0;

//         for (0..L) |j| {
//             if (i != j) {
//                 stencil[i] *= (point - grid[j]) / (grid[i] - grid[j]);
//             }
//         }
//     }

//     return stencil;
// }

// /// Computes the derivative stencil given a grid and a point.
// pub fn derivativeStencil(comptime L: usize, grid: [L]f64, point: f64) [L]f64 {
//     var stencil: [L]f64 = undefined;

//     for (0..L) |i| {
//         stencil[i] = 0.0;

//         for (0..L) |j| {
//             if (i != j) {
//                 var result: f64 = 1.0;

//                 for (0..L) |k| {
//                     if (i != k and j != k) {
//                         result *= (point - grid[k]) / (grid[i] - grid[k]);
//                     }
//                 }

//                 stencil[i] += result / (grid[i] - grid[j]);
//             }
//         }
//     }

//     return stencil;
// }

// /// Computes the second derivative stencil given a grid and a point.
// pub fn secondDerivativeStencil(comptime L: usize, grid: [L]f64, point: f64) [L]f64 {
//     @setEvalBranchQuota(10000);
//     var stencil: [L]f64 = undefined;

//     for (0..L) |i| {
//         stencil[i] = 0.0;

//         for (0..L) |j| {
//             if (i != j) {
//                 var result1: f64 = 0.0;

//                 for (0..L) |k| {
//                     if (k != i and k != j) {
//                         var result2: f64 = 1.0;

//                         for (0..L) |l| {
//                             if (l != i and l != j and l != k) {
//                                 result2 *= (point - grid[l]) / (grid[i] - grid[l]);
//                             }
//                         }

//                         result1 += result2 / (grid[i] - grid[k]);
//                     }
//                 }

//                 stencil[i] += result1 / (grid[i] - grid[j]);
//             }
//         }
//     }

//     return stencil;
// }
