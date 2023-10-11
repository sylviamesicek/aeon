const std = @import("std");

/// Computes the interpolation stencil given a grid and a point.
pub fn valueStencil(comptime L: usize, grid: [L]f64, point: f64) [L]f64 {
    var stencil: [L]f64 = undefined;

    for (0..L) |i| {
        stencil[i] = 1.0;

        for (0..L) |j| {
            if (i != j) {
                stencil[i] *= (point - grid[j]) / (grid[i] - grid[j]);
            }
        }
    }

    return stencil;
}

/// Computes the derivative stencil given a grid and a point.
pub fn derivativeStencil(comptime L: usize, grid: [L]f64, point: f64) [L]f64 {
    var stencil: [L]f64 = undefined;

    for (0..L) |i| {
        stencil[i] = 0.0;

        for (0..L) |j| {
            if (i != j) {
                var result: f64 = 1.0;

                for (0..L) |k| {
                    if (i != k and j != k) {
                        result *= (point - grid[k]) / (grid[i] - grid[k]);
                    }
                }

                stencil[i] += result / (grid[i] - grid[j]);
            }
        }
    }

    return stencil;
}

/// Computes the second derivative stencil given a grid and a point.
pub fn secondDerivativeStencil(comptime L: usize, grid: [L]f64, point: f64) [L]f64 {
    var stencil: [L]f64 = undefined;

    for (0..L) |i| {
        stencil[i] = 0.0;

        for (0..L) |j| {
            if (i != j) {
                var result1: f64 = 0.0;

                for (0..L) |k| {
                    if (k != i and k != j) {
                        var result2: f64 = 1.0;

                        for (0..L) |l| {
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

    return stencil;
}

// test "lagrange stencils" {
//     const expectEqualSlices = std.testing.expectEqualSlices;

//     const grid = [_]f64{ -0.5, 0.5 };

//     try expectEqualSlices(f64, &[_]f64{ 0.5, 0.5 }, &valueStencil(grid.len, grid, 0.0));
// }
