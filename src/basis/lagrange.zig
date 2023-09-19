const std = @import("std");

/// Computes the interpolation stencil given a grid and a point.
pub fn valueStencil(comptime T: type, comptime L: usize, grid: [L]T, point: T) [L]T {
    var stencil: [L]T = undefined;

    for (0..L) |i| {
        stencil[i] = @as(T, 1);

        for (0..L) |j| {
            if (i != j) {
                stencil[i] *= (point - grid[j]) / (grid[i] - grid[j]);
            }
        }
    }

    return stencil;
}

/// Computes the derivative stencil given a grid and a point.
pub fn derivativeStencil(comptime T: type, comptime L: usize, grid: [L]T, point: T) [L]T {
    var stencil: [L]T = undefined;

    for (0..L) |i| {
        stencil[i] = @as(T, 0);

        for (0..L) |j| {
            if (i != j) {
                var result = @as(T, 1);

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
pub fn secondDerivativeStencil(comptime T: type, comptime L: usize, grid: [L]T, point: T) [L]T {
    var stencil: [L]T = undefined;

    for (0..L) |i| {
        stencil[i] = @as(T, 0);

        for (0..L) |j| {
            if (i != j) {
                var result1 = @as(T, 0);

                for (0..L) |k| {
                    if (k != i and k != j) {
                        var result2 = @as(T, 1);

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
