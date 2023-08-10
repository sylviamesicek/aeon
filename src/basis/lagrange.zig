const std = @import("std");

fn basis_filtered(comptime T: type, grid: []const T, i: usize, filter: []const usize, point: T) T {
    var result = @as(T, 1);

    for (0..grid.len) |idx| {
        var matches = i == idx;

        for (filter) |f| {
            matches = matches or f == idx;
        }

        if (!matches) {
            result *= (point - grid[idx]) / (grid[i] - grid[idx]);
        }
    }

    return result;
}

/// Computes the interpolation stencil given a grid and a point.
pub fn value_stencil(comptime T: type, comptime L: usize, grid: [L]T, point: T) [L]T {
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
pub fn derivative_stencil(comptime T: type, comptime L: usize, grid: [L]T, point: T) [L]T {
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

pub fn second_derivative_stencil(comptime T: type, comptime L: usize, grid: [L]T, point: T) [L]T {
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

fn cell_grid(comptime T: type, comptime L: usize, comptime R: usize) [L + R + 1]T {
    var grid: [L + R + 1]T = undefined;

    for (0..(L + R + 1)) |i| {
        const signed: isize = @intCast(i);
        grid[i] = @floatFromInt(signed - L);
    }

    return grid;
}
