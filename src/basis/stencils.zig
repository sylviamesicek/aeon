const std = @import("std");
const geometry = @import("../geometry/geometry.zig");

const lagrange = @import("lagrange.zig");

/// This namespace provides several functions for computing derivative stencils, and
/// centered prolongation and restriction stencils.
///
/// Here by convention, L refers to the number of equispaced support points to the left of the central node
/// and R refers to the number of rightwards support points. If a function takes in a single value of `M` the function
/// instead produces a centered stencil of order `2M`.
pub const Stencils = struct {
    const Lagrange = lagrange.Lagrange(f128);

    /// Produces a value stencil, which is always a centered delta function.
    pub fn value(comptime L: usize, comptime R: usize) [L + R + 1]f64 {
        var result: [L + R + 1]f128 = undefined;
        Lagrange.value(&grid(L, R), 0.0, &result);
        return floatCastSlice(L + R + 1, result);
    }

    /// Produces a derivative stencil.
    pub fn derivative(comptime L: usize, comptime R: usize) [L + R + 1]f64 {
        var result: [L + R + 1]f128 = undefined;
        Lagrange.derivative(&grid(L, R), 0.0, &result);
        return floatCastSlice(L + R + 1, result);
    }

    /// Produces a second derivative stencil.
    pub fn secondDerivative(comptime L: usize, comptime R: usize) [L + R + 1]f64 {
        var result: [L + R + 1]f128 = undefined;
        Lagrange.secondDerivative(&grid(L, R), 0.0, &result);
        return floatCastSlice(L + R + 1, result);
    }

    /// Produces an extrapolation stencil, which is always a centered delta function.
    pub fn extrapolate(comptime L: usize, comptime R: usize, off: isize) [L + R + 1]f64 {
        const point: f128 = @floatFromInt(off);

        var result: [L + R + 1]f128 = undefined;
        Lagrange.value(&grid(L, R), point, &result);
        return floatCastSlice(L + R + 1, result);
    }

    /// Produces a prolongation stencil.
    pub fn prolong(comptime M: usize) [2 * M]f64 {
        return switch (M) {
            0 => .{},
            1 => .{ 1.0 / 2.0, 1.0 / 2.0 },
            2 => .{ -1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0 },
            else => @compileError("Prolongation only supported for M <= 2"),
        };
    }

    /// Produces a full weighting restriction stencil.
    pub fn restrict(comptime M: usize) [2 * M + 1]f64 {
        return switch (M) {
            0 => .{1.0},
            1 => .{ 1.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0 },
            else => @compileError("Prolongation only supported for M < 2"),
        };
    }

    /// Computes the dissipation stencil.
    pub fn dissipation(comptime M: usize) [2 * M + 1]f64 {
        var scale: f64 = if (M % 2 == 0) -1.0 else 1.0;

        for (0..2 * M) |_| {
            scale /= 2.0;
        }

        var result: [2 * M + 1]f64 = switch (M) {
            0 => .{1.0},
            1 => .{ 1.0, -2.0, 1.0 },
            2 => .{ 1.0, -4.0, 6.0, -4.0, 1.0 },
            3 => .{ 1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0 },
            else => @compileError("Dissipation only supported for M < 4."),
        };

        for (0..result.len) |i| {
            result[i] *= scale;
        }

        return result;
    }

    /// Generates a centered grid consisting of points
    fn grid(comptime L: usize, comptime R: usize) [L + R + 1]f128 {
        var result: [L + R + 1]f128 = undefined;

        for (0..(L + R + 1)) |i| {
            result[i] = @as(f128, @floatFromInt(i)) - @as(f128, @floatFromInt(L));
        }

        return result;
    }

    /// Casts arrays of `[Len]f128` -> `[Len]f64` using the `@floatCast` builtin.
    fn floatCastSlice(comptime Len: usize, slice: [Len]f128) [Len]f64 {
        var result: [Len]f64 = undefined;

        for (0..Len) |i| {
            result[i] = @floatCast(slice[i]);
        }

        return result;
    }
};

test "stencils" {
    const expectEqualDeep = std.testing.expectEqualDeep;

    try expectEqualDeep([_]f128{ -1.0, 0.0, 1.0, 2.0 }, Stencils.grid(1, 2));

    try expectEqualDeep([_]f64{ 0.0, 1.0, 0.0 }, Stencils.value(1, 1));
    try expectEqualDeep([_]f64{ -1.0 / 2.0, 0.0, 1.0 / 2.0 }, Stencils.derivative(1, 1));
    try expectEqualDeep([_]f64{ 1.0, -2.0, 1.0 }, Stencils.secondDerivative(1, 1));

    try expectEqualDeep([_]f64{ 0.0, -0.0, 1.0, 0.0, -0.0 }, Stencils.value(2, 2));
    try expectEqualDeep([_]f64{ 1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0 }, Stencils.derivative(2, 2));
    try expectEqualDeep([_]f64{ -1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0 }, Stencils.secondDerivative(2, 2));
}
