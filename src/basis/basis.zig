const std = @import("std");
const lagrange = @import("lagrange.zig");

pub const BasisType = enum { lagrange };
pub const OperatorType = enum {
    value,
    derivative,
    second_derivative,
};

/// A computes the stencil corresponding to the application of the given operator
/// with the given basis on a grid at a point.
pub fn stencil(comptime T: type, comptime L: usize, b_type: BasisType, comptime o_type: OperatorType, grid: [L]T, point: T) [L]T {
    switch (b_type) {
        .lagrange => switch (o_type) {
            .value => return lagrange.value_stencil(T, L, grid, point),
            .derivative => return lagrange.derivative_stencil(T, L, grid, point),
            .second_derivative => return lagrange.second_derivative_stencil(T, L, grid, point),
        },
    }
}

/// Builds a cell centered grid, with one central point, L points on the left,
/// and R points on the right.
pub fn cell_centered_grid(comptime T: type, comptime L: usize, comptime R: usize) [L + R + 1]T {
    var grid: [L + R + 1]T = undefined;

    for (0..(L + R + 1)) |i| {
        grid[i] = @as(T, @floatFromInt(i)) - @as(T, @floatFromInt(L));
    }

    return grid;
}

/// Builds a vertex centered grid, with L points on the left and R points on the right.
pub fn vertex_centered_grid(comptime T: type, comptime L: usize, comptime R: usize) [L + R]T {
    var grid: [L + R]T = undefined;

    for (0..(L + R)) |i| {
        grid[i] = @as(T, @floatFromInt(i)) + @as(T, 0.5) - @as(T, @floatFromInt(L));
    }

    return grid;
}

test "basis grids" {
    const expect = std.testing.expect;
    const eql = std.mem.eql;

    const cgrid = cell_centered_grid(f64, 1, 1);
    const vgrid = vertex_centered_grid(f64, 1, 1);

    try expect(eql(f64, &cgrid, &[_]f64{ -1.0, 0.0, 1.0 }));
    try expect(eql(f64, &vgrid, &[_]f64{ -0.5, 0.5 }));
}

test "basis stencils" {
    const expect = std.testing.expect;
    const eql = std.mem.eql;

    const grid = cell_centered_grid(f64, 1, 1);

    // Stencils
    const vstencil = stencil(f64, grid.len, .lagrange, .value, grid, 0.0);
    const dstencil = stencil(f64, grid.len, .lagrange, .derivative, grid, 0.0);
    const sdstencil = stencil(f64, grid.len, .lagrange, .second_derivative, grid, 0.0);
    try expect(eql(f64, &[_]f64{ 0.0, 1.0, 0.0 }, &vstencil));
    try expect(eql(f64, &[_]f64{ -0.5, 0.0, 0.5 }, &dstencil));
    try expect(eql(f64, &[_]f64{ 1.0, -2.0, 1.0 }, &sdstencil));
}
