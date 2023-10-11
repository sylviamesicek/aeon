const std = @import("std");

/// Builds a node centered grid, with one central point, L points on the left,
/// and R points on the right, each separated by 1 unit of distance.
pub fn nodeCenteredGrid(comptime T: type, comptime L: usize, comptime R: usize) [L + R + 1]T {
    var grid: [L + R + 1]T = undefined;

    for (0..(L + R + 1)) |i| {
        grid[i] = @as(T, @floatFromInt(i)) - @as(T, @floatFromInt(L));
    }

    return grid;
}

/// Builds a vertex centered grid, with L points on the left and R points on the right,
/// each separated by 1 unit of distance.
pub fn vertexCenteredGrid(comptime T: type, comptime L: usize, comptime R: usize) [L + R]T {
    var grid: [L + R]T = undefined;

    for (0..(L + R)) |i| {
        grid[i] = @as(T, @floatFromInt(i)) + @as(T, 0.5) - @as(T, @floatFromInt(L));
    }

    return grid;
}

test "basis grids" {
    const expectEqualSlices = std.testing.expectEqualSlices;

    const ngrid = nodeCenteredGrid(f64, 1, 2);
    const vgrid = vertexCenteredGrid(f64, 2, 1);

    try expectEqualSlices(f64, &[_]f64{ -1.0, 0.0, 1.0, 2.0 }, &ngrid);
    try expectEqualSlices(f64, &[_]f64{ -1.5, -0.5, 0.5 }, &vgrid);
}
