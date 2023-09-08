const std = @import("std");
const pow = std.math.pow;
const assert = std.debug.assert;

const lagrange = @import("lagrange.zig");

const geometry = @import("../geometry/geometry.zig");
const mesh = @import("../mesh/mesh.zig");

const BoundaryCondition = mesh.BoundaryCondition;

/// Manages the application of stencil products on functions. Supports computing values, centered derivatives
/// positions, boundary positions, boundary values, boundary derivatives, prolongation, and restriction.
/// All cell indices are in standard index space (ie without ghost cells included).
pub fn StencilSpace(comptime N: usize, comptime O: usize) type {
    return struct {
        physical_bounds: RealBox,
        size: [N]usize,

        const Self = @This();
        const RealBox = geometry.Box(N, f64);
        const IndexBox = geometry.Box(N, usize);
        const IndexSpace = geometry.IndexSpace(N);
        const Region = geometry.Region(N, O);
        const Face = geometry.Face(N);

        // Gets position of cell.
        pub fn position(self: Self, cell: [N]usize) [N]f64 {
            var result: [N]f64 = undefined;

            for (0..N) |i| {
                const origin: f64 = self.physical_bounds.origin[i];
                const width: f64 = self.physical_bounds.size[i];
                const ratio: f64 = (@as(f64, @floatFromInt(cell[i])) + 0.5) / @as(f64, @floatFromInt(self.size[i]));
                result[i] = origin + width * ratio;
            }

            return result;
        }

        // Gets boundary position of cell.
        pub fn boundaryPosition(self: Self, comptime extents: [N]isize, cell: [N]usize) [N]f64 {
            var result = self.position(cell);

            for (0..N) |i| {
                if (extents[i] > 0) {
                    result[i] = self.physical_bounds.origin[i];
                } else if (extents[i] < 0) {
                    result[i] = self.physical_bounds.origin[i] + self.physical_bounds.size[i];
                }
            }

            return result;
        }

        /// Computes the value of a field at a cell.
        pub fn value(self: Self, cell: [N]usize, field: []const f64) f64 {
            const space = IndexSpace.fromSize(sizeWithGhost(self.size));
            const linear = space.linearFromCartesian(cellWithGhost(cell));
            return field[linear];
        }

        pub fn setValue(self: Self, cell: [N]usize, field: []f64, v: f64) void {
            const space = IndexSpace.fromSize(sizeWithGhost(self.size));
            const linear = space.linearFromCartesian(cellWithGhost(cell));
            field[linear] = v;
        }

        /// Computes the diagonal coefficient of the value stencil.
        pub fn valueDiagonal(_: Self) f64 {
            return 1.0;
        }

        /// Computes the derivative of a field at a cell.
        pub fn derivative(self: Self, comptime ranks: [N]usize, cell: [N]usize, field: []const f64) f64 {
            comptime var stencil_sizes: [N]usize = undefined;

            inline for (0..N) |i| {
                stencil_sizes[i] = if (ranks[i] == 0) 1 else 2 * O + 1;
            }

            comptime var stencils: [N][2 * O + 1]f64 = undefined;

            inline for (0..N) |i| {
                stencils[i] = derivativeStencil(ranks[i], O);
            }

            const stencil_space: IndexSpace = comptime IndexSpace.fromSize(stencil_sizes);
            const space: IndexSpace = IndexSpace.fromSize(sizeWithGhost(self.size));

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (stencil_indices.next()) |stencil_index| {
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    if (ranks[i] != 0) {
                        coef *= stencils[i][stencil_index[i]];
                    }
                }

                var offset_cell: [N]usize = undefined;

                inline for (0..N) |i| {
                    // This actually has an additional term -O (to correctly offset from center) +O (to account for ghost nodes).
                    offset_cell[i] = cell[i] + stencil_index[i];
                }

                const linear = space.linearFromCartesian(offset_cell);

                result += coef * field[linear];
            }

            // Covariantly transform result
            inline for (0..N) |i| {
                var scale: f64 = @floatFromInt(self.size[i]);
                scale /= self.physical_bounds.size[i];

                inline for (0..ranks[i]) |_| {
                    result *= scale;
                }
            }

            return result;
        }

        /// Computes the diagonal coefficient of the derivative stencil.
        pub fn derivativeDiagonal(self: Self, comptime ranks: [N]usize) f64 {
            comptime var stencils: [N][2 * O + 1]f64 = undefined;

            inline for (0..N) |i| {
                stencils[i] = derivativeStencil(ranks[i], O);
            }

            var result: f64 = 1.0;

            for (0..N) |i| {
                if (ranks[i] > 0) {
                    result *= stencils[O];
                }
            }

            // Covariantly transform result
            inline for (0..N) |i| {
                var scale: f64 = @floatFromInt(self.size[i]);
                scale /= self.physical_bounds.size[i];

                inline for (0..ranks[i]) |_| {
                    result *= scale;
                }
            }

            return result;
        }

        /// Computes the value at a boundary of a field.
        pub fn boundaryValue(self: Self, comptime extents: [N]isize, cell: [N]usize, field: []const f64) f64 {
            return self.boundaryDerivative([1]usize{0} ** N, extents, cell, field);
        }

        /// Computes the outmost coefficient of a boundary stencil.
        pub fn boundaryValueCoef(self: Self, comptime extents: [N]isize) f64 {
            return self.boundaryDerivativeCoef([1]usize{0} ** N, extents);
        }

        /// Computes the derivative at a bounday of a field.
        pub fn boundaryDerivative(self: Self, comptime ranks: [N]usize, comptime extents: [N]isize, cell: [N]usize, field: []const f64) f64 {
            comptime var stencil_sizes: [N]usize = undefined;

            inline for (0..N) |i| {
                stencil_sizes[i] = if (ranks[i] == 0) 1 else if (extents[i] == 0) 1 else 2 * O + 1 + absSigned(extents[i]);
            }

            comptime var stencils: [N][3 * O + 1]f64 = undefined;

            inline for (0..N) |i| {
                stencils[i] = boundaryDerivativeStencil(
                    ranks[i],
                    extents[i],
                    O,
                );
            }

            const stencil_space: IndexSpace = comptime IndexSpace.fromSize(stencil_sizes);
            const space: IndexSpace = IndexSpace.fromSize(sizeWithGhost(self.size));

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (stencil_indices.next()) |stencil_index| {
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    if (ranks[i] != 0 and extents[i] != 0) {
                        coef *= stencils[i][stencil_index[i]];
                    }
                }

                var offset_cell: [N]usize = undefined;

                inline for (0..N) |i| {
                    if (extents[i] > 0) {
                        offset_cell[i] = cell[i] + stencil_index[i] - 2 * O - 1 + O;
                    } else if (extents[i] < 0) {
                        offset_cell[i] = cell[i] + stencil_index[i] - absSigned(extents[i]) + O;
                    } else {
                        offset_cell[i] = cell[i] + O;
                    }
                }

                const linear = space.linearFromCartesian(offset_cell);

                result += coef * field[linear];
            }

            // Covariantly transform result
            inline for (0..N) |i| {
                var scale: f64 = @floatFromInt(self.size[i]);
                scale /= self.physical_bounds.size[i];

                inline for (0..ranks[i]) |_| {
                    result *= scale;
                }
            }

            return result;
        }

        /// Computes the outmost coefficient of a boundary derivative stencil.
        pub fn boundaryDerivativeCoef(self: Self, comptime ranks: [N]usize, comptime extents: [N]isize) f64 {
            comptime var stencils: [N][3 * O + 1]f64 = undefined;

            inline for (0..N) |i| {
                stencils[i] = boundaryDerivativeStencil(ranks[i], extents[i], O);
            }

            var result: f64 = 1.0;

            for (0..N) |i| {
                if (ranks[i] > 0) {
                    if (extents[i] > 0) {
                        result *= stencils[i][2 * O + absSigned(extents[i])];
                    } else {
                        result *= stencils[i][0];
                    }
                }
            }

            // Covariantly transform result
            inline for (0..N) |i| {
                var scale: f64 = @floatFromInt(self.size[i]);
                scale /= self.physical_bounds.size[i];

                inline for (0..ranks[i]) |_| {
                    result *= scale;
                }
            }

            return result;
        }

        /// Prolongs the value of a field to a subcell.
        pub fn prolong(self: Self, subcell: [N]usize, field: []const f64) f64 {
            comptime var lstencils: [N][2 * O + 1]f64 = undefined;
            comptime var rstencils: [N][2 * O + 1]f64 = undefined;

            inline for (0..N) |i| {
                lstencils[i] = prolongStencil(false, O);
                rstencils[i] = prolongStencil(true, O);
            }

            const stencil_space: IndexSpace = comptime IndexSpace.fromSize([1]usize{2 * O + 1} ** N);
            const space: IndexSpace = IndexSpace.fromSize(sizeWithGhost(self.size));

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (stencil_indices.next()) |stencil_index| {
                var coef: f64 = 1.0;

                for (0..N) |i| {
                    if (subcell[i] % 2 == 0) {
                        coef *= rstencils[i][stencil_index[i]];
                    } else {
                        coef *= lstencils[i][stencil_index[i]];
                    }
                }

                var offset_cell: [N]usize = undefined;

                inline for (0..N) |i| {
                    // This actually has an additional term -O (to correctly offset from center) +O (to account for ghost nodes).
                    offset_cell[i] = @divTrunc(subcell[i] + 1, 2) + stencil_index[i];
                }

                const linear = space.linearFromCartesian(offset_cell);

                result += coef * field[linear];
            }

            return result;
        }

        /// Restricts the value of a field to a supercell.
        pub fn restrict(self: Self, supercell: [N]usize, field: []const f64) f64 {
            const stencils: [N][2 * O + 1]f64 = [1][2 * O + 2]f64{restrictStencil(O)} ** N;

            const stencil_space: IndexSpace = comptime IndexSpace.fromSize([1]usize{2 * O + 2} ** N);

            const space: IndexSpace = IndexSpace.fromSize(sizeWithGhost(self.size));

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (stencil_indices.next()) |stencil_index| {
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    coef *= stencils[i][stencil_index[i]];
                }

                var offset_cell: [N]usize = undefined;

                inline for (0..N) |i| {
                    // This actually has an additional term -O (to correctly offset from center) +O (to account for ghost nodes).
                    offset_cell[i] = 2 * supercell[i] + stencil_index[i] - 1;
                }

                const linear = space.linearFromCartesian(offset_cell);

                result += coef * field[linear];
            }

            return result;
        }

        fn cellWithGhost(cell: [N]usize) [N]usize {
            var result: [N]usize = cell;
            for (0..N) |i| {
                result[i] + O;
            }
            return result;
        }

        fn sizeWithGhost(size: [N]usize) [N]usize {
            var result: [N]usize = size;
            for (0..N) |i| {
                result[i] + 2 * O;
            }
            return result;
        }

        pub fn fillBoundary(self: Self, comptime region: Region(N), boundary: anytype, field: []f64) void {
            var inner_face_cells = region.innerFaceIndices(O, self.index_size);

            while (inner_face_cells.next()) |inner_face_cell| {
                var cell: [N]usize = inner_face_cell;

                for (0..N) |i| {
                    cell[i] -= O;
                }

                comptime var extent_indices = region.extentOffsets();

                inline while (extent_indices) |extents| {
                    var target: [N]usize = cell;

                    for (0..N) |i| {
                        target[i] += extents[i];
                    }

                    self.setValue(target, field, 0.0);

                    const pos = self.boundaryPosition(extents, cell);
                    _ = pos;

                    var v: f64 = 0.0;
                    var normals: [N]usize = undefined;
                    var rhs: f64 = 0.0;

                    for (0..N) |i| {
                        if (extents[i] != 0) {
                            const condition: BoundaryCondition = boundary.condition(position, Face{
                                .side = extents[i] > 0,
                                .axis = i,
                            });

                            v += condition.value;
                            normals[i] = condition.normal;
                            rhs += condition.rhs;
                        }
                    }

                    var sum: f64 = v * self.boundaryValue(extents, cell, field);
                    var coef: f64 = v * self.boundaryValueCoef(extents);

                    inline for (0..N) |i| {
                        if (extents[i] != 0) {
                            var ranks: [N]usize = [1]usize{0} ** N;
                            ranks[i] = 1;

                            sum += normals[i] * self.boundaryDerivative(ranks, extents, cell, field);
                            coef += normals[i] * self.boundaryDerivativeCoef(ranks, extents);
                        }
                    }

                    self.setValue(target, field, (rhs - sum) / coef);
                }
            }
        }
    };
}

fn derivativeStencil(comptime R: usize, comptime O: usize) [2 * O + 1]f64 {
    const grid = cellCenteredGrid(f64, O, O);

    return switch (R) {
        0 => lagrange.valueStencil(f64, 2 * O + 1, grid, 0.0),
        1 => lagrange.derivativeStencil(f64, 2 * O + 1, grid, 0.0),
        2 => lagrange.secondDerivativeStencil(f64, 2 * O + 1, grid, 0.0),
        else => @compileError("Rank of derivative stencil must be <= 2"),
    };
}

fn absSigned(i: isize) usize {
    return @intCast(if (i < 0) -i else i);
}

fn boundaryDerivativeStencil(comptime R: usize, comptime extent: isize, comptime O: usize) [3 * O + 1]f64 {
    var result: [3 * O + 1]f64 = [1]f64{0} ** (3 * O + 1);

    if (extent <= 0) {
        const grid = vertexCenteredGrid(f64, absSigned(extent), 2 * O + 1);

        const stencil = switch (R) {
            0 => lagrange.valueStencil(f64, grid.len, grid, 0.0),
            1 => lagrange.derivativeStencil(f64, grid.len, grid, 0.0),
            2 => lagrange.secondDerivativeStencil(f64, grid.len, grid, 0.0),
            else => @compileError("Rank of boundary derivative stencil must be <= 2"),
        };

        for (stencil, 0..) |s, i| {
            result[i] = s;
        }
    } else {
        const grid = vertexCenteredGrid(f64, 2 * O + 1, absSigned(extent));

        const stencil = switch (R) {
            0 => lagrange.valueStencil(f64, grid.len, grid, 0.0),
            1 => lagrange.derivativeStencil(f64, grid.len, grid, 0.0),
            2 => lagrange.secondDerivativeStencil(f64, grid.len, grid, 0.0),
            else => @compileError("Rank of boundary derivative stencil must be <= 2"),
        };

        for (stencil, 0..) |s, i| {
            result[i] = s;
        }
    }

    return result;
}

fn prolongStencil(comptime side: bool, comptime O: usize) [2 * O + 1]f64 {
    const grid = cellCenteredGrid(f64, O, O);
    const point = if (side) 0.25 else -0.25;

    return lagrange.valueStencil(f64, 2 * O + 1, grid, point);
}

fn restrictStencil(comptime O: usize) [2 * O + 2]f64 {
    const grid = vertexCenteredGrid(f64, O + 1, O + 1);

    return lagrange.valueStencil(f64, 2 * O + 2, grid, 0.0);
}

/// Builds a cell centered grid, with one central point, L points on the left,
/// and R points on the right.
fn cellCenteredGrid(comptime T: type, comptime L: usize, comptime R: usize) [L + R + 1]T {
    var grid: [L + R + 1]T = undefined;

    for (0..(L + R + 1)) |i| {
        grid[i] = @as(T, @floatFromInt(i)) - @as(T, @floatFromInt(L));
    }

    return grid;
}

/// Builds a vertex centered grid, with L points on the left and R points on the right.
fn vertexCenteredGrid(comptime T: type, comptime L: usize, comptime R: usize) [L + R]T {
    var grid: [L + R]T = undefined;

    for (0..(L + R)) |i| {
        grid[i] = @as(T, @floatFromInt(i)) + @as(T, 0.5) - @as(T, @floatFromInt(L));
    }

    return grid;
}

test "basis grids" {
    const expect = std.testing.expect;
    const eql = std.mem.eql;

    const cgrid = cellCenteredGrid(f64, 1, 1);
    const vgrid = vertexCenteredGrid(f64, 1, 1);

    try expect(eql(f64, &cgrid, &[_]f64{ -1.0, 0.0, 1.0 }));
    try expect(eql(f64, &vgrid, &[_]f64{ -0.5, 0.5 }));
}

test "basis stencils" {
    const expectEqualSlices = std.testing.expectEqualSlices;

    // Stencils
    try expectEqualSlices(f64, &[_]f64{ 0.0, 1.0, 0.0 }, &derivativeStencil(0, 1));
    try expectEqualSlices(f64, &[_]f64{ -0.5, 0.0, 0.5 }, &derivativeStencil(1, 1));
    try expectEqualSlices(f64, &[_]f64{ 1.0, -2.0, 1.0 }, &derivativeStencil(2, 1));

    try expectEqualSlices(f64, &[_]f64{ 0.15625, 0.9375, -0.09375 }, &prolongStencil(false, 1));
    try expectEqualSlices(f64, &[_]f64{ -0.09375, 0.9375, 0.15625 }, &prolongStencil(true, 1));
    try expectEqualSlices(f64, &[_]f64{ -0.0625, 0.5625, 0.5625, -0.0625 }, &restrictStencil(1));
}
