const std = @import("std");

// Imported modules

const lagrange = @import("lagrange.zig");
const geometry = @import("../geometry/geometry.zig");

/// Provides an interface for interpolating and setting values on a block of a certain size
/// with a buffer region `2*O` along each axis, to the order `O`.
pub fn CellSpace(comptime N: usize, comptime O: usize) type {
    return struct {
        size: [N]usize,

        const Self = @This();
        const IndexSpace = geometry.IndexSpace(N);

        pub fn fromSize(size: [N]usize) Self {
            return .{ .size = size };
        }

        pub fn sizeWithGhost(self: Self) [N]usize {
            var result: [N]usize = undefined;

            for (0..N) |i| {
                result[i] = self.size[i] + 4 * O;
            }

            return result;
        }

        pub fn indexSpace(self: Self) IndexSpace {
            return IndexSpace.fromSize(self.sizeWithGhost());
        }

        pub fn indexFromCell(cell: [N]isize) [N]usize {
            var index: [N]usize = undefined;

            for (0..N) |i| {
                index[i] = @intCast(2 * @as(isize, @intCast(O)) + cell[i]);
            }

            return index;
        }

        pub fn offsetFromOrigin(origin: [N]usize, offset: [N]isize) [N]isize {
            var result: [N]isize = undefined;

            for (0..N) |i| {
                result[i] = @as(isize, @intCast(origin[i])) + offset[i];
            }

            return result;
        }

        /// Computes the value of a field at a cell.
        pub fn value(self: Self, cell: [N]isize, field: []const f64) f64 {
            const linear = self.indexSpace().linearFromCartesian(indexFromCell(cell));
            return field[linear];
        }

        /// Sets the value of a field at a cell.
        pub fn setValue(self: Self, cell: [N]isize, field: []f64, v: f64) void {
            const linear = self.indexSpace().linearFromCartesian(indexFromCell(cell));
            field[linear] = v;
        }

        /// Prolongs the value of a field to a subcell.
        pub fn prolong(self: Self, subcell: [N]isize, field: []const f64) f64 {
            // Build stencils for both the left and right case at comptime.
            const lstencil: [2 * O + 1]f64 = prolong(false, O);
            const rstencil: [2 * O + 1]f64 = prolong(false, O);

            const stencil_space: IndexSpace = IndexSpace.fromSize([1]usize{2 * O + 1} ** N);
            const index_space: IndexSpace = self.indexSpace();

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (stencil_indices.next()) |stencil_index| {
                var coef: f64 = 1.0;

                for (0..N) |i| {
                    if (subcell[i] % 2 == 0) {
                        coef *= rstencil[stencil_index[i]];
                    } else {
                        coef *= lstencil[stencil_index[i]];
                    }
                }

                var offset_cell: [N]usize = undefined;

                inline for (0..N) |i| {
                    offset_cell[i] = @divTrunc(subcell[i] + 1, 2) + stencil_index[i] - O;
                }

                const linear = index_space.linearFromCartesian(indexFromCell(offset_cell));

                result += coef * field[linear];
            }

            return result;
        }

        /// Restricts the value of a field to a supercell.
        pub fn restrict(self: Self, supercell: [N]usize, field: []const f64) f64 {
            const stencil: [2 * O + 2]f64 = restrictStencil(O);

            const stencil_space: IndexSpace = IndexSpace.fromSize([1]usize{2 * O + 2} ** N);
            const index_space: IndexSpace = self.indexSpace();

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (stencil_indices.next()) |stencil_index| {
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    coef *= stencil[stencil_index[i]];
                }

                var offset_cell: [N]usize = undefined;

                inline for (0..N) |i| {
                    offset_cell[i] = 2 * supercell[i] + stencil_index[i] - O - 1;
                }

                const linear = index_space.linearFromCartesian(indexFromCell(offset_cell));

                result += coef * field[linear];
            }

            return result;
        }

        pub fn CellIterator(comptime E: usize) type {
            return struct {
                inner: IndexSpace.CartesianIterator,

                pub fn init(size: [N]usize) @This() {
                    var index_size: [N]usize = size;

                    for (0..N) |i| {
                        index_size[i] += 2 * E;
                    }

                    return .{ .inner = IndexSpace.fromSize(index_size).cartesianIndices() };
                }

                pub fn next(self: *@This()) ?[N]isize {
                    const index = self.inner.next() orelse return null;

                    var result: [N]isize = undefined;

                    for (0..N) |i| {
                        result[i] = @as(isize, @intCast(index[i])) - E;
                    }

                    return result;
                }
            };
        }

        pub fn cellsWtihExtent(self: Self, comptime E: usize) CellIterator(E) {
            return CellIterator(E).init(self.size);
        }

        pub fn cells(self: Self) CellIterator(0) {
            return self.cellsWtihExtent(0);
        }

        pub fn fullCells(self: Self) CellIterator(O) {
            return self.cellsWtihExtent(O);
        }
    };
}

/// Manages the application of stencil products on functions. Supports computing values, centered derivatives
/// positions, boundary positions, boundary values, boundary derivatives, prolongation, and restriction.
/// If full is false, all cell indices are in standard index space (ie without ghost cells included).
pub fn StencilSpace(comptime N: usize, comptime O: usize) type {
    return struct {
        physical_bounds: RealBox,
        size: [N]usize,

        const Self = @This();
        const RealBox = geometry.Box(N, f64);
        const IndexBox = geometry.Box(N, usize);
        const IndexSpace = geometry.IndexSpace(N);
        const Region = geometry.Region(N);
        const Face = geometry.Face(N);
        const CSpace = CellSpace(N, O);

        pub fn cellSpace(self: Self) CSpace {
            return CSpace.fromSize(self.size);
        }

        /// Return the position of the given cell.
        pub fn position(self: Self, cell: [N]isize) [N]f64 {
            var result: [N]f64 = undefined;

            for (0..N) |i| {
                const origin: f64 = self.physical_bounds.origin[i];
                const width: f64 = self.physical_bounds.size[i];
                const ratio: f64 = (@as(f64, @floatFromInt(cell[i])) + 0.5) / @as(f64, @floatFromInt(self.size[i]));
                result[i] = origin + width * ratio;
            }

            return result;
        }

        /// Returns the position of the given vertex.
        pub fn vertexPosition(self: Self, vertex: [N]isize) [N]f64 {
            var result: [N]f64 = undefined;

            for (0..N) |i| {
                const origin: f64 = self.physical_bounds.origin[i];
                const width: f64 = self.physical_bounds.size[i];
                const ratio: f64 = (@as(f64, @floatFromInt(vertex[i]))) / @as(f64, @floatFromInt(self.size[i]));
                result[i] = origin + width * ratio;
            }

            return result;
        }

        /// Computes the value of a field at a cell.
        pub fn value(self: Self, cell: [N]isize, field: []const f64) f64 {
            return CSpace.fromSize(self.size).value(cell, field);
        }

        /// Computes the diagonal coefficient of the value stencil.
        pub fn valueDiagonal(_: Self) f64 {
            return 1.0;
        }

        /// Computes the derivative of a field at a cell.
        pub fn derivative(self: Self, comptime ranks: [N]usize, cell: [N]isize, field: []const f64) f64 {
            comptime var stencil_sizes: [N]usize = undefined;

            inline for (0..N) |i| {
                stencil_sizes[i] = if (ranks[i] == 0) 1 else 2 * O + 1;
            }

            comptime var stencils: [N][2 * O + 1]f64 = undefined;

            inline for (0..N) |i| {
                stencils[i] = comptime derivativeStencil(ranks[i], O);
            }

            const stencil_space: IndexSpace = comptime IndexSpace.fromSize(stencil_sizes);
            const index_space: IndexSpace = CSpace.fromSize(self.size).indexSpace();

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (comptime stencil_indices.next()) |stencil_index| {
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    if (ranks[i] != 0) {
                        coef *= stencils[i][stencil_index[i]];
                    }
                }

                var offset_cell: [N]isize = undefined;

                inline for (0..N) |i| {
                    // This actually has an additional term -O (to correctly offset from center).
                    offset_cell[i] = cell[i] + stencil_index[i] - O;
                }

                const linear = index_space.linearFromCartesian(CSpace.indexFromCell(offset_cell));

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

        /// Returns the boundary position of the given cell, taking into account the extents.
        /// This cell is always given in standard index space.
        pub fn boundaryPosition(self: Self, comptime extents: [N]isize, cell: [N]isize) [N]f64 {
            var result: [N]f64 = undefined;

            for (0..N) |i| {
                if (extents[i] < 0) {
                    result[i] = self.physical_bounds.origin[i];
                } else if (extents[i] > 0) {
                    result[i] = self.physical_bounds.origin[i] + self.physical_bounds.size[i];
                } else {
                    const origin: f64 = self.physical_bounds.origin[i];
                    const width: f64 = self.physical_bounds.size[i];
                    const ratio: f64 = (@as(f64, @floatFromInt(cell[i])) + 0.5) / @as(f64, @floatFromInt(self.size[i]));
                    result[i] = origin + width * ratio;
                }
            }

            return result;
        }

        /// Computes the value at a boundary of a field.
        pub fn boundaryValue(self: Self, comptime extents: [N]isize, cell: [N]isize, field: []const f64) f64 {
            return self.boundaryDerivative([1]usize{0} ** N, extents, cell, field);
        }

        /// Computes the outmost coefficient of a boundary stencil.
        pub fn boundaryValueCoef(self: Self, comptime extents: [N]isize) f64 {
            return self.boundaryDerivativeCoef([1]usize{0} ** N, extents);
        }

        /// Computes the derivative at a bounday of a field.
        pub fn boundaryDerivative(self: Self, comptime ranks: [N]usize, comptime extents: [N]isize, cell: [N]isize, field: []const f64) f64 {
            comptime var stencils: [N][3 * O]f64 = undefined;

            inline for (0..N) |i| {
                stencils[i] = comptime boundaryDerivativeStencil(ranks[i], extents[i], O);
            }

            comptime var stencil_sizes: [N]usize = undefined;

            inline for (0..N) |i| {
                stencil_sizes[i] = comptime if (ranks[i] == 0) 1 else if (extents[i] == 0) 1 else 2 * O + @as(usize, @intCast(absSigned(extents[i])));
            }

            const stencil_space: IndexSpace = comptime IndexSpace.fromSize(stencil_sizes);
            const index_space: IndexSpace = CSpace.fromSize(self.size).indexSpace();

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (comptime stencil_indices.next()) |stencil_index| {
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    if (ranks[i] != 0 and extents[i] != 0) {
                        coef *= stencils[i][stencil_index[i]];
                    }
                }

                var offset_cell: [N]isize = undefined;

                inline for (0..N) |i| {
                    if (extents[i] > 0) {
                        offset_cell[i] = cell[i] + stencil_index[i] - @as(isize, @intCast(2 * O));
                    } else if (extents[i] < 0) {
                        offset_cell[i] = cell[i] + stencil_index[i] - absSigned(extents[i]);
                    } else {
                        offset_cell[i] = cell[i];
                    }
                }

                const linear = index_space.linearFromCartesian(CSpace.indexFromCell(offset_cell));

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
            comptime var stencils: [N][3 * O]f64 = undefined;

            inline for (0..N) |i| {
                stencils[i] = comptime boundaryDerivativeStencil(ranks[i], extents[i], O);
            }

            comptime var result: f64 = 1.0;

            inline for (0..N) |i| {
                if (ranks[i] > 0) {
                    if (extents[i] > 0) {
                        result *= comptime stencils[i][2 * O + extents[i] - 1];
                    } else {
                        result *= comptime stencils[i][0];
                    }
                }
            }

            var scaled_result = result;

            // Covariantly transform result
            inline for (0..N) |i| {
                var scale: f64 = @floatFromInt(self.size[i]);
                scale /= self.physical_bounds.size[i];

                inline for (0..ranks[i]) |_| {
                    scaled_result *= scale;
                }
            }

            return scaled_result;
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

fn absSigned(i: isize) isize {
    return if (i < 0) -i else i;
}

fn boundaryDerivativeStencil(comptime R: usize, comptime extent: isize, comptime O: usize) [3 * O]f64 {
    if (extent > O) {
        @compileError("Extent must be <= O");
    }

    var result: [3 * O]f64 = undefined;

    if (extent <= 0) {
        const grid = vertexCenteredGrid(f64, absSigned(extent), 2 * O);

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
        const grid = vertexCenteredGrid(f64, 2 * O, absSigned(extent));

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
