const std = @import("std");
const geometry = @import("../geometry/geometry.zig");

const grids = @import("grids.zig");
const lagrange = @import("lagrange.zig");

/// A set of cells over which basic stencils can be applied. This includes a buffer region of
/// length `E`, and all stencils are built to order `O`.
pub fn NodeSpace(comptime N: usize, comptime O: usize) type {
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

        pub fn total(self: Self) usize {
            return self.indexSpace().total();
        }

        pub fn indexFromNode(node: [N]isize) [N]usize {
            var index: [N]usize = undefined;

            for (0..N) |i| {
                index[i] = @intCast(@as(isize, @intCast(2 * O)) + node[i]);
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
            const linear = self.indexSpace().linearFromCartesian(indexFromNode(cell));
            return field[linear];
        }

        /// Sets the value of a field at a cell.
        pub fn setValue(self: Self, cell: [N]isize, field: []f64, v: f64) void {
            const linear = self.indexSpace().linearFromCartesian(indexFromNode(cell));
            field[linear] = v;
        }

        /// Prolongs the value of a field to a subcell.
        pub fn prolong(self: Self, subcell: [N]isize, field: []const f64) f64 {
            // Build stencils for both the left and right case at comptime.
            const lstencil: [2 * O + 1]f64 = comptime prolongStencil(false, O);
            const rstencil: [2 * O + 1]f64 = comptime prolongStencil(true, O);

            var result: f64 = 0.0;

            const index_space: IndexSpace = self.indexSpace();

            comptime var stencil_indices = IndexSpace.fromSize([1]usize{2 * O + 1} ** N).cartesianIndices();

            inline while (comptime stencil_indices.next()) |stencil_index| {
                var coef: f64 = 1.0;

                for (0..N) |i| {
                    if (@mod(subcell[i], 2) == 1) {
                        coef *= rstencil[stencil_index[i]];
                    } else {
                        coef *= lstencil[stencil_index[i]];
                    }
                }

                var offset_cell: [N]isize = undefined;

                inline for (0..N) |i| {
                    offset_cell[i] = @divTrunc(subcell[i], 2) + stencil_index[i] - O;
                }

                const linear = index_space.linearFromCartesian(indexFromNode(offset_cell));

                result += coef * field[linear];
            }

            return result;
        }

        /// Restricts the value of a field to a supercell.
        pub fn restrict(self: Self, supercell: [N]isize, field: []const f64) f64 {
            const stencil: [2 * O + 2]f64 = comptime restrictStencil(O);

            const stencil_space: IndexSpace = comptime IndexSpace.fromSize([1]usize{2 * O + 2} ** N);
            const index_space: IndexSpace = self.indexSpace();

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (comptime stencil_indices.next()) |stencil_index| {
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    coef *= stencil[stencil_index[i]];
                }

                var offset_cell: [N]isize = undefined;

                inline for (0..N) |i| {
                    offset_cell[i] = 2 * supercell[i] + stencil_index[i] - O;
                }

                const linear = index_space.linearFromCartesian(indexFromNode(offset_cell));

                result += coef * field[linear];
            }

            return result;
        }

        pub fn NodeIterator(comptime F: usize) type {
            if (comptime F > 2 * O) {
                @compileError("F must be less than or equal to E.");
            }

            return struct {
                inner: IndexSpace.CartesianIterator,

                pub fn init(size: [N]usize) @This() {
                    var index_size: [N]usize = size;

                    for (0..N) |i| {
                        index_size[i] += 2 * F;
                    }

                    return .{ .inner = IndexSpace.fromSize(index_size).cartesianIndices() };
                }

                pub fn next(self: *@This()) ?[N]isize {
                    const index = self.inner.next() orelse return null;

                    var result: [N]isize = undefined;

                    for (0..N) |i| {
                        result[i] = @as(isize, @intCast(index[i])) - F;
                    }

                    return result;
                }
            };
        }

        pub fn nodesToExtent(self: Self, comptime F: usize) NodeIterator(F) {
            return NodeIterator(F).init(self.size);
        }

        pub fn nodes(self: Self) NodeIterator(0) {
            return self.nodesToExtent(0);
        }

        pub fn fullNodes(self: Self) NodeIterator(2 * O) {
            return self.nodesToExtent(2 * O);
        }
    };
}

fn prolongStencil(comptime side: bool, comptime O: usize) [2 * O + 1]f64 {
    const ngrid = grids.nodeCenteredGrid(f64, O, O);
    const point = if (side) 0.25 else -0.25;
    return lagrange.valueStencil(2 * O + 1, ngrid, point);
}

fn restrictStencil(comptime O: usize) [2 * O + 2]f64 {
    const vgrid = grids.vertexCenteredGrid(f64, O + 1, O + 1);
    return lagrange.valueStencil(2 * O + 2, vgrid, 0.0);
}

test "node iteration" {
    const expectEqualSlices = std.testing.expectEqualSlices;

    const node_space = NodeSpace(2, 2).fromSize([_]usize{ 1, 2 });

    const expected = [_][2]isize{
        [2]isize{ -2, -2 },
        [2]isize{ -2, -1 },
        [2]isize{ -2, 0 },
        [2]isize{ -2, 1 },
        [2]isize{ -2, 2 },
        [2]isize{ -2, 3 },
        [2]isize{ -1, -2 },
        [2]isize{ -1, -1 },
        [2]isize{ -1, 0 },
        [2]isize{ -1, 1 },
        [2]isize{ -1, 2 },
        [2]isize{ -1, 3 },
        [2]isize{ 0, -2 },
        [2]isize{ 0, -1 },
        [2]isize{ 0, 0 },
        [2]isize{ 0, 1 },
        [2]isize{ 0, 2 },
        [2]isize{ 0, 3 },
        [2]isize{ 1, -2 },
        [2]isize{ 1, -1 },
        [2]isize{ 1, 0 },
        [2]isize{ 1, 1 },
        [2]isize{ 1, 2 },
        [2]isize{ 1, 3 },
        [2]isize{ 2, -2 },
        [2]isize{ 2, -1 },
        [2]isize{ 2, 0 },
        [2]isize{ 2, 1 },
        [2]isize{ 2, 2 },
        [2]isize{ 2, 3 },
    };

    var nodes = node_space.nodesToExtent(2);
    var index: usize = 0;

    while (nodes.next()) |node| : (index += 1) {
        try expectEqualSlices(isize, &node, &expected[index]);
    }
}

test "interpolation stencils" {
    const expectEqualSlices = std.testing.expectEqualSlices;

    try expectEqualSlices(f64, &[_]f64{ 0.5, 0.5 }, &restrictStencil(0));
    try expectEqualSlices(f64, &[_]f64{ -1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0 }, &restrictStencil(1));
}
