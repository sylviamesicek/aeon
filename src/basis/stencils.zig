const std = @import("std");
const geometry = @import("../geometry/geometry.zig");

const lagrange = @import("lagrange.zig");
const nodes = @import("nodes.zig");

/// This namespace provides several functions for computing cell centered interpolation, differentiation,
/// prolongation, restriction, and boundary stencils. Here `M` corresponds to the number of support points
/// to either side of the center of the stencil (or the number of interior support points in the case of
/// boundary stencils).
pub fn Stencils(comptime M: usize) type {
    return struct {
        const Lagrange = lagrange.Lagrange(f128);

        /// Computes a value stencil of order `2M` (for lagrange polynomials, the stencil is always 1.0 at the central
        /// support point and 0.0 everywhere else).
        pub fn value() [2 * M + 1]f64 {
            var result: [2 * M + 1]f128 = undefined;
            Lagrange.value(&centeredCellGrid(), 0.0, &result);
            return floatCastSlice(2 * M + 1, result);
        }

        /// Computes a first derivative stencil of order `2M`.
        pub fn derivative() [2 * M + 1]f64 {
            var result: [2 * M + 1]f128 = undefined;
            Lagrange.derivative(&centeredCellGrid(), 0.0, &result);
            return floatCastSlice(2 * M + 1, result);
        }

        /// Computes a second derivative stencil of order `2M`.
        pub fn secondDerivative() [2 * M + 1]f64 {
            var result: [2 * M + 1]f128 = undefined;
            Lagrange.secondDerivative(&centeredCellGrid(), 0.0, &result);
            return floatCastSlice(2 * M + 1, result);
        }

        /// Computes an interpolation stencil of order `M + E - 1` for the boundary
        /// of a cell. `E` denotes the number of exterior support points.
        pub fn boundaryValue(comptime E: usize) [M + E]f64 {
            if (comptime E > M) {
                @compileError("E must be <= M when computing boundary stencils.");
            }

            var result: [M + E]f128 = undefined;
            Lagrange.value(&boundaryGrid(E), 0.0, &result);
            return floatCastSlice(M + E, result);
        }

        /// Computes an differentiation stencil of order `M + E - 1` for the boundary
        /// of a cell. `E` denotes the number of exterior support points.
        pub fn boundaryFlux(comptime E: usize) [M + E]f64 {
            if (comptime E > M) {
                @compileError("E must be <= M when computing boundary stencils.");
            }

            var result: [M + E]f128 = undefined;
            Lagrange.derivative(&boundaryGrid(E), 0.0, &result);
            return floatCastSlice(M + E, result);
        }

        /// Computes a prolongation stencil which is centered on a cell. `side` denotes which
        /// subcell to prolong to.
        pub fn prolongCell(comptime side: bool) [2 * M + 1]f64 {
            var result: [2 * M + 1]f128 = undefined;
            Lagrange.value(&centeredCellGrid(), if (side) 0.25 else -0.25, &result);
            return floatCastSlice(2 * M + 1, result);
        }

        /// Computes a prolongation stencil which is centered on a vertex.
        pub fn prolongVertex(comptime side: bool) [2 * M]f64 {
            var result: [2 * M]f128 = undefined;
            Lagrange.value(&centeredVertexGrid(), if (side) 0.25 else -0.25, &result);
            return floatCastSlice(2 * M, result);
        }

        /// Computes a restriction stencil.
        pub fn restrict() [2 * M]f64 {
            var result: [2 * M]f128 = undefined;
            Lagrange.value(&centeredVertexGrid(), 0.0, &result);
            return floatCastSlice(2 * M, result);
        }

        // ************************
        // Helpers ****************
        // ************************

        fn centeredCellGrid() [2 * M + 1]f128 {
            return cellGrid(M, M);
        }

        fn centeredVertexGrid() [2 * M]f128 {
            return vertexGrid(M, M);
        }

        fn boundaryGrid(comptime E: usize) [M + E]f128 {
            return vertexGrid(M, E);
        }

        fn cellGrid(comptime L: usize, comptime R: usize) [L + R + 1]f128 {
            var grid: [L + R + 1]f128 = undefined;

            for (0..(L + R + 1)) |i| {
                grid[i] = @as(f128, @floatFromInt(i)) - @as(f128, @floatFromInt(L));
            }

            return grid;
        }

        fn vertexGrid(comptime L: usize, comptime R: usize) [L + R]f128 {
            var grid: [L + R]f128 = undefined;

            for (0..(L + R)) |i| {
                grid[i] = @as(f128, @floatFromInt(i)) + 0.5 - @as(f128, @floatFromInt(L));
            }

            return grid;
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
}

test "stencil grids" {
    const S2 = Stencils(2);

    const expectEqualDeep = std.testing.expectEqualDeep;

    try expectEqualDeep([_]f128{ -1.0, 0.0, 1.0, 2.0 }, S2.cellGrid(1, 2));
    try expectEqualDeep([_]f128{ -1.5, -0.5, 0.5 }, S2.vertexGrid(2, 1));
    try expectEqualDeep([_]f128{ -2.0, -1.0, 0.0, 1.0, 2.0 }, S2.centeredCellGrid());
    try expectEqualDeep([_]f128{ -1.5, -0.5, 0.5, 1.5 }, S2.centeredVertexGrid());
}

test "stencils" {
    const S2 = Stencils(2);

    const expectEqualDeep = std.testing.expectEqualDeep;

    try expectEqualDeep([_]f64{ 0.0, -0.0, 1.0, 0.0, -0.0 }, S2.value());
    try expectEqualDeep([_]f64{ 1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0 }, S2.derivative());
    try expectEqualDeep([_]f64{ -1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0 }, S2.secondDerivative());
    try expectEqualDeep([_]f64{ -1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0 }, S2.restrict());
}

/// Manages the application of stencil products on functions. Supports computing values, centered derivatives
/// positions, boundary positions, boundary values, boundary derivatives, prolongation, and restriction.
/// If full is false, all cell indices are in standard index space (ie without ghost cells included).
pub fn StencilSpace(comptime N: usize, comptime E: usize) type {
    return struct {
        physical_bounds: RealBox,
        size: [N]usize,

        const Self = @This();
        const RealBox = geometry.Box(N, f64);
        const IndexBox = geometry.Box(N, usize);
        const IndexSpace = geometry.IndexSpace(N);
        const Region = geometry.Region(N);
        const Face = geometry.Face(N);
        const NodeSpace = nodes.NodeSpace(N, E);

        pub fn nodeSpace(self: Self) NodeSpace {
            return NodeSpace.fromSize(self.size);
        }

        /// Return the position of the given node.
        pub fn position(self: Self, node: [N]isize) [N]f64 {
            var result: [N]f64 = undefined;

            for (0..N) |i| {
                const origin: f64 = self.physical_bounds.origin[i];
                const width: f64 = self.physical_bounds.size[i];
                const ratio: f64 = (@as(f64, @floatFromInt(node[i])) + 0.5) / @as(f64, @floatFromInt(self.size[i]));
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
        pub fn value(self: Self, node: [N]isize, field: []const f64) f64 {
            return NodeSpace.fromSize(self.size).value(node, field);
        }

        /// Computes the diagonal coefficient of the value stencil.
        pub fn valueDiagonal(_: Self) f64 {
            return 1.0;
        }

        /// Computes the derivative of a field at a cell.
        pub fn derivative(self: Self, comptime O: usize, comptime ranks: [N]usize, node: [N]isize, field: []const f64) f64 {
            comptime var stencil_sizes: [N]usize = undefined;

            inline for (0..N) |i| {
                stencil_sizes[i] = if (ranks[i] == 0) 1 else 2 * O + 1;
            }

            comptime var stencils: [N][2 * O + 1]f64 = undefined;

            inline for (0..N) |i| {
                stencils[i] = comptime derivativeStencil(O, ranks[i]);
            }

            const stencil_space: IndexSpace = comptime IndexSpace.fromSize(stencil_sizes);
            const index_space: IndexSpace = NodeSpace.fromSize(self.size).indexSpace();

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (comptime stencil_indices.next()) |stencil_index| {
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    if (ranks[i] != 0) {
                        coef *= stencils[i][stencil_index[i]];
                    }
                }

                var offset_node: [N]isize = undefined;

                inline for (0..N) |i| {
                    if (ranks[i] != 0) {
                        offset_node[i] = node[i] + stencil_index[i] - O;
                    } else {
                        offset_node[i] = node[i];
                    }
                }

                const linear = index_space.linearFromCartesian(NodeSpace.indexFromNode(offset_node));

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
        pub fn derivativeDiagonal(self: Self, comptime O: usize, comptime ranks: [N]usize) f64 {
            comptime var stencils: [N][2 * O + 1]f64 = undefined;

            inline for (0..N) |i| {
                stencils[i] = comptime derivativeStencil(O, ranks[i]);
            }

            var result: f64 = 1.0;

            for (0..N) |i| {
                if (ranks[i] > 0) {
                    result *= stencils[i][O];
                }
            }

            // Covariantly transform result
            for (0..N) |i| {
                var scale: f64 = @floatFromInt(self.size[i]);
                scale /= self.physical_bounds.size[i];

                for (0..ranks[i]) |_| {
                    result *= scale;
                }
            }

            return result;
        }

        /// Returns the boundary position of the given cell, taking into account the extents.
        /// This cell is always given in standard index space.
        pub fn boundaryPosition(self: Self, comptime extents: [N]isize, node: [N]isize) [N]f64 {
            var result: [N]f64 = undefined;

            for (0..N) |i| {
                if (extents[i] < 0) {
                    result[i] = self.physical_bounds.origin[i];
                } else if (extents[i] > 0) {
                    result[i] = self.physical_bounds.origin[i] + self.physical_bounds.size[i];
                } else {
                    const origin: f64 = self.physical_bounds.origin[i];
                    const width: f64 = self.physical_bounds.size[i];
                    const ratio: f64 = (@as(f64, @floatFromInt(node[i])) + 0.5) / @as(f64, @floatFromInt(self.size[i]));
                    result[i] = origin + width * ratio;
                }
            }

            return result;
        }

        /// Computes the value at a boundary of a field.
        pub fn boundaryValue(
            self: Self,
            comptime L: usize,
            comptime extents: [N]isize,
            node: [N]isize,
            field: []const f64,
        ) f64 {
            return self.boundaryDerivative(L, extents, [1]usize{0} ** N, node, field);
        }

        /// Computes the outmost coefficient of a boundary stencil.
        pub fn boundaryValueCoef(
            self: Self,
            comptime L: usize,
            comptime extents: [N]isize,
        ) f64 {
            return self.boundaryDerivativeCoef(L, extents, [1]usize{0} ** N);
        }

        /// Computes the derivative at a bounday of a field.
        pub fn boundaryDerivative(
            self: Self,
            comptime L: usize,
            comptime extents: [N]isize,
            comptime ranks: [N]usize,
            node: [N]isize,
            field: []const f64,
        ) f64 {
            @setEvalBranchQuota(10000);
            comptime var stencils: [N][2 * L]f64 = undefined;
            comptime var stencil_lens: [N]usize = undefined;

            // std.debug.print("Computing Boundary Derivative of {any}\n with ranks {any} and extents {any}\n", .{ cell, ranks, extents });

            inline for (0..N) |i| {
                const stencil = comptime boundaryStencil(L, absSigned(extents[i]), ranks[i]);

                inline for (0..stencil.len) |j| {
                    stencils[i][j] = stencil[j];
                }

                if (extents[i] == 0) {
                    stencil_lens[i] = 1;
                } else {
                    stencil_lens[i] = stencil.len;
                }
            }

            const stencil_space: IndexSpace = comptime IndexSpace.fromSize(stencil_lens);
            const index_space: IndexSpace = NodeSpace.fromSize(self.size).indexSpace();

            var result: f64 = 0.0;

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (comptime stencil_indices.next()) |stencil_index| {
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    if (extents[i] != 0) {
                        coef *= stencils[i][stencil_index[i]];
                    }
                }

                var offset_node: [N]isize = undefined;

                inline for (0..N) |i| {
                    if (extents[i] > 0) {
                        offset_node[i] = @as(isize, @intCast(self.size[i] - 1)) + stencil_index[i] - L + 1;
                    } else if (extents[i] < 0) {
                        offset_node[i] = @as(isize, @intCast(L - 1)) - stencil_index[i];
                    } else {
                        offset_node[i] = node[i];
                    }
                }

                const linear = index_space.linearFromCartesian(NodeSpace.indexFromNode(offset_node));

                // std.debug.print("Cell: {any}, Coef: {}, Value: {}\n", .{ offset_cell, coef, field[linear] });

                result += coef * field[linear];
            }

            // std.debug.print("Result: {}\n", .{result});

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
        pub fn boundaryDerivativeCoef(
            self: Self,
            comptime L: usize,
            comptime extents: [N]isize,
            comptime ranks: [N]usize,
        ) f64 {
            comptime var result: f64 = 1.0;

            inline for (0..N) |i| {
                if (extents[i] != 0) {
                    const stencil = comptime boundaryStencil(L, absSigned(extents[i]), ranks[i]);

                    result *= stencil[stencil.len - 1];
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

fn derivativeStencil(comptime O: usize, comptime R: usize) [2 * O + 1]f64 {
    return switch (R) {
        0 => Stencils(O).value(),
        1 => Stencils(O).derivative(),
        2 => Stencils(O).secondDerivative(),
        else => @compileError("Rank of derivative stencil must be <= 2"),
    };
}

fn absSigned(i: isize) isize {
    return if (i < 0) -i else i;
}

fn boundaryStencil(comptime L: usize, comptime M: usize, comptime R: usize) [M + L]f64 {
    return switch (R) {
        0 => Stencils(L).boundaryValue(M),
        1 => Stencils(L).boundaryFlux(M),
        else => @compileError("Boundary stencil only supports R <= 1."),
    };
}

test "derivative stencils" {
    const expectEqualSlices = std.testing.expectEqualSlices;

    try expectEqualSlices(f64, &[_]f64{ 0.0, 1.0, 0.0 }, &derivativeStencil(1, 0));
    try expectEqualSlices(f64, &[_]f64{ -0.5, 0.0, 0.5 }, &derivativeStencil(1, 1));
    try expectEqualSlices(f64, &[_]f64{ 1.0, -2.0, 1.0 }, &derivativeStencil(1, 2));

    // std.debug.print("{any}\n", .{boundaryStencil(4, 4, 0)});
}
