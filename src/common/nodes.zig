const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;
const panic = std.debug.panic;

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const lac = @import("../lac/lac.zig");

const boundary = @import("boundary.zig");

const BoundaryKind = boundary.BoundaryKind;
const BoundaryEngine = boundary.BoundaryEngine;
const Robin = boundary.Robin;

/// Represents a single uniform block of nodes, along with ghost nodes along
/// each boundary. Provides routines for applying centered stencils of order `2M + 1` to
/// each node in the block. Nodes are indexed by a `[N]isize` where the isize on each axis
/// may range from (-M, size + M).
pub fn NodeSpace(comptime N: usize, comptime M: usize) type {
    return struct {
        /// The number of nodes along each axis.
        size: [N]usize,
        /// The physical bounds of the node space.
        bounds: RealBox,

        const Self = @This();
        const AxisMask = geometry.AxisMask(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexBox = geometry.IndexBox(N);
        const RealBox = geometry.RealBox(N);
        const Region = geometry.Region(N);

        const Stencils = basis.Stencils;

        const IndexMixin = geometry.IndexMixin(N);
        const coarsened = IndexMixin.coarsened;
        const refined = IndexMixin.refined;
        const toSigned = IndexMixin.toSigned;
        const toUnsigned = IndexMixin.toUnsigned;

        /// Returns the number of cells along each axis for this node space.
        pub fn cellSize(self: Self) [N]usize {
            return self.size;
        }

        pub fn cellSpace(self: Self) IndexSpace {
            return IndexSpace.fromSize(self.size);
        }

        pub fn cellPosition(self: Self, cell: [N]usize) [N]f64 {
            return self.nodePosition(toSigned(cell));
        }

        // *****************************
        // Nodes ***********************
        // *****************************

        /// Returns the number of nodes along each axis for this node space.
        pub fn nodeSize(self: Self) [N]usize {
            var result: [N]usize = undefined;

            for (0..N) |i| {
                result[i] = self.size[i] + 2 * M;
            }

            return result;
        }

        /// Return the position of the given node.
        pub fn nodePosition(self: Self, node: [N]isize) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                result[i] = @as(f64, @floatFromInt(node[i])) + 0.5;
                result[i] /= @as(f64, @floatFromInt(self.size[i]));
            }

            return self.bounds.transformPos(result);
        }

        pub fn nodeOffsetPosition(self: Self, region: Region, node: [N]isize) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                result[i] = switch (region.sides[i]) {
                    .left => 0.0,
                    .right => 1.0,
                    .middle => (@as(f64, @floatFromInt(node[i])) + 0.5) / @as(f64, @floatFromInt(self.size[i])),
                };
            }

            return self.bounds.transformPos(result);
        }

        /// Returns an index space over the node space.
        pub fn indexSpace(self: Self) IndexSpace {
            return IndexSpace.fromSize(self.nodeSize());
        }

        /// Compute an index from the given node
        pub fn indexFromNode(node: [N]isize) [N]usize {
            var index: [N]usize = undefined;

            inline for (0..N) |i| {
                index[i] = @intCast(@as(isize, @intCast(M)) + node[i]);
            }

            return index;
        }

        /// Computes the total number of nodes in this `NodeSpace`.
        pub fn numNodes(self: Self) usize {
            return self.indexSpace().total();
        }

        /// Computes the value of a field at a given node.
        pub fn nodeValue(self: Self, node: [N]isize, field: []const f64) f64 {
            const linear = self.indexSpace().linearFromCartesian(indexFromNode(node));
            return field[linear];
        }

        /// Sets the value of a field at a given node.
        pub fn setNodeValue(self: Self, node: [N]isize, field: []f64, v: f64) void {
            const linear = self.indexSpace().linearFromCartesian(indexFromNode(node));
            field[linear] = v;
        }

        pub fn NodeIterator(comptime F: usize) type {
            if (comptime F > M) {
                @compileError("F must be less than or equal to M.");
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

        pub fn nodes(self: Self, comptime F: usize) NodeIterator(F) {
            return NodeIterator(F).init(self.size);
        }

        // *************************************
        // Vertices ****************************
        // *************************************

        /// Returns the position of the given vertex.
        pub fn vertexPosition(self: Self, vertex: [N]isize) [N]f64 {
            var result: [N]f64 = undefined;

            for (0..N) |i| {
                result[i] = @floatFromInt(vertex[i]);
                result[i] /= @as(f64, @floatFromInt(self.size[i]));
            }

            return self.bounds.transformPos(result);
        }

        // **************************************
        // Boundary *****************************
        // **************************************

        pub fn boundaryPosition(self: Self, comptime region: Region, cell: [N]usize) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                result[i] = switch (comptime region.sides[i]) {
                    .left => 0.0,
                    .right => 1.0,
                    .middle => (@as(f64, @floatFromInt(cell[i])) + 0.5) / @as(f64, @floatFromInt(self.size[i])),
                };
            }

            return self.bounds.transformPos(result);
        }

        // *************************************
        // Stencils ****************************
        // *************************************

        /// Finds the value of a field at a certain cell.
        pub fn value(self: Self, cell: [N]usize, field: []const f64) f64 {
            assert(field.len == self.numNodes());
            return self.nodeValue(toSigned(cell), field);
        }

        /// Sets the value of a field at a certain cell.
        pub fn setValue(self: Self, cell: [N]usize, field: []f64, v: f64) void {
            assert(field.len == self.numNodes());
            self.setNodeValue(toSigned(cell), field, v);
        }

        // *********************************
        // Order ***************************
        // *********************************

        pub fn order(self: Self, comptime O: usize) Order(O) {
            return .{
                .space = self,
            };
        }

        pub fn Order(comptime O: usize) type {
            return struct {
                space: Self,

                /// Prolongs the value of the field to some subcell.
                pub fn prolongCell(self: @This(), subcell: [N]usize, field: []const f64) f64 {
                    assert(field.len == self.space.numNodes());

                    // Build stencils for both the left and right case at comptime.
                    const lstencil: [2 * O + 1]f64 = comptime Stencils(O).prolongCell(false);
                    const rstencil: [2 * O + 1]f64 = comptime Stencils(O).prolongCell(true);

                    // Accumulate result to this variable.
                    var result: f64 = 0.0;
                    // Find central node for applying stencil.
                    const central_node = toSigned(coarsened(subcell));
                    // Loop over stencil space.
                    comptime var stencil_indices = IndexSpace.fromSize([1]usize{2 * O + 1} ** N).cartesianIndices();

                    inline while (comptime stencil_indices.next()) |stencil_index| {
                        comptime var offset: [N]isize = undefined;
                        var coef: f64 = 1.0;

                        inline for (0..N) |i| {
                            const idx: isize = @intCast(stencil_index[i]);

                            if (@mod(subcell[i], 2) == 1) {
                                coef *= rstencil[stencil_index[i]];
                            } else {
                                coef *= lstencil[stencil_index[i]];
                            }

                            offset[i] = idx - O;
                        }

                        // std.debug.print("Coef {}, Offset ({}, {}), Value {}\n", .{ coef, offset[0], offset[1], self.nodeValue(offset_node, field) });
                        const node = IndexMixin.addSigned(central_node, offset);
                        result += coef * self.space.nodeValue(node, field);
                    }

                    return result;
                }

                /// Prolongs the value of the field to some subcell.
                pub fn prolongVertex(self: @This(), subcell: [N]usize, field: []const f64) f64 {
                    if (comptime O == 0) {
                        return self.prolongCell(subcell, field);
                    }

                    assert(field.len == self.space.numNodes());

                    // Build stencils for both the left and right case at comptime.
                    // Vertex centered
                    const lstencil: [2 * O]f64 = comptime Stencils(O).prolongVertex(false);
                    const rstencil: [2 * O]f64 = comptime Stencils(O).prolongVertex(true);

                    // Accumulate result to this variable.
                    var result: f64 = 0.0;
                    // Find central node for applying stencil.
                    const central_node = toSigned(coarsened(subcell));
                    // Loop over stencil space.
                    comptime var stencil_indices = IndexSpace.fromSize([1]usize{2 * O} ** N).cartesianIndices();

                    inline while (comptime stencil_indices.next()) |stencil_index| {
                        var coef: f64 = 1.0;

                        var offset_node: [N]isize = undefined;

                        inline for (0..N) |i| {
                            if (@mod(subcell[i], 2) == 0) {
                                coef *= rstencil[stencil_index[i]];
                                offset_node[i] = central_node[i] + stencil_index[i] - O;
                            } else {
                                coef *= lstencil[stencil_index[i]];
                                offset_node[i] = central_node[i] + stencil_index[i] - O + 1;
                            }
                        }

                        result += coef * self.space.nodeValue(offset_node, field);
                    }

                    return result;
                }

                /// Restricts the value of a field to a supercell.
                pub fn restrict(self: @This(), supercell: [N]usize, field: []const f64) f64 {
                    assert(field.len == self.space.numNodes());

                    // Compute the stencil and space of indices over which the stencil is defined.
                    const stencil: [2 * O + 2]f64 = comptime Stencils(O + 1).restrict();
                    const stencil_space: IndexSpace = comptime IndexSpace.fromSize([1]usize{2 * O + 2} ** N);
                    // Accumulate the resulting value in this variable
                    var result: f64 = 0.0;
                    // Compute the central node for applying the stencil
                    const central_node = toSigned(refined(supercell));
                    // Loop over each stencil index at comptime
                    comptime var stencil_indices = stencil_space.cartesianIndices();

                    inline while (comptime stencil_indices.next()) |stencil_index| {
                        // Compute a coefficient for this index at comptime.
                        comptime var coef: f64 = 1.0;
                        comptime var offset: [N]isize = undefined;

                        inline for (0..N) |i| {
                            const idx: isize = @intCast(stencil_index[i]);

                            coef *= stencil[stencil_index[i]];
                            offset[i] = idx - (O + 1) + 1;
                        }

                        const node = IndexMixin.addSigned(central_node, offset);
                        result += coef * self.space.nodeValue(node, field);

                        // std.debug.print("Coef {}, Offset ({}, {}), Value {}\n", .{ coef, offset[0], offset[1], self.nodeValue(offset_node, field) });
                    }

                    return result;
                }

                // *********************************
                // Operators ***********************
                // *********************************

                /// Computes the result of an operation (specified by the `ranks` argument) at the given
                /// cell, acting on the field defined over the whole nodespace.
                pub fn op(self: @This(), comptime ranks: [N]usize, cell: [N]usize, field: []const f64) f64 {
                    assert(field.len == self.space.numNodes());

                    const stencils = comptime opStencils(ranks);
                    const stencil_space = comptime IndexSpace.fromSize([_]usize{2 * O + 1} ** N);

                    var result: f64 = 0.0;

                    const central_node = toSigned(cell);

                    comptime var stencil_indices = stencil_space.cartesianIndices();

                    inline while (comptime stencil_indices.next()) |stencil_index| {
                        // Compute coefficient for this position in the stencil space.
                        comptime var coef: f64 = 1.0;
                        comptime var offset: [N]isize = undefined;

                        inline for (0..N) |i| {
                            const idx: isize = @intCast(stencil_index[i]);

                            coef *= stencils[i][stencil_index[i]];
                            offset[i] = idx - O;
                        }

                        if (comptime (@abs(coef) == 0.0)) {
                            continue;
                        }

                        const node = IndexMixin.addSigned(central_node, offset);
                        result += coef * self.space.nodeValue(node, field);
                    }

                    // Covariantly transform result
                    inline for (0..N) |i| {
                        var scale: f64 = @floatFromInt(self.space.size[i]);
                        scale /= self.space.bounds.size[i];

                        inline for (0..ranks[i]) |_| {
                            result *= scale;
                        }
                    }

                    return result;
                }

                /// Computes the diagonal of the stencil product corresponding to
                /// the operator set by `ranks`.
                pub fn opDiagonal(self: @This(), comptime ranks: [N]usize) f64 {
                    const stencils = comptime opStencils(ranks);

                    comptime var coef: f64 = 1.0;

                    inline for (0..N) |i| {
                        coef *= stencils[i][O];
                    }

                    var result: f64 = coef;

                    // Covariantly transform result
                    inline for (0..N) |i| {
                        var scale: f64 = @floatFromInt(self.space.size[i]);
                        scale /= self.space.bounds.size[i];

                        inline for (0..ranks[i]) |_| {
                            result *= scale;
                        }
                    }

                    return result;
                }

                /// Builds an set of stencils for each axis for a given operator.
                fn opStencils(ranks: [N]usize) [N][2 * O + 1]f64 {
                    var result: [N][2 * O + 1]f64 = undefined;

                    for (0..N) |i| {
                        switch (ranks[i]) {
                            0 => result[i] = Stencils(O).value(),
                            1 => result[i] = Stencils(O).derivative(),
                            2 => result[i] = Stencils(O).secondDerivative(),
                            else => panic("operators only defined for ranks <=2", .{}),
                        }
                    }

                    return result;
                }

                // *********************************
                // Dissispation ********************
                // *********************************

                pub fn dissipation(self: @This(), comptime axis: usize, cell: [N]usize, field: []const f64) f64 {
                    const stencil = Stencils(O).dissipation();

                    var result: f64 = 0.0;

                    inline for (0..2 * O + 1) |i| {
                        const offset: isize = @as(isize, @intCast(i)) - O;

                        var node = IndexMixin.toSigned(cell);
                        node[axis] += offset;

                        result += stencil[i] * self.space.nodeValue(node, field);
                    }

                    return result;
                }

                // *********************************
                // Boundary ************************
                // *********************************

                const BO = 2 * O + 1;

                pub fn boundaryOp(self: @This(), comptime extents: [N]isize, comptime flux: ?usize, node: [N]isize, field: []const f64) f64 {
                    @setEvalBranchQuota(10000);

                    assert(field.len == self.space.numNodes());

                    const stencils = comptime boundaryStencils(extents, flux);
                    const stencil_lens = comptime boundaryStencilLens(extents);

                    const stencil_space: IndexSpace = comptime IndexSpace.fromSize(stencil_lens);

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
                            const idx: isize = @intCast(stencil_index[i]);

                            if (comptime extents[i] > 0) {
                                offset_node[i] = @as(isize, @intCast(self.space.size[i] - 1)) + idx - BO + 1;
                            } else if (comptime extents[i] < 0) {
                                offset_node[i] = @as(isize, @intCast(BO - 1)) - idx;
                            } else {
                                offset_node[i] = node[i];
                            }
                        }

                        result += coef * self.space.nodeValue(offset_node, field);
                    }

                    // Covariantly transform result

                    if (flux) |axis| {
                        var scale: f64 = @floatFromInt(self.space.size[axis]);
                        scale /= self.space.bounds.size[axis];
                        result *= scale;
                    }

                    return result;
                }

                pub fn boundaryOpCoef(self: @This(), comptime extents: [N]isize, comptime flux: ?usize) f64 {
                    @setEvalBranchQuota(10000);

                    const stencils = comptime boundaryStencils(extents, flux);
                    const stencil_lens = comptime boundaryStencilLens(extents);

                    comptime var coef: f64 = 1.0;

                    inline for (0..N) |i| {
                        if (extents[i] != 0) {
                            coef *= stencils[i][stencil_lens[i] - 1];
                        }
                    }

                    // Covariantly transform result
                    var result = coef;

                    if (flux) |axis| {
                        var scale: f64 = @floatFromInt(self.space.size[axis]);
                        scale /= self.space.bounds.size[axis];
                        result *= scale;
                    }

                    return result;
                }

                fn boundaryStencils(comptime extents: [N]isize, comptime flux: ?usize) [N][2 * BO]f64 {
                    var result: [N][2 * BO]f64 = undefined;

                    for (0..N) |axis| {
                        if (flux == axis) {
                            const stencil = comptime Stencils(BO).boundaryFlux(@abs(extents[axis]));

                            inline for (0..stencil.len) |j| {
                                result[axis][j] = stencil[j];
                            }
                        } else {
                            const stencil = comptime Stencils(BO).boundaryValue(@abs(extents[axis]));

                            inline for (0..stencil.len) |j| {
                                result[axis][j] = stencil[j];
                            }
                        }
                    }

                    return result;
                }

                fn boundaryStencilLens(extents: [N]isize) [N]usize {
                    var result: [N]usize = undefined;

                    for (0..N) |axis| {
                        if (extents[axis] != 0) {
                            result[axis] = BO + @abs(extents[axis]);
                        } else {
                            result[axis] = 1;
                        }
                    }

                    return result;
                }
            };
        }
    };
}

test {
    _ = boundary;
}

test "node space iteration" {
    const expectEqualSlices = std.testing.expectEqualSlices;

    const N = 2;
    const M = 1;

    const Nodes = NodeSpace(N, M);
    const RealBox = geometry.RealBox(N);

    const space: Nodes = .{
        .size = .{ 1, 2 },
        .bounds = RealBox.unit,
    };

    const expected = [_][2]isize{
        [2]isize{ -1, -1 },
        [2]isize{ 0, -1 },
        [2]isize{ 1, -1 },
        [2]isize{ -1, 0 },
        [2]isize{ 0, 0 },
        [2]isize{ 1, 0 },
        [2]isize{ -1, 1 },
        [2]isize{ 0, 1 },
        [2]isize{ 1, 1 },
        [2]isize{ -1, 2 },
        [2]isize{ 0, 2 },
        [2]isize{ 1, 2 },
    };

    var nodes = space.nodes(M);
    var linear: usize = 0;

    while (nodes.next()) |node| : (linear += 1) {
        try expectEqualSlices(isize, &node, &expected[linear]);
    }
}

test "node space restriction and prolongation" {
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;

    const N = 2;
    const M = 1;
    const Nodes = NodeSpace(N, M);
    const RealBox = geometry.RealBox(N);

    const space: Nodes = .{ .size = .{ 20, 20 }, .bounds = RealBox.unit };

    const function = try allocator.alloc(f64, space.numNodes());
    defer allocator.free(function);

    // **************************
    // Set function values

    var nodes = space.nodes(M);

    while (nodes.next()) |node| {
        const x: f64 = @floatFromInt(node[0]);
        const y: f64 = @floatFromInt(node[1]);
        space.setNodeValue(node, function, x + y);
    }

    // **************************
    // Test restriction

    const order = space.order(M);

    try expect(order.restrict(.{ 0, 0 }, function) == 1.0);
    try expect(order.restrict(.{ 1, 1 }, function) == 5.0);
    try expect(order.restrict(.{ 2, 2 }, function) == 9.0);

    // **************************
    // Test prolongation

    try expect(order.prolongCell(.{ 0, 0 }, function) == -0.5);
    try expect(order.prolongCell(.{ 1, 1 }, function) == 0.5);
    try expect(order.prolongCell(.{ 2, 2 }, function) == 1.5);
}

test "node space smoothing" {
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;

    const N = 2;
    const M = 2;

    const FaceIndex = geometry.FaceIndex(N);
    const RealBox = geometry.RealBox(N);
    const Nodes = NodeSpace(N, M);

    const Boundary = struct {
        pub fn kind(_: FaceIndex) BoundaryKind {
            return .robin;
        }

        pub fn robin(_: @This(), _: [2]f64, _: FaceIndex) Robin {
            return Robin.diritchlet(0.0);
        }
    };

    const Poisson = struct {
        fn op(nodes: Nodes, cell: [2]usize, field: []const f64) f64 {
            var result: f64 = 0.0;

            inline for (0..2) |i| {
                comptime var ranks: [2]usize = [1]usize{0} ** 2;
                ranks[i] = 2;

                result += nodes.order(M).op(ranks, cell, field);
            }

            return -result;
        }

        fn opDiag(nodes: Nodes) f64 {
            var result: f64 = 0.0;

            inline for (0..2) |i| {
                comptime var ranks: [2]usize = [1]usize{0} ** 2;
                ranks[i] = 2;

                result += nodes.order(M).opDiagonal(ranks);
            }

            return -result;
        }
    };

    // **************************
    // Mesh

    const space: Nodes = .{
        .size = [2]usize{ 8, 8 },
        .bounds = RealBox.unit,
    };

    // ***************************
    // Variables

    const sol: []f64 = try allocator.alloc(f64, space.numNodes());
    defer allocator.free(sol);

    const scratch: []f64 = try allocator.alloc(f64, space.numNodes());
    defer allocator.free(scratch);

    const rhs: []f64 = try allocator.alloc(f64, space.numNodes());
    defer allocator.free(rhs);

    @memset(sol, 0.0);
    @memset(rhs, 1.0);

    // ***************************
    // Perform smoothing

    var residual: f64 = 0.0;

    for (0..1000) |iter| {
        _ = iter;
        BoundaryEngine(2, 2, 2).new(space).fill(Boundary{}, sol);

        // Compute residual
        {
            residual = 0.0;

            var cells = space.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const vrhs = space.value(cell, rhs);
                const lap = Poisson.op(space, cell, sol);

                residual += (vrhs - lap) * (vrhs - lap);
            }

            residual = @sqrt(residual);
        }

        // Smooth
        {
            var cells = space.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const vsol = space.value(cell, sol);
                const vrhs = space.value(cell, rhs);
                const lap = Poisson.op(space, cell, sol);
                const diag = Poisson.opDiag(space);

                const v = vsol + 2.0 / 3.0 * (vrhs - lap) / diag;
                space.setValue(cell, scratch, v);
            }

            @memcpy(sol, scratch);
        }
    }

    try expect(residual <= 10e-12);
}

test "node space two grid" {
    const ArenaAllocator = std.heap.ArenaAllocator;
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;

    const N = 2;
    const M = 2;

    const FaceIndex = geometry.FaceIndex(N);
    const RealBox = geometry.RealBox(N);
    const BiCGStabSolver = lac.BiCGStabSolver;
    const Nodes = NodeSpace(N, M);
    const BoundaryEngine_ = BoundaryEngine(N, M, M);

    const Boundary = struct {
        pub fn kind(_: FaceIndex) BoundaryKind {
            return .robin;
        }

        pub fn robin(_: @This(), _: [N]f64, _: FaceIndex) Robin {
            return Robin.diritchlet(0.0);
        }
    };

    const Poisson = struct {
        base: Nodes,
        scratch: []f64,

        pub fn apply(self: *const @This(), out: []f64, in: []const f64) void {
            // Copy to scratch
            {
                var cells = self.base.cellSpace().cartesianIndices();
                var linear: usize = 0;

                while (cells.next()) |cell| : (linear += 1) {
                    self.base.setValue(cell, self.scratch, in[linear]);
                }
            }

            BoundaryEngine_.new(self.base).fill(Boundary{}, self.scratch);

            // Apply
            {
                var cells = self.base.cellSpace().cartesianIndices();
                var linear: usize = 0;

                while (cells.next()) |cell| : (linear += 1) {
                    out[linear] = op(self.base, cell, self.scratch);
                }
            }
        }

        fn op(nodes: Nodes, cell: [2]usize, field: []const f64) f64 {
            var result: f64 = 0.0;

            inline for (0..N) |i| {
                comptime var ranks: [2]usize = [1]usize{0} ** 2;
                ranks[i] = 2;

                result += nodes.order(M).op(ranks, cell, field);
            }

            return -result;
        }

        fn opDiag(nodes: Nodes) f64 {
            var result: f64 = 0.0;

            inline for (0..N) |i| {
                comptime var ranks: [2]usize = [1]usize{0} ** 2;
                ranks[i] = 2;

                result += nodes.order(M).opDiagonal(ranks);
            }

            return -result;
        }
    };

    // **************************
    // Mesh

    const base: Nodes = .{
        .size = [2]usize{ 8, 8 },
        .bounds = RealBox.unit,
    };

    const fine: Nodes = .{
        .size = [2]usize{ 16, 16 },
        .bounds = RealBox.unit,
    };

    // **************************
    // Variables

    const base_solution: []f64 = try allocator.alloc(f64, base.numNodes());
    defer allocator.free(base_solution);

    const base_old: []f64 = try allocator.alloc(f64, base.numNodes());
    defer allocator.free(base_old);

    const base_rhs: []f64 = try allocator.alloc(f64, base.numNodes());
    defer allocator.free(base_rhs);

    const base_err: []f64 = try allocator.alloc(f64, base.numNodes());
    defer allocator.free(base_err);

    const base_scratch: []f64 = try allocator.alloc(f64, base.numNodes());
    defer allocator.free(base_scratch);

    const fine_solution: []f64 = try allocator.alloc(f64, fine.numNodes());
    defer allocator.free(fine_solution);

    const fine_res: []f64 = try allocator.alloc(f64, fine.numNodes());
    defer allocator.free(fine_res);

    const fine_scratch: []f64 = try allocator.alloc(f64, fine.numNodes());
    defer allocator.free(fine_scratch);

    const fine_rhs: []f64 = try allocator.alloc(f64, fine.numNodes());
    defer allocator.free(fine_rhs);

    // ****************************
    // Setup problem

    @memset(base_solution, 0.0);
    @memset(fine_solution, 0.0);
    @memset(fine_rhs, 1.0);

    // ****************************
    // Multigrid iteration

    // Build scratch allocator
    var arena: ArenaAllocator = ArenaAllocator.init(allocator);
    defer arena.deinit();

    const scratch: Allocator = arena.allocator();

    var residual: f64 = 0.0;

    for (0..20) |_| {
        defer _ = arena.reset(.retain_capacity);

        // Compute current error
        {
            residual = 0.0;

            BoundaryEngine_.new(fine).fill(Boundary{}, fine_solution);

            var cells = fine.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const rhs = fine.value(cell, fine_rhs);
                const lap = Poisson.op(fine, cell, fine_solution);

                residual += (rhs - lap) * (rhs - lap);
            }

            residual = @sqrt(residual);

            // std.debug.print("Iteration {}, Residual {}\n", .{ iter, residual });
        }

        // Perform presmoothing
        for (0..10) |_| {
            BoundaryEngine_.new(fine).fill(Boundary{}, fine_solution);

            var cells = fine.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const val = fine.value(cell, fine_solution);
                const rhs = fine.value(cell, fine_rhs);
                const lap = Poisson.op(fine, cell, fine_solution);
                const diag = Poisson.opDiag(fine);

                const v = val + 2.0 / 3.0 * (rhs - lap) / diag;
                fine.setValue(cell, fine_scratch, v);
            }

            @memcpy(fine_solution, fine_scratch);
        }

        // Residual calculation
        {
            BoundaryEngine_.new(fine).fill(Boundary{}, fine_solution);

            var cells = fine.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const rhs = fine.value(cell, fine_rhs);
                const lap = Poisson.op(fine, cell, fine_solution);

                fine.setValue(cell, fine_res, rhs - lap);
            }
        }

        // Restrict Solution
        {
            BoundaryEngine_.new(fine).fill(Boundary{}, fine_solution);

            var cells = base.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const sol = fine.order(M).restrict(cell, fine_solution);
                base.setValue(cell, base_solution, sol);
            }
        }

        // Right Hand Side computation
        {
            BoundaryEngine_.new(base).fill(Boundary{}, base_solution);

            var cells = base.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const res = fine.order(0).restrict(cell, fine_res);
                const val = Poisson.op(base, cell, base_solution);

                base.setValue(cell, base_rhs, res + val);
            }

            @memcpy(base_old, base_solution);
        }

        // ****************************
        // Base solving

        const x = try scratch.alloc(f64, base.cellSpace().total());
        defer scratch.free(x);

        const b = try scratch.alloc(f64, base.cellSpace().total());
        defer scratch.free(b);

        // Transfer from base solution to x and base rhs to b
        {
            var cells = base.cellSpace().cartesianIndices();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                x[linear] = base.value(cell, base_solution);
                b[linear] = base.value(cell, base_rhs);
            }
        }

        // Solve
        try BiCGStabSolver.new(1000, 10e-15).solve(scratch, Poisson{
            .base = base,
            .scratch = base_scratch,
        }, x, b);

        // Transfer from x to base solution
        {
            var cells = base.cellSpace().cartesianIndices();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                base.setValue(cell, base_solution, x[linear]);
            }
        }

        // Compute Error
        {
            BoundaryEngine_.new(base).fill(Boundary{}, base_solution);
            BoundaryEngine_.new(base).fill(Boundary{}, base_old);

            for (0..base.numNodes()) |idx| {
                base_err[idx] = base_solution[idx] - base_old[idx];
            }
        }

        // Correct fine solution
        {
            var cells = fine.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const sol = fine.value(cell, fine_solution);
                const err = base.order(M).prolongCell(cell, base_err);

                fine.setValue(cell, fine_solution, sol + err);
            }
        }

        // Post smoothing
        for (0..10) |_| {
            BoundaryEngine_.new(fine).fill(Boundary{}, fine_solution);

            var cells = fine.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const val = fine.value(cell, fine_solution);
                const rhs = fine.value(cell, fine_rhs);
                const lap = Poisson.op(fine, cell, fine_solution);
                const diag = Poisson.opDiag(fine);

                const v = val + 2.0 / 3.0 * (rhs - lap) / diag;
                fine.setValue(cell, fine_scratch, v);
            }

            @memcpy(fine_solution, fine_scratch);
        }
    }

    try expect(residual <= 10e-12);
}
