//! This module handles nodes, a kind of "unpacked" representation of a function in FD codes.
//! This module is mesh agnostic, and provides routines to apply physical boundary conditions to
//! nodes, applying tensor products of stencils to node vectors (through the `NodeSpace` type), and
//! an API for applying operators to node vectors. Individual meshes must provide functions for
//! transfering between cell vectors and node vectors.

const std = @import("std");
const Allocator = std.mem.Allocator;
const panic = std.debug.panic;

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");

// Submodules
const boundary = @import("boundary.zig");

pub const BoundaryKind = boundary.BoundaryKind;
pub const Robin = boundary.Robin;
pub const isBoundary = boundary.isBoundary;

/// A map from blocks (defined in a mesh agnostic way) into `NodeVector`s. This is filled by the appropriate mesh routine.
pub const NodeMap = struct {
    offsets: []const usize,

    /// Frees a `NodeMap`.
    pub fn deinit(self: NodeMap, allocator: Allocator) void {
        allocator.free(self.offsets);
    }

    /// Returns the node offset for a given block.
    pub fn offset(self: NodeMap, block: usize) usize {
        return self.offsets[block];
    }

    /// Returns the total number of nodes in a given block.
    pub fn total(self: NodeMap, block: usize) usize {
        return self.offsets[block + 1] - self.offsets[block];
    }

    /// Returns of slice of nodes for a given block.
    pub fn slice(self: NodeMap, block: usize, nodes: anytype) @TypeOf(nodes) {
        return nodes[self.offsets[block]..self.offsets[block + 1]];
    }

    /// Returns the number of nodes in the total map.
    pub fn numNodes(self: NodeMap) usize {
        return self.offsets[self.offsets.len - 1];
    }
};

/// Represents a single uniform block of nodes, along with ghost nodes along
/// each boundary. Provides routines for applying centered stencils of order `2M + 1` to
/// each node in the block. Nodes are indexed by a `[N]isize` where the isize on each axis
/// may range from (-M, size + M). The node space is map to a simple unit cube coordinate system
/// when calculating and transforming derivatives and positions.
pub fn NodeSpace(comptime N: usize, comptime M: usize) type {
    return struct {
        /// The number of cells along each axis.
        size: [N]usize,

        const Self = @This();
        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexBox = geometry.IndexBox(N);
        const RealBox = geometry.RealBox(N);
        const Region = geometry.Region(M);

        const Stencils = basis.Stencils;

        const IndexMixin = geometry.IndexMixin(N);
        const coarsened = IndexMixin.coarsened;
        const refined = IndexMixin.refined;
        const toSigned = IndexMixin.toSigned;
        const toUnsigned = IndexMixin.toUnsigned;

        /// Constructs the node space from a given cell size.
        pub fn fromCellSize(size: [N]usize) Self {
            return .{ .size = size };
        }

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

            for (0..N) |i| {
                result[i] = (@as(f64, @floatFromInt(node[i])) + 0.5) / @as(f64, @floatFromInt(self.size[i]));
            }

            return result;
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
                result[i] = @as(f64, @floatFromInt(vertex[i])) / @as(f64, @floatFromInt(self.size[i]));
            }

            return result;
        }

        // *************************************
        // Stencils ****************************
        // *************************************

        /// Finds the value of a field at a certain cell.
        pub fn value(self: Self, cell: [N]usize, field: []const f64) f64 {
            return self.nodeValue(toSigned(cell), field);
        }

        /// Sets the value of a field at a certain cell.
        pub fn setValue(self: Self, cell: [N]usize, field: []f64, v: f64) void {
            self.setNodeValue(toSigned(cell), field, v);
        }

        /// Prolongs the value of the field to some subcell.
        pub fn prolong(self: Self, subcell: [N]usize, field: []const f64) f64 {
            // Default order to M
            const O = M;
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
                var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    if (@mod(subcell[i], 2) == 1) {
                        coef *= rstencil[stencil_index[i]];
                    } else {
                        coef *= lstencil[stencil_index[i]];
                    }
                }

                var offset_node: [N]isize = undefined;

                inline for (0..N) |i| {
                    offset_node[i] = central_node[i] + stencil_index[i] - O;
                }

                result += coef * self.nodeValue(offset_node, field);
            }

            return result;
        }

        /// Restricts the value of a field to a supercell.
        pub fn restrict(self: Self, supercell: [N]usize, field: []const f64) f64 {
            const O = M + 1;

            // Compute the stencil and space of indices over which the stencil is defined.
            const stencil: [2 * O]f64 = Stencils(O).restrict();
            const stencil_space: IndexSpace = comptime IndexSpace.fromSize([1]usize{2 * O} ** N);
            // Accumulate the resulting value in this variable
            var result: f64 = 0.0;
            // Compute the central node for applying the stencil
            const central_node = toSigned(refined(supercell));
            // Loop over each stencil index at comptime
            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (comptime stencil_indices.next()) |stencil_index| {
                // Compute a coefficient for this index at comptime.
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    coef *= stencil[stencil_index[i]];
                }

                var offset_node: [N]isize = undefined;

                inline for (0..N) |i| {
                    offset_node[i] = central_node[i] + stencil_index[i] - O + 1;
                }

                result += coef * self.nodeValue(offset_node, field);
            }

            return result;
        }

        /// Computes the result of an operation (specified by the `ranks` argument) at the given
        /// cell, acting on the field defined over the whole nodespace.
        pub fn op(self: Self, comptime ranks: [N]usize, cell: [N]usize, field: []const f64) f64 {
            const stencils = comptime opStencils(ranks);
            const stencil_space = comptime IndexSpace.fromSize([_]usize{2 * M + 1} ** N);

            var result: f64 = 0.0;

            const central_node = toSigned(cell);

            comptime var stencil_indices = stencil_space.cartesianIndices();

            inline while (comptime stencil_indices.next()) |stencil_index| {
                // Compute coefficient for this position in the stencil space.
                comptime var coef: f64 = 1.0;

                inline for (0..N) |i| {
                    coef *= stencils[i][stencil_index[i]];
                }

                if (comptime (@fabs(coef) == 0.0)) {
                    continue;
                }

                var offset_node: [N]isize = undefined;

                for (0..N) |i| {
                    offset_node[i] = central_node[i] + stencil_index[i] - M;
                }

                result += coef * self.nodeValue(offset_node, field);
            }

            // Covariantly transform result
            inline for (0..N) |i| {
                const scale: f64 = @floatFromInt(self.size[i]);

                inline for (0..ranks[i]) |_| {
                    result *= scale;
                }
            }

            return result;
        }

        /// Computes the diagonal of the stencil product corresponding to
        /// the operator set by `ranks`.
        pub fn opDiagonal(self: Self, comptime ranks: [N]usize) f64 {
            const stencils = comptime opStencils(ranks);

            comptime var coef: f64 = 1.0;

            inline for (0..N) |i| {
                coef *= stencils[i][M];
            }

            var result: f64 = coef;

            // Covariantly transform result
            inline for (0..N) |i| {
                const scale: f64 = @floatFromInt(self.size[i]);

                inline for (0..ranks[i]) |_| {
                    result *= scale;
                }
            }

            return result;
        }

        /// Builds an set of stencils for each axis for a given operator.
        fn opStencils(ranks: [N]usize) [N][2 * M + 1]f64 {
            var result: [N][2 * M + 1]f64 = undefined;

            for (0..N) |i| {
                switch (ranks[i]) {
                    0 => result[i] = Stencils(M).value(),
                    1 => result[i] = Stencils(M).derivative(),
                    2 => result[i] = Stencils(M).secondDerivative(),
                    else => panic("operators only defined for ranks <=2", .{}),
                }
            }

            return result;
        }

        // *********************************
        // Boundary ************************
        // *********************************

        pub fn boundaryPosition(self: Self, comptime region: Region, cell: [N]usize) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                result[i] = switch (comptime region.sides[i]) {
                    .left => 0.0,
                    .right => 1.0,
                    .middle => (@as(f64, @floatFromInt(cell[i])) + 0.5) / @as(f64, @floatFromInt(self.size[i])),
                };
            }

            return result;
        }

        pub fn fillBoundary(self: Self, bound: anytype, field: []f64) void {
            const regions = comptime Region.orderedRegions();

            inline for (comptime regions[1..]) |region| {
                self.fillBoundaryRegion(region, bound, field);
            }
        }

        pub fn fillBoundaryRegion(self: Self, comptime region: Region, bound: anytype, field: []f64) void {
            // Short circuit if the region does not actually have any boundary nodes
            if (comptime region.adjacency() == 0) {
                return;
            }

            // Loop over the cells touching this face.
            var inner_face_cells = region.innerFaceCells(self.size);

            while (inner_face_cells.next()) |cell| {
                // Cast to signed node index.
                const node = toSigned(cell);
                // Find position of boundary
                const pos: [N]f64 = self.boundaryPosition(region, cell);

                // Cache robin boundary conditions (if any)
                var robin: [N]Robin = undefined;

                inline for (0..N) |axis| {
                    const kind: BoundaryKind = comptime @TypeOf(bound).kind(axis);

                    if (comptime kind == .robin) {
                        const face = FaceIndex{
                            .side = region.sides[axis] == .right,
                            .axis = axis,
                        };

                        robin[axis] = bound.condition(pos, face);
                    }
                }

                // Loop over extends
                comptime var extent_indices = region.extentOffsets(M);

                inline while (comptime extent_indices.next()) |extents| {
                    // Compute target node
                    var target: [N]isize = undefined;

                    inline for (0..N) |i| {
                        target[i] = node[i] + extents[i];
                    }

                    // Set target to zero
                    self.setNodeValue(target, field, 0.0);

                    var result: f64 = 0.0;

                    // Accumulate result value
                    inline for (0..N) |axis| {
                        if (comptime region.sides[axis] == .middle) {
                            // We need not do anything
                            continue;
                        }

                        const kind: BoundaryKind = comptime @TypeOf(bound).kind(axis);

                        switch (kind) {
                            .odd, .even => {
                                var source: [N]isize = target;

                                if (comptime extents[axis] > 0) {
                                    source[axis] = node[axis] + 1 - extents[axis];
                                } else {
                                    source[axis] = node[axis] - extents[axis] - 1;
                                }

                                const source_value: f64 = self.nodeValue(source, field);
                                const fsign: f64 = comptime if (kind == .odd) -1.0 else 1.0;

                                result += fsign * source_value;
                            },
                            .robin => {
                                const vres: f64 = robin[axis].value * self.boundaryOp(extents, null, cell, field);
                                const fres: f64 = robin[axis].flux * self.boundaryOp(extents, axis, cell, field);

                                const vcoef: f64 = robin[axis].value * self.boundaryOpCoef(extents, null);
                                const fcoef: f64 = robin[axis].flux * self.boundaryOpCoef(extents, axis);

                                const rhs: f64 = robin[axis].rhs;

                                result += (rhs - vres - fres) / (vcoef + fcoef);
                            },
                        }
                    }

                    // Take the average
                    const adj: f64 = @floatFromInt(region.adjacency());
                    result /= adj;

                    // Set target to result.
                    self.setNodeValue(target, field, result);
                }
            }
        }

        const BM = 2 * M + 1;

        pub fn boundaryOp(self: Self, comptime extents: [N]isize, comptime flux: ?usize, cell: [N]usize, field: []const f64) f64 {
            @setEvalBranchQuota(10000);

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
                    if (extents[i] > 0) {
                        offset_node[i] = @as(isize, @intCast(self.size[i] - 1)) + stencil_index[i] - BM + 1;
                    } else if (extents[i] < 0) {
                        offset_node[i] = @as(isize, @intCast(BM - 1)) - stencil_index[i];
                    } else {
                        offset_node[i] = @intCast(cell[i]);
                    }
                }

                result += coef * self.nodeValue(offset_node, field);
            }

            // Covariantly transform result

            if (flux) |axis| {
                const scale: f64 = @floatFromInt(self.size[axis]);
                result *= scale;
            }

            return result;
        }

        pub fn boundaryOpCoef(self: Self, comptime extents: [N]isize, comptime flux: ?usize) f64 {
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
                const scale: f64 = @floatFromInt(self.size[axis]);
                result *= scale;
            }

            return result;
        }

        fn boundaryStencils(comptime extents: [N]isize, comptime flux: ?usize) [N][2 * BM]f64 {
            const absCast = std.math.absCast;

            var result: [N][2 * BM]f64 = undefined;

            for (0..N) |axis| {
                if (flux == axis) {
                    const stencil = comptime Stencils(BM).boundaryFlux(absCast(extents[axis]));

                    inline for (0..stencil.len) |j| {
                        result[axis][j] = stencil[j];
                    }
                } else {
                    const stencil = comptime Stencils(BM).boundaryValue(absCast(extents[axis]));

                    inline for (0..stencil.len) |j| {
                        result[axis][j] = stencil[j];
                    }
                }
            }

            return result;
        }

        fn boundaryStencilLens(extents: [N]isize) [N]usize {
            const absCast = std.math.absCast;

            var result: [N]usize = undefined;

            for (0..N) |axis| {
                if (extents[axis] != 0) {
                    result[axis] = BM + absCast(extents[axis]);
                } else {
                    result[axis] = 1;
                }
            }

            return result;
        }
    };
}

test "node iteration" {
    const expectEqualSlices = std.testing.expectEqualSlices;

    const node_space = NodeSpace(2, 4).fromCellSize([_]usize{ 1, 2 });

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

    var nodes = node_space.nodes(2);
    var index: usize = 0;

    while (nodes.next()) |node| : (index += 1) {
        try expectEqualSlices(isize, &node, &expected[index]);
    }
}

test "node space" {
    const IndexBox = geometry.IndexBox(2);
    const IndexSpace = geometry.IndexSpace(2);
    _ = IndexSpace;
    const FaceIndex = geometry.FaceIndex(2);
    const RealBox = geometry.RealBox(2);
    const Nodes = NodeSpace(2, 2);

    const expect = std.testing.expect;
    const allocator = std.testing.allocator;
    const pi = std.math.pi;

    const domain: RealBox = .{ .origin = .{ 0.0, 0.0 }, .size = .{ pi, pi } };

    const node_space = Nodes.fromCellSize([2]usize{ 100, 100 });
    const cell_space = node_space.cellSpace();

    // *********************************
    // Set field values ****************

    const field: []f64 = try allocator.alloc(f64, node_space.numNodes());
    defer allocator.free(field);

    var cells = cell_space.cartesianIndices();

    while (cells.next()) |cell| {
        const pos = domain.transformPos(node_space.cellPosition(cell));

        node_space.setValue(cell, field, @sin(pos[0]) * @sin(pos[1]));
    }

    // ********************************
    // Set exact values ***************

    const exact: []f64 = try allocator.alloc(f64, node_space.numNodes());
    defer allocator.free(exact);

    var nodes = node_space.nodes(2);

    while (nodes.next()) |node| {
        const pos = domain.transformPos(node_space.nodePosition(node));

        node_space.setNodeValue(node, exact, @sin(pos[0]) * @sin(pos[1]));
    }

    try expect(@fabs(node_space.boundaryOp([2]isize{ -1, 0 }, null, [2]usize{ 0, 50 }, exact)) < 1e-10);
    try expect(@fabs(node_space.boundaryOp([2]isize{ -2, 0 }, null, [2]usize{ 0, 50 }, exact)) < 1e-10);
    try expect(@fabs(node_space.boundaryOp([2]isize{ -1, -1 }, null, [2]usize{ 0, 0 }, exact)) < 1e-10);
    try expect(@fabs(node_space.boundaryOp([2]isize{ -2, -2 }, null, [2]usize{ 0, 0 }, exact)) < 1e-10);

    // *********************************

    const DiritchletBC = struct {
        pub fn kind(_: usize) BoundaryKind {
            return .robin;
        }

        pub fn condition(_: @This(), _: [2]f64, _: FaceIndex) Robin {
            return Robin.diritchlet(0.0);
        }
    };

    node_space.fillBoundary(DiritchletBC{}, field);

    // **********************************

    const window = IndexBox{
        .origin = .{ 0, 0 },
        .size = .{ 4, 4 },
    };

    const field_window: []f64 = try allocator.alloc(f64, 16);
    defer allocator.free(field_window);

    node_space.indexSpace().copyWindow(window, f64, field_window, field);

    const exact_window: []f64 = try allocator.alloc(f64, 16);
    defer allocator.free(exact_window);

    node_space.indexSpace().copyWindow(window, f64, exact_window, exact);

    for (0..16) |i| {
        const diff = field_window[i] - exact_window[i];
        // std.debug.print("diff {}\n", .{diff});
        try expect(@fabs(diff) < 1e-10);
    }
}

test {
    _ = boundary;
}
