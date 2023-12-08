//! This module handles nodes, a kind of "unpacked" representation of a function in FD codes.
//! This module is mesh agnostic, and provides routines to apply physical boundary conditions to
//! nodes, applying tensor products of stencils to node vectors (through the `NodeSpace` type), and
//! an API for applying operators to node vectors. Individual meshes must provide functions for
//! transfering between cell vectors and node vectors.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;
const panic = std.debug.panic;

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const lac = @import("../lac/lac.zig");

// Submodules
const boundary = @import("boundary.zig");

pub const BoundaryKind = boundary.BoundaryKind;
pub const Robin = boundary.Robin;
pub const isBoundary = boundary.isBoundary;

/// A map from blocks (defined in a mesh agnostic way) into `NodeVector`s. This is filled by the appropriate mesh routine.
pub const NodeMap = struct {
    offsets: ArrayList(usize),

    pub fn init(allocator: Allocator) NodeMap {
        const offsets = ArrayList(usize).init(allocator);
        return .{ .offsets = offsets };
    }

    /// Frees a `NodeMap`.
    pub fn deinit(self: *NodeMap) void {
        self.offsets.deinit();
    }

    /// Returns the node offset for a given block.
    pub fn offset(self: NodeMap, block: usize) usize {
        return self.offsets.items[block];
    }

    /// Returns the total number of nodes in a given block.
    pub fn total(self: NodeMap, block: usize) usize {
        return self.offsets.items[block + 1] - self.offsets.items[block];
    }

    /// Returns of slice of nodes for a given block.
    pub fn slice(self: NodeMap, block: usize, nodes: anytype) @TypeOf(nodes) {
        return nodes[self.offsets.items[block]..self.offsets.items[block + 1]];
    }

    /// Returns the number of nodes in the total map.
    pub fn numNodes(self: NodeMap) usize {
        return self.offsets.items[self.offsets.items.len - 1];
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
        const Region = geometry.Region(N);

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
            assert(field.len == self.numNodes());
            return self.nodeValue(toSigned(cell), field);
        }

        /// Sets the value of a field at a certain cell.
        pub fn setValue(self: Self, cell: [N]usize, field: []f64, v: f64) void {
            assert(field.len == self.numNodes());
            self.setNodeValue(toSigned(cell), field, v);
        }

        pub fn prolong(self: Self, subcell: [N]usize, field: []const f64) f64 {
            return self.prolongCell(M, subcell, field);
        }

        /// Prolongs the value of the field to some subcell.
        pub fn prolongCell(self: Self, comptime O: usize, subcell: [N]usize, field: []const f64) f64 {
            if (comptime O > M) {
                @compileError("Order must be less than or equal to number of ghost layers.");
            }

            assert(field.len == self.numNodes());

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

                var offset_node: [N]isize = undefined;

                inline for (0..N) |i| {
                    offset_node[i] = central_node[i] + offset[i];
                }

                // std.debug.print("Coef {}, Offset ({}, {}), Value {}\n", .{ coef, offset[0], offset[1], self.nodeValue(offset_node, field) });

                result += coef * self.nodeValue(offset_node, field);
            }

            return result;
        }

        /// Prolongs the value of the field to some subcell.
        pub fn prolongVertex(self: Self, comptime O: usize, subcell: [N]usize, field: []const f64) f64 {
            if (comptime O > M) {
                @compileError("Order must be less than or equal to number of ghost layers.");
            }

            assert(field.len == self.numNodes());

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

                result += coef * self.nodeValue(offset_node, field);
            }

            return result;
        }

        /// Restricts the value of a field to a supercell.
        pub fn restrict(self: Self, supercell: [N]usize, field: []const f64) f64 {
            return self.restrictOrder(M + 1, supercell, field);
        }

        pub fn restrictOrder(self: Self, comptime O: usize, supercell: [N]usize, field: []const f64) f64 {
            if (comptime O > M + 1) {
                @compileError("Order must be less than ghost extent plus one.");
            }

            assert(field.len == self.numNodes());

            // Compute the stencil and space of indices over which the stencil is defined.
            const stencil: [2 * O]f64 = comptime Stencils(O).restrict();
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
                comptime var offset: [N]isize = undefined;

                inline for (0..N) |i| {
                    const idx: isize = @intCast(stencil_index[i]);

                    coef *= stencil[stencil_index[i]];
                    offset[i] = idx - O + 1;
                }

                var offset_node: [N]isize = undefined;

                inline for (0..N) |i| {
                    offset_node[i] = central_node[i] + offset[i];
                }

                result += coef * self.nodeValue(offset_node, field);

                // std.debug.print("Coef {}, Offset ({}, {}), Value {}\n", .{ coef, offset[0], offset[1], self.nodeValue(offset_node, field) });
            }

            return result;
        }

        // *********************************
        // Operators ***********************
        // *********************************

        /// Computes the result of an operation (specified by the `ranks` argument) at the given
        /// cell, acting on the field defined over the whole nodespace.
        pub fn op(self: Self, comptime ranks: [N]usize, cell: [N]usize, field: []const f64) f64 {
            assert(field.len == self.numNodes());

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
                    const offset: isize = @intCast(stencil_index[i]);
                    offset_node[i] = central_node[i] + offset - M;
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
            assert(field.len == self.numNodes());

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
                    const face = comptime FaceIndex{
                        .side = region.sides[axis] == .right,
                        .axis = axis,
                    };

                    if (@TypeOf(bound).kind(face) == .robin) {
                        robin[axis] = bound.robin(pos, face);
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

                        const face = comptime FaceIndex{
                            .side = region.sides[axis] == .right,
                            .axis = axis,
                        };

                        const kind: BoundaryKind = comptime @TypeOf(bound).kind(face);

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

            assert(field.len == self.numNodes());

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
                        offset_node[i] = @as(isize, @intCast(self.size[i] - 1)) + idx - BM + 1;
                    } else if (comptime extents[i] < 0) {
                        offset_node[i] = @as(isize, @intCast(BM - 1)) - idx;
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

test {
    _ = boundary;
}

test "node space iteration" {
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

test "node space boundaries" {
    const IndexBox = geometry.IndexBox(2);
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
        pub fn kind(_: FaceIndex) BoundaryKind {
            return .robin;
        }

        pub fn robin(_: @This(), _: [2]f64, _: FaceIndex) Robin {
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

test "node space restriction and prolongation" {
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;

    const N = 2;
    const M = 1;
    const Nodes = NodeSpace(N, M);

    const space = Nodes.fromCellSize(.{ 20, 20 });

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

    try expect(space.restrict(.{ 0, 0 }, function) == 1.0);
    try expect(space.restrict(.{ 1, 1 }, function) == 5.0);
    try expect(space.restrict(.{ 2, 2 }, function) == 9.0);

    // **************************
    // Test prolongation

    try expect(space.prolong(.{ 0, 0 }, function) == -0.5);
    try expect(space.prolong(.{ 1, 1 }, function) == 0.5);
    try expect(space.prolong(.{ 2, 2 }, function) == 1.5);
}

test "node space smoothing" {
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;

    const FaceIndex = geometry.FaceIndex(2);
    const RealBox = geometry.RealBox(2);
    const Nodes = NodeSpace(2, 2);

    const Boundary = struct {
        pub fn kind(_: FaceIndex) BoundaryKind {
            return .robin;
        }

        pub fn robin(_: @This(), _: [2]f64, _: FaceIndex) Robin {
            return Robin.diritchlet(0.0);
        }
    };

    const Poisson = struct {
        fn op(domain: RealBox, nodes: Nodes, cell: [2]usize, field: []const f64) f64 {
            var result: f64 = 0.0;

            inline for (0..2) |i| {
                comptime var ranks: [2]usize = [1]usize{0} ** 2;
                ranks[i] = 2;

                result += domain.transformOp(ranks, nodes.op(ranks, cell, field));
            }

            return -result;
        }

        fn opDiag(domain: RealBox, nodes: Nodes) f64 {
            var result: f64 = 0.0;

            inline for (0..2) |i| {
                comptime var ranks: [2]usize = [1]usize{0} ** 2;
                ranks[i] = 2;

                result += domain.transformOp(ranks, nodes.opDiagonal(ranks));
            }

            return -result;
        }
    };

    // **************************
    // Mesh

    const domain: RealBox = .{ .origin = .{ 0.0, 0.0 }, .size = .{ 1.0, 1.0 } };

    const space = Nodes.fromCellSize([2]usize{ 8, 8 });

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
        space.fillBoundary(Boundary{}, sol);

        // Compute residual
        {
            residual = 0.0;

            var cells = space.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const vrhs = space.value(cell, rhs);
                const lap = Poisson.op(domain, space, cell, sol);

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
                const lap = Poisson.op(domain, space, cell, sol);
                const diag = Poisson.opDiag(domain, space);

                const v = vsol + 2.0 / 3.0 * (vrhs - lap) / diag;
                space.setValue(cell, scratch, v);
            }

            @memcpy(sol, scratch);
        }
    }

    try expect(residual <= 10e-12);
}

test "node space multigrid" {
    const ArenaAllocator = std.heap.ArenaAllocator;
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;

    const FaceIndex = geometry.FaceIndex(2);
    const RealBox = geometry.RealBox(2);
    const BiCGStabSolver = lac.BiCGStabSolver;
    const Nodes = NodeSpace(2, 1);

    const Boundary = struct {
        pub fn kind(_: FaceIndex) BoundaryKind {
            return .robin;
        }

        pub fn robin(_: @This(), _: [2]f64, _: FaceIndex) Robin {
            return Robin.diritchlet(0.0);
        }
    };

    const Poisson = struct {
        domain: RealBox,
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

            self.base.fillBoundary(Boundary{}, self.scratch);

            // Apply
            {
                var cells = self.base.cellSpace().cartesianIndices();
                var linear: usize = 0;

                while (cells.next()) |cell| : (linear += 1) {
                    out[linear] = op(self.domain, self.base, cell, self.scratch);
                }
            }
        }

        fn op(domain: RealBox, nodes: Nodes, cell: [2]usize, field: []const f64) f64 {
            var result: f64 = 0.0;

            inline for (0..2) |i| {
                comptime var ranks: [2]usize = [1]usize{0} ** 2;
                ranks[i] = 2;

                result += domain.transformOp(ranks, nodes.op(ranks, cell, field));
            }

            return -result;
        }

        fn opDiag(domain: RealBox, nodes: Nodes) f64 {
            var result: f64 = 0.0;

            inline for (0..2) |i| {
                comptime var ranks: [2]usize = [1]usize{0} ** 2;
                ranks[i] = 2;

                result += domain.transformOp(ranks, nodes.opDiagonal(ranks));
            }

            return -result;
        }
    };

    // **************************
    // Mesh

    const domain: RealBox = .{ .origin = .{ 0.0, 0.0 }, .size = .{ 1.0, 1.0 } };

    const base = Nodes.fromCellSize([2]usize{ 8, 8 });
    const fine = Nodes.fromCellSize([2]usize{ 16, 16 });

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

    for (0..10) |iter| {
        _ = iter;
        defer _ = arena.reset(.retain_capacity);

        // Compute current error
        {
            residual = 0.0;

            fine.fillBoundary(Boundary{}, fine_solution);

            var cells = fine.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const rhs = fine.value(cell, fine_rhs);
                const lap = Poisson.op(domain, fine, cell, fine_solution);

                residual += (rhs - lap) * (rhs - lap);
            }

            residual = @sqrt(residual);

            // std.debug.print("Iteration {}, Residual {}\n", .{ iter, residual });
        }

        // Perform presmoothing
        for (0..10) |_| {
            fine.fillBoundary(Boundary{}, fine_solution);

            var cells = fine.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const val = fine.value(cell, fine_solution);
                const rhs = fine.value(cell, fine_rhs);
                const lap = Poisson.op(domain, fine, cell, fine_solution);
                const diag = Poisson.opDiag(domain, fine);

                const v = val + 2.0 / 3.0 * (rhs - lap) / diag;
                fine.setValue(cell, fine_scratch, v);
            }

            @memcpy(fine_solution, fine_scratch);
        }

        // Residual calculation
        {
            fine.fillBoundary(Boundary{}, fine_solution);

            var cells = fine.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const rhs = fine.value(cell, fine_rhs);
                const lap = Poisson.op(domain, fine, cell, fine_solution);

                fine.setValue(cell, fine_res, rhs - lap);
            }
        }

        // Restrict Solution to old
        {
            fine.fillBoundary(Boundary{}, fine_solution);

            var cells = base.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const sol = fine.restrict(cell, fine_solution);
                base.setValue(cell, base_solution, sol);
            }
        }

        // Right Hand Side computation
        {
            base.fillBoundary(Boundary{}, base_solution);

            var cells = base.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const res = fine.restrictOrder(1, cell, fine_res);
                const val = Poisson.op(domain, base, cell, base_solution);

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
            .domain = domain,
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
            base.fillBoundary(Boundary{}, base_solution);
            base.fillBoundary(Boundary{}, base_old);

            for (0..base.numNodes()) |idx| {
                base_err[idx] = base_solution[idx] - base_old[idx];
            }
        }

        // Correct fine solution
        {
            var cells = fine.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const sol = fine.value(cell, fine_solution);
                const err = base.prolong(cell, base_err);

                fine.setValue(cell, fine_solution, sol + err);
            }
        }

        // Post smoothing
        for (0..100) |_| {
            fine.fillBoundary(Boundary{}, fine_solution);

            var cells = fine.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const val = fine.value(cell, fine_solution);
                const rhs = fine.value(cell, fine_rhs);
                const lap = Poisson.op(domain, fine, cell, fine_solution);
                const diag = Poisson.opDiag(domain, fine);

                const v = val + 2.0 / 3.0 * (rhs - lap) / diag;
                fine.setValue(cell, fine_scratch, v);
            }

            @memcpy(fine_solution, fine_scratch);
        }
    }

    try expect(residual <= 10e-12);
}
