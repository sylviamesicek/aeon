const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;
const panic = std.debug.panic;
const hasFn = std.meta.hasFn;

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const lac = @import("../lac/lac.zig");
const utils = @import("../utils.zig");
const Range = utils.Range;

const engine_ = @import("engine.zig");
const traits = @import("traits.zig");

const checkFunction = traits.checkFunction;

pub const NodeOperatorRank = union(enum) {
    value,
    derivative,
    second_derivative,
    extrapolate: isize,
};

/// Represents an N-dimensional stencil product approximating some combination of
/// differential operators applied along each axis.
pub fn NodeOperator(comptime N: usize) type {
    return struct {
        left: [N]usize,
        right: [N]usize,
        ranks: [N]NodeOperatorRank,

        const Stencils = basis.Stencils;

        /// Returns the operator that simply returns the value of the field at the point.
        pub fn value() @This() {
            return .{
                .ranks = [1]NodeOperatorRank{.value} ** N,
                .left = [1]usize{0} ** N,
                .right = [1]usize{0} ** N,
            };
        }

        pub fn centered(order: usize, ranks: [N]NodeOperatorRank) @This() {
            return .{
                .ranks = ranks,
                .left = [1]usize{order} ** N,
                .right = [1]usize{order} ** N,
            };
        }

        pub fn extrapolate(order: usize, extent: [N]isize) @This() {
            const result = value();

            for (0..N) |axis| {
                if (extent[axis] > 0) {
                    result.left[axis] = 2 * order;
                    result.right[axis] = 0;
                    result.ranks[axis] = .{ .extrapolate = extent[axis] };
                } else if (extent[axis] < 0) {
                    result.right[axis] = 2 * order;
                    result.left[axis] = 0;
                    result.ranks[axis] = .{ .extrapolate = extent[axis] };
                }
            }

            return result;
        }

        pub fn setAxisCentered(self: *@This(), axis: usize, order: usize, rank: NodeOperatorRank) void {
            self.ranks[axis] = rank;
            self.left[axis] = order;
            self.right[axis] = order;
        }

        pub fn setAxisBoundaryDerivative(self: *@This(), axis: usize, order: usize, extent: isize) void {
            if (extent == 0) {
                self.left[axis] = 0;
                self.right[axis] = 0;
                self.ranks[axis] = .value;
            } else if (extent > 0) {
                self.right[axis] = @abs(extent);
                self.left[axis] = 2 * order;
                self.ranks[axis] = .derivative;
            } else {
                self.left[axis] = @abs(extent);
                self.right[axis] = 2 * order;
                self.ranks[axis] = .derivative;
            }
        }

        pub fn maxSupportSize(self: @This()) usize {
            var result: usize = 0;

            for (0..N) |axis| {
                result = @max(result, self.left[axis] + self.right[axis] + 1);
            }

            return result;
        }

        pub fn stencilSizes(comptime self: @This()) [N]usize {
            var result: [N]usize = undefined;

            for (0..N) |axis| {
                result[axis] = self.left[axis] + 1 + self.right[axis];
            }

            return result;
        }

        pub fn stencils(comptime self: @This()) [N][self.maxSupportSize()]f64 {
            var result: [N][self.maxSupportSize()]f64 = undefined;

            for (0..N) |axis| {
                const size = self.left[axis] + self.right[axis] + 1;
                const stencil = switch (self.ranks[axis]) {
                    .value => Stencils.value(self.left[axis], self.right[axis]),
                    .derivative => Stencils.derivative(self.left[axis], self.right[axis]),
                    .second_derivative => Stencils.secondDerivative(self.left[axis], self.right[axis]),
                    .extrapolate => |off| Stencils.extrapolate(self.left[axis], self.right[axis], off),
                };

                @memcpy(result[axis][0..size], stencil);
            }

            return result;
        }
    };
}

/// Represents a single uniform block of nodes, along with ghost nodes along
/// each boundary. Provides routines for applying centered stencils of order `2M + 1` to
/// each node in the block. Nodes are indexed by a `[N]isize` where the isize on each axis
/// may range from (-M, size - 1 + M).
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

        // *******************************
        // Vertices

        /// Returns the number of vertices along each axis for this node space.
        pub fn vertexSize(self: Self) [N]usize {
            return self.size;
        }

        // *******************************
        // Nodes

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
                result[i] = @floatFromInt(node[i]);
                result[i] /= @floatFromInt(self.size[i] - 1);
            }

            return self.bounds.localToGlobal(result);
        }

        /// Returns an index space over the entire node space.
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
        pub fn value(self: Self, node: [N]isize, field: []const f64) f64 {
            const linear = self.indexSpace().linearFromCartesian(indexFromNode(node));
            return field[linear];
        }

        /// Sets the value of a field at a given node.
        pub fn setValue(self: Self, node: [N]isize, field: []f64, v: f64) void {
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

        /// Iterates all nodes in the `NodeSpace`, including `F` layers of ghost nodes.
        pub fn nodes(self: Self, comptime F: usize) NodeIterator(F) {
            return NodeIterator(F).init(self.size);
        }

        // *********************************
        // Prolong/Restriction *************
        // *********************************

        /// Prolongs values to the given supervertex.
        pub fn prolong(self: @This(), comptime O: usize, supernode: [N]isize, field: []const f64) f64 {
            assert(field.len == self.numNodes());

            // Compute parity of super vertex.
            var parity: AxisMask = AxisMask.initEmpty();

            for (0..N) |axis| {
                parity.setValue(axis, @mod(supernode[axis], 2) == 1);
            }

            return switch (parity) {
                inline else => |val| self.prolongParity(O, val, supernode, field),
            };
        }

        /// Prolong values to the given supervertex assuming
        /// a known parity.
        fn prolongParity(self: @This(), comptime O: usize, comptime parity: AxisMask, supernode: [N]isize, field: []const f64) f64 {
            // Generate Prolongation stencil
            const pstencil: [2 * O]f64 = comptime Stencils.prolong(O);

            // Accumulate result to this variable
            var result: f64 = 0.0;

            // Find central node for applying stencil
            var central: [N]isize = undefined;

            for (0..N) |i| {
                central[i] = @divFloor(supernode[i], 2);
            }

            // Loop over stencil space.
            comptime var stencil_sizes: [N]usize = [1]usize{1} ** N;

            inline for (0..N) |axis| {
                if (comptime parity.isSet(axis)) {
                    stencil_sizes[axis] = 2 * O;
                }
            }

            comptime var indices = IndexSpace.fromSize(stencil_sizes).cartesianIndices();

            inline while (comptime indices.next()) |index| {
                const sindex = comptime toSigned(index);

                comptime var offset: [N]isize = [1]isize{0} ** N;
                comptime var coef: f64 = 1.0;

                inline for (0..N) |axis| {
                    if (comptime parity.isSet(axis)) {
                        coef *= pstencil[index[axis]];
                        offset[axis] = sindex[axis] - O + 1;
                    }
                }

                // Short circuit in comptime
                if (comptime coef == 0.0 or coef == -0.0) {
                    continue;
                }

                const node = IndexMixin.addSigned(central, offset);
                result += coef * self.value(node, field);
            }

            return result;
        }

        /// Performs direct restriction (i.e. injection).
        pub fn restrict(self: @This(), subnode: [N]isize, field: []const f64) f64 {
            var node: [N]isize = undefined;

            for (0..N) |i| {
                node[i] = 2 * subnode[i];
            }

            return self.value(node, field);
        }

        /// Performs restriction with full weighting (i.e. the restriction compatible with the prolongation operator
        /// of the same order).
        pub fn restrictFull(self: @This(), comptime O: usize, subnode: [N]isize, field: []const f64) f64 {
            const stencil: [2 * O + 1]f64 = comptime Stencils.restrict(O);

            // Accumulate result to this variable
            var result: f64 = 0.0;
            // Find central node for applying stencil
            var central: [N]isize = undefined;

            for (0..N) |i| {
                central[i] = 2 * subnode[i];
            }

            // Iterate over stencil space
            comptime var indices = IndexSpace.fromSize(IndexMixin.splat(2 * O + 1)).cartesianIndices();

            inline while (comptime indices.next()) |index| {
                const sindex = comptime toSigned(index);

                // Compute a coefficient for this index at comptime.
                comptime var coef: f64 = 1.0;
                comptime var offset: [N]isize = undefined;

                inline for (0..N) |axis| {
                    coef *= stencil[index[axis]];
                    offset[axis] = sindex[axis] - O;
                }

                // Short circuit in comptime
                if (comptime coef == 0.0 or coef == -0.0) {
                    continue;
                }

                const node = IndexMixin.addSigned(central, offset);
                result += coef * self.value(node, field);
            }

            return result;
        }

        // *********************************
        // Dissispation ********************
        // *********************************

        /// Apply dissipation at the given node.
        pub fn dissipation(self: @This(), comptime O: usize, node: [N]isize, field: []const f64) f64 {
            var result: f64 = 0.0;

            for (0..N) |axis| {
                result += self.dissipationAxis(O, axis, node, field);
            }

            return result;
        }

        fn dissipationAxis(self: @This(), comptime O: usize, comptime axis: usize, node: [N]isize, field: []const f64) f64 {
            const stencil = Stencils.dissipation(O);

            var result: f64 = 0.0;

            inline for (0..2 * O + 1) |i| {
                const offset: isize = @as(isize, @intCast(i)) - O;

                var offset_node = node;
                offset_node[axis] += offset;

                result += stencil[i] * self.value(offset_node, field);
            }

            var scale: f64 = @floatFromInt(self.size[axis]);
            scale /= self.bounds.size[axis];

            return scale * result;
        }

        // ******************************
        // Operators ********************
        // ******************************

        /// Evaluates an operator at the given vertex.
        pub fn eval(self: @This(), comptime op: NodeOperator(N), vertex: [N]isize, field: []const f64) f64 {
            // Accumulate result
            var result: f64 = 0.0;

            const stencil_sizes = op.stencilSizes();
            const stencils = op.stencils();

            comptime var indices = IndexSpace.fromSize(stencil_sizes).cartesianIndices();

            inline while (indices.next()) |index| {
                const sindex = toSigned(index);

                comptime var offset: [N]isize = undefined;
                comptime var coef: f64 = 1.0;

                inline for (0..N) |axis| {
                    coef *= stencils[axis][index[axis]];
                    offset[axis] = sindex[axis] - op.left[axis];
                }

                // Short circuit in comptime
                if (comptime coef == 0.0 or coef == -0.0) {
                    continue;
                }

                const node = IndexMixin.addSigned(vertex, offset);
                result += coef * self.value(node, field);
            }

            // Covariantly transform result
            inline for (0..N) |i| {
                var scale: f64 = @floatFromInt(self.size[i]);
                scale /= self.bounds.size[i];

                inline for (0..op.ranks[i]) |_| {
                    result *= scale;
                }
            }

            return result;
        }

        /// Evaluates an operator at the given vertex by lazily applying a function.
        pub fn evalAnalytic(self: @This(), comptime op: NodeOperator(N), vertex: [N]isize, field: anytype) f64 {
            const Field: type = @TypeOf(field);
            // Check field satisfies isAnalyticField trait.
            traits.checkAnalyticField(N, Field);

            // Central Vertex
            const central = toSigned(vertex);
            // Accumulate result
            var result: f64 = 0.0;

            const stencil_sizes = op.stencilSizes();
            const stencils = op.stencils();

            comptime var indices = IndexSpace.fromSize(stencil_sizes).cartesianIndices();

            inline while (indices.next()) |index| {
                const sindex = toSigned(index);

                comptime var offset: [N]isize = undefined;
                comptime var coef: f64 = 1.0;

                inline for (0..N) |axis| {
                    coef *= stencils[axis][index[axis]];
                    offset[axis] = sindex[axis] - op.left[axis];
                }

                // Short circuit in comptime
                if (comptime coef == 0.0 or coef == -0.0) {
                    continue;
                }

                const node = IndexMixin.addSigned(central, offset);
                result += coef * field.eval(self.nodePosition(node));
            }

            // Covariantly transform result
            inline for (0..N) |i| {
                var scale: f64 = @floatFromInt(self.size[i]);
                scale /= self.bounds.size[i];

                inline for (0..op.ranks[i]) |_| {
                    result *= scale;
                }
            }

            return result;
        }

        /// Computes the coefficient for a given operator at a particular index.
        pub fn evalCoef(self: @This(), comptime op: NodeOperator(N), comptime index: [N]isize) f64 {
            comptime var coef: f64 = 1.0;

            comptime {
                for (0..N) |axis| {
                    const stencil = switch (op.ranks[axis]) {
                        0 => Stencils.value(op.left[axis], op.right[axis]),
                        1 => Stencils.derivative(op.left[axis], op.right[axis]),
                        2 => Stencils.secondDerivative(op.left[axis], op.right[axis]),
                        else => @compileError("Only ranks <= 2 are supported."),
                    };

                    assert(index[axis] >= -op.left[axis]);
                    assert(index[axis] <= op.right[axis]);

                    const i: usize = @intCast(index[axis] + op.left[axis]);

                    coef *= stencil[i];
                }
            }

            var result: f64 = coef;

            // Covariantly transform result
            inline for (0..N) |i| {
                var scale: f64 = @floatFromInt(self.size[i]);
                scale /= self.bounds.size[i];

                inline for (0..op.ranks[i]) |_| {
                    result *= scale;
                }
            }

            return result;
        }
    };
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

// test "node space restriction and prolongation" {
//     const expect = std.testing.expect;
//     const allocator = std.testing.allocator;

//     const N = 2;
//     const M = 1;
//     const Nodes = NodeSpace(N, M);
//     const RealBox = geometry.RealBox(N);

//     const space: Nodes = .{ .size = .{ 20, 20 }, .bounds = RealBox.unit };

//     const function = try allocator.alloc(f64, space.numNodes());
//     defer allocator.free(function);

//     // **************************
//     // Set function values

//     var nodes = space.nodes(M);

//     while (nodes.next()) |node| {
//         const x: f64 = @floatFromInt(node[0]);
//         const y: f64 = @floatFromInt(node[1]);
//         space.setNodeValue(node, function, x + y);
//     }

//     // **************************
//     // Test restriction

//     const order = space.order(M);

//     try expect(order.restrict(.{ 0, 0 }, function) == 1.0);
//     try expect(order.restrict(.{ 1, 1 }, function) == 5.0);
//     try expect(order.restrict(.{ 2, 2 }, function) == 9.0);

//     // **************************
//     // Test prolongation

//     try expect(order.prolongCell(.{ 0, 0 }, function) == -0.5);
//     try expect(order.prolongCell(.{ 1, 1 }, function) == 0.5);
//     try expect(order.prolongCell(.{ 2, 2 }, function) == 1.5);
// }

// test "node space smoothing" {
//     const expect = std.testing.expect;
//     const allocator = std.testing.allocator;

//     const N = 2;
//     const M = 2;

//     const FaceIndex = geometry.FaceIndex(N);
//     const RealBox = geometry.RealBox(N);
//     const Nodes = NodeSpace(N, M);

//     const Boundary = struct {
//         pub fn kind(_: FaceIndex) BoundaryKind {
//             return .robin;
//         }

//         pub fn robin(_: @This(), _: [2]f64, _: FaceIndex) Robin {
//             return Robin.diritchlet(0.0);
//         }
//     };

//     const Poisson = struct {
//         fn op(nodes: Nodes, cell: [2]usize, field: []const f64) f64 {
//             var result: f64 = 0.0;

//             inline for (0..2) |i| {
//                 comptime var ranks: [2]usize = [1]usize{0} ** 2;
//                 ranks[i] = 2;

//                 result += nodes.order(M).op(ranks, cell, field);
//             }

//             return -result;
//         }

//         fn opDiag(nodes: Nodes) f64 {
//             var result: f64 = 0.0;

//             inline for (0..2) |i| {
//                 comptime var ranks: [2]usize = [1]usize{0} ** 2;
//                 ranks[i] = 2;

//                 result += nodes.order(M).opDiagonal(ranks);
//             }

//             return -result;
//         }
//     };

//     // **************************
//     // Mesh

//     const space: Nodes = .{
//         .size = [2]usize{ 8, 8 },
//         .bounds = RealBox.unit,
//     };

//     // ***************************
//     // Variables

//     const sol: []f64 = try allocator.alloc(f64, space.numNodes());
//     defer allocator.free(sol);

//     const scratch: []f64 = try allocator.alloc(f64, space.numNodes());
//     defer allocator.free(scratch);

//     const rhs: []f64 = try allocator.alloc(f64, space.numNodes());
//     defer allocator.free(rhs);

//     @memset(sol, 0.0);
//     @memset(rhs, 1.0);

//     // ***************************
//     // Perform smoothing

//     var residual: f64 = 0.0;

//     for (0..1000) |iter| {
//         _ = iter;
//         BoundaryEngine(2, 2, 2).new(space).fill(Boundary{}, sol);

//         // Compute residual
//         {
//             residual = 0.0;

//             var cells = space.cellSpace().cartesianIndices();

//             while (cells.next()) |cell| {
//                 const vrhs = space.value(cell, rhs);
//                 const lap = Poisson.op(space, cell, sol);

//                 residual += (vrhs - lap) * (vrhs - lap);
//             }

//             residual = @sqrt(residual);
//         }

//         // Smooth
//         {
//             var cells = space.cellSpace().cartesianIndices();

//             while (cells.next()) |cell| {
//                 const vsol = space.value(cell, sol);
//                 const vrhs = space.value(cell, rhs);
//                 const lap = Poisson.op(space, cell, sol);
//                 const diag = Poisson.opDiag(space);

//                 const v = vsol + 2.0 / 3.0 * (vrhs - lap) / diag;
//                 space.setValue(cell, scratch, v);
//             }

//             @memcpy(sol, scratch);
//         }
//     }

//     try expect(residual <= 10e-12);
// }

// test "node space two grid" {
//     const ArenaAllocator = std.heap.ArenaAllocator;
//     const expect = std.testing.expect;
//     const allocator = std.testing.allocator;

//     const N = 2;
//     const M = 2;

//     const FaceIndex = geometry.FaceIndex(N);
//     const RealBox = geometry.RealBox(N);
//     const BiCGStabSolver = lac.BiCGStabSolver;
//     const Nodes = NodeSpace(N, M);
//     const BoundaryEngine_ = BoundaryEngine(N, M, M);

//     const Boundary = struct {
//         pub fn kind(_: FaceIndex) BoundaryKind {
//             return .robin;
//         }

//         pub fn robin(_: @This(), _: [N]f64, _: FaceIndex) Robin {
//             return Robin.diritchlet(0.0);
//         }
//     };

//     const Poisson = struct {
//         base: Nodes,
//         scratch: []f64,

//         pub fn apply(self: *const @This(), out: []f64, in: []const f64) void {
//             // Copy to scratch
//             {
//                 var cells = self.base.cellSpace().cartesianIndices();
//                 var linear: usize = 0;

//                 while (cells.next()) |cell| : (linear += 1) {
//                     self.base.setValue(cell, self.scratch, in[linear]);
//                 }
//             }

//             BoundaryEngine_.new(self.base).fill(Boundary{}, self.scratch);

//             // Apply
//             {
//                 var cells = self.base.cellSpace().cartesianIndices();
//                 var linear: usize = 0;

//                 while (cells.next()) |cell| : (linear += 1) {
//                     out[linear] = op(self.base, cell, self.scratch);
//                 }
//             }
//         }

//         fn op(nodes: Nodes, cell: [2]usize, field: []const f64) f64 {
//             var result: f64 = 0.0;

//             inline for (0..N) |i| {
//                 comptime var ranks: [2]usize = [1]usize{0} ** 2;
//                 ranks[i] = 2;

//                 result += nodes.order(M).op(ranks, cell, field);
//             }

//             return -result;
//         }

//         fn opDiag(nodes: Nodes) f64 {
//             var result: f64 = 0.0;

//             inline for (0..N) |i| {
//                 comptime var ranks: [2]usize = [1]usize{0} ** 2;
//                 ranks[i] = 2;

//                 result += nodes.order(M).opDiagonal(ranks);
//             }

//             return -result;
//         }
//     };

//     // **************************
//     // Mesh

//     const base: Nodes = .{
//         .size = [2]usize{ 8, 8 },
//         .bounds = RealBox.unit,
//     };

//     const fine: Nodes = .{
//         .size = [2]usize{ 16, 16 },
//         .bounds = RealBox.unit,
//     };

//     // **************************
//     // Variables

//     const base_solution: []f64 = try allocator.alloc(f64, base.numNodes());
//     defer allocator.free(base_solution);

//     const base_old: []f64 = try allocator.alloc(f64, base.numNodes());
//     defer allocator.free(base_old);

//     const base_rhs: []f64 = try allocator.alloc(f64, base.numNodes());
//     defer allocator.free(base_rhs);

//     const base_err: []f64 = try allocator.alloc(f64, base.numNodes());
//     defer allocator.free(base_err);

//     const base_scratch: []f64 = try allocator.alloc(f64, base.numNodes());
//     defer allocator.free(base_scratch);

//     const fine_solution: []f64 = try allocator.alloc(f64, fine.numNodes());
//     defer allocator.free(fine_solution);

//     const fine_res: []f64 = try allocator.alloc(f64, fine.numNodes());
//     defer allocator.free(fine_res);

//     const fine_scratch: []f64 = try allocator.alloc(f64, fine.numNodes());
//     defer allocator.free(fine_scratch);

//     const fine_rhs: []f64 = try allocator.alloc(f64, fine.numNodes());
//     defer allocator.free(fine_rhs);

//     // ****************************
//     // Setup problem

//     @memset(base_solution, 0.0);
//     @memset(fine_solution, 0.0);
//     @memset(fine_rhs, 1.0);

//     // ****************************
//     // Multigrid iteration

//     // Build scratch allocator
//     var arena: ArenaAllocator = ArenaAllocator.init(allocator);
//     defer arena.deinit();

//     const scratch: Allocator = arena.allocator();

//     var residual: f64 = 0.0;

//     for (0..20) |_| {
//         defer _ = arena.reset(.retain_capacity);

//         // Compute current error
//         {
//             residual = 0.0;

//             BoundaryEngine_.new(fine).fill(Boundary{}, fine_solution);

//             var cells = fine.cellSpace().cartesianIndices();

//             while (cells.next()) |cell| {
//                 const rhs = fine.value(cell, fine_rhs);
//                 const lap = Poisson.op(fine, cell, fine_solution);

//                 residual += (rhs - lap) * (rhs - lap);
//             }

//             residual = @sqrt(residual);

//             // std.debug.print("Iteration {}, Residual {}\n", .{ iter, residual });
//         }

//         // Perform presmoothing
//         for (0..10) |_| {
//             BoundaryEngine_.new(fine).fill(Boundary{}, fine_solution);

//             var cells = fine.cellSpace().cartesianIndices();

//             while (cells.next()) |cell| {
//                 const val = fine.value(cell, fine_solution);
//                 const rhs = fine.value(cell, fine_rhs);
//                 const lap = Poisson.op(fine, cell, fine_solution);
//                 const diag = Poisson.opDiag(fine);

//                 const v = val + 2.0 / 3.0 * (rhs - lap) / diag;
//                 fine.setValue(cell, fine_scratch, v);
//             }

//             @memcpy(fine_solution, fine_scratch);
//         }

//         // Residual calculation
//         {
//             BoundaryEngine_.new(fine).fill(Boundary{}, fine_solution);

//             var cells = fine.cellSpace().cartesianIndices();

//             while (cells.next()) |cell| {
//                 const rhs = fine.value(cell, fine_rhs);
//                 const lap = Poisson.op(fine, cell, fine_solution);

//                 fine.setValue(cell, fine_res, rhs - lap);
//             }
//         }

//         // Restrict Solution
//         {
//             BoundaryEngine_.new(fine).fill(Boundary{}, fine_solution);

//             var cells = base.cellSpace().cartesianIndices();

//             while (cells.next()) |cell| {
//                 const sol = fine.order(M).restrict(cell, fine_solution);
//                 base.setValue(cell, base_solution, sol);
//             }
//         }

//         // Right Hand Side computation
//         {
//             BoundaryEngine_.new(base).fill(Boundary{}, base_solution);

//             var cells = base.cellSpace().cartesianIndices();

//             while (cells.next()) |cell| {
//                 const res = fine.order(0).restrict(cell, fine_res);
//                 const val = Poisson.op(base, cell, base_solution);

//                 base.setValue(cell, base_rhs, res + val);
//             }

//             @memcpy(base_old, base_solution);
//         }

//         // ****************************
//         // Base solving

//         const x = try scratch.alloc(f64, base.cellSpace().total());
//         defer scratch.free(x);

//         const b = try scratch.alloc(f64, base.cellSpace().total());
//         defer scratch.free(b);

//         // Transfer from base solution to x and base rhs to b
//         {
//             var cells = base.cellSpace().cartesianIndices();
//             var linear: usize = 0;

//             while (cells.next()) |cell| : (linear += 1) {
//                 x[linear] = base.value(cell, base_solution);
//                 b[linear] = base.value(cell, base_rhs);
//             }
//         }

//         // Solve
//         try BiCGStabSolver.new(1000, 10e-15).solve(scratch, Poisson{
//             .base = base,
//             .scratch = base_scratch,
//         }, x, b);

//         // Transfer from x to base solution
//         {
//             var cells = base.cellSpace().cartesianIndices();
//             var linear: usize = 0;

//             while (cells.next()) |cell| : (linear += 1) {
//                 base.setValue(cell, base_solution, x[linear]);
//             }
//         }

//         // Compute Error
//         {
//             BoundaryEngine_.new(base).fill(Boundary{}, base_solution);
//             BoundaryEngine_.new(base).fill(Boundary{}, base_old);

//             for (0..base.numNodes()) |idx| {
//                 base_err[idx] = base_solution[idx] - base_old[idx];
//             }
//         }

//         // Correct fine solution
//         {
//             var cells = fine.cellSpace().cartesianIndices();

//             while (cells.next()) |cell| {
//                 const sol = fine.value(cell, fine_solution);
//                 const err = base.order(M).prolongCell(cell, base_err);

//                 fine.setValue(cell, fine_solution, sol + err);
//             }
//         }

//         // Post smoothing
//         for (0..10) |_| {
//             BoundaryEngine_.new(fine).fill(Boundary{}, fine_solution);

//             var cells = fine.cellSpace().cartesianIndices();

//             while (cells.next()) |cell| {
//                 const val = fine.value(cell, fine_solution);
//                 const rhs = fine.value(cell, fine_rhs);
//                 const lap = Poisson.op(fine, cell, fine_solution);
//                 const diag = Poisson.opDiag(fine);

//                 const v = val + 2.0 / 3.0 * (rhs - lap) / diag;
//                 fine.setValue(cell, fine_scratch, v);
//             }

//             @memcpy(fine_solution, fine_scratch);
//         }
//     }

//     try expect(residual <= 10e-12);
// }
