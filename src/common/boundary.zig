const std = @import("std");
const assert = std.debug.assert;

const geometry = @import("../geometry/geometry.zig");
const utils = @import("../utils.zig");

const Range = utils.Range;

const nodes_ = @import("nodes.zig");
const traits = @import("traits.zig");

/// Describes a the type of boundary condition
/// to be used to fill ghost cells on a given axis.
pub const BoundaryKind = enum {
    /// The function is either symmetric or antisymmetric across the boundary.
    symmetric,
    /// The function satisfies some set of robin-style boundary
    /// conditions at the boundary.
    robin,
    /// Boundary values are extrapolated from existed data. This should only be used in conjunction with
    /// some other boundary condition (sommerfeld for instance).
    extrapolate,
};

/// Describes the polarity of a symmetric boundary conditions.
pub const BoundaryPolarity = enum {
    /// An antisymmetric boundary.
    odd,
    /// A symmetric boundary.
    even,
};

/// A utility wrapper around a node space which handles boundary conditions.
pub fn BoundaryEngine(comptime N: usize, comptime M: usize, comptime Set: type) type {
    traits.checkBoundarySet(N, Set);

    return struct {
        space: NodeSpace,
        range: Range,
        set: Set,

        const AxisMask = geometry.AxisMask(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexMixin = geometry.IndexMixin(N);
        const Region = geometry.Region(N);
        const NodeOperator = nodes_.NodeOperator(N);
        const NodeSpace = nodes_.NodeSpace(N, M);

        pub fn fill(self: @This(), comptime O: usize, field: []f64) void {
            const regions = comptime Region.enumerateOrdered();

            inline for (comptime regions[1..]) |region| {
                self.fillRegion(O, region, AxisMask.initFull(), field);
            }
        }

        pub fn fillRegion(self: @This(), comptime O: usize, comptime region: Region, comptime mask: AxisMask, field: []f64) void {
            comptime var nmask: AxisMask = AxisMask.initEmpty();
            comptime var kind: BoundaryKind = .symmetric;

            comptime {
                const mregion = region.masked(mask);

                var priority: usize = 0;

                for (mregion.adjacentFaces()) |face| {
                    const boundary_id = Set.boundaryIdFromFace(face);
                    const BoundaryType: type = traits.BoundaryTypeFromId(Set, boundary_id);

                    if (BoundaryType.priority >= priority) {
                        priority = BoundaryType.priority;
                        kind = BoundaryType.kind;
                    }
                }

                for (mregion.adjacentFaces()) |face| {
                    const boundary_id = Set.boundaryIdFromFace(face);
                    const BoundaryType: type = traits.BoundaryTypeFromId(Set, boundary_id);

                    if (BoundaryType.priority == priority) {
                        assert(BoundaryType.kind == kind);

                        nmask.set(face.axis);
                    }
                }
            }

            self.fillRegionKind(O, region, nmask, kind, field);
        }

        pub fn fillRegionKind(
            self: @This(),
            comptime O: usize,
            comptime region: Region,
            comptime mask: AxisMask,
            comptime kind: BoundaryKind,
            field: []f64,
        ) void {
            // Did someone say comptime?
            @setEvalBranchQuota(10000);

            comptime {
                if (O > M) {
                    @compileError("O must be <= M.");
                }

                // Short circuit if the region does not actually have any boundary nodes
                if (region.adjacency() == 0 or mask.isEmpty()) {
                    return;
                }
            }

            // Masked version of the region. Used for wonderfully ergonomic ways of (ab)using the regions API
            // to properly handle node space corners which do not actually touch the domain corner.
            const mregion = comptime region.masked(mask);
            const oregion = comptime region.masked(mask.complement());

            // Loop over the cells touching this face.
            var inner_face_cells = region.innerFaceCells(self.space.size);

            while (inner_face_cells.next()) |cell| {
                // Some functional patterns perhaps?
                comptime var offsets = oregion.extentOffsets(O);

                inline while (comptime offsets.next()) |off| {
                    // This is the central node, off of which we are filling ghost nodes.
                    const node = IndexMixin.addSigned(off, IndexMixin.toSigned(cell));

                    // if (comptime kind == .symmetric) {
                    //     var is_odd = false;

                    //     inline for (comptime mregion.adjacentFaces()) |face| {
                    //         const boundary = traits.boundaryFromId(self.set, Set.boundaryIdFromFace(face));

                    //         // Flip sign if we have odd polarity
                    //         if (boundary.polarity == .odd) {
                    //             is_odd = true;
                    //         }
                    //     }

                    //     if (is_odd) {
                    //         self.space.setValue(node, field[self.range.start..self.range.end], 0.0);
                    //     }
                    // }

                    // Loop over extends
                    comptime var extents = mregion.extentOffsets(O);

                    inline while (comptime extents.next()) |extent| {
                        self.fillGhostNode(
                            O,
                            mregion,
                            kind,
                            extent,
                            node,
                            field,
                        );
                    }
                }
            }
        }

        /// Here there be dragons.
        ///
        /// This is a chaotic and fantastic jugle of comptime, all used for generating specialized
        /// functions for filling a specific kind of boundary condition in a given masked region,
        /// to a given order, out to a certain extent, all in a certain dimension.
        fn fillGhostNode(
            self: @This(),
            comptime O: usize,
            comptime region: Region,
            comptime kind: BoundaryKind,
            comptime extent: [N]isize,
            node: [N]isize,
            field: []f64,
        ) void {
            // Get proper slice of field.
            const bfield = field[self.range.start..self.range.end];
            // Get value of function at node.
            const f = self.space.value(node, bfield);

            // Target node
            const target = IndexMixin.addSigned(node, extent);

            // Set target to zero
            self.space.setValue(target, bfield, 0.0);

            switch (kind) {
                .symmetric => {
                    var source: [N]isize = target;
                    var sign: bool = true;

                    inline for (comptime region.adjacentFaces()) |face| {
                        const boundary = traits.boundaryFromId(self.set, Set.boundaryIdFromFace(face));

                        // Flip source across axis
                        const axis = face.axis;

                        source[axis] = node[axis] - extent[axis];

                        // Flip sign if we have odd polarity
                        if (boundary.polarity == .odd) {
                            sign = !sign;
                        }
                    }

                    const value = self.space.value(source, bfield);
                    self.space.setValue(target, bfield, if (sign) value else -value);
                },
                .robin => {
                    // Relavent axes
                    comptime var axes: [N]usize = [1]usize{0} ** N;
                    // Signs of each normal derivative.
                    comptime var signs: [N]f64 = undefined;
                    // Sign of overall mixed derivative.
                    comptime var sign: f64 = 1.0;
                    // Overall mixed derivative stencil.
                    comptime var stencil: NodeOperator = NodeOperator.value();

                    comptime {
                        var cur: usize = 0;

                        for (0..N) |axis| {
                            if (extent[axis] != 0) {
                                axes[cur] = axis;
                                cur += 1;
                            }

                            signs[axis] = if (extent[axis] >= 0) 1.0 else -1.0;

                            if (extent[axis] < 0) {
                                sign *= -1.0;
                            }

                            stencil.setAxisBoundaryDerivative(axis, O, extent[axis]);
                        }
                    }

                    // Compute target value for mixed derivative.
                    const normal = switch (comptime region.adjacency()) {
                        1 => blk: {
                            const axis = axes[0];
                            const face = comptime region.faceFromAxis(axis);
                            const boundary = traits.boundaryFromId(self.set, Set.boundaryIdFromFace(face));

                            const alpha = fieldValue(self, node, boundary.robin_value);
                            const beta = fieldValue(self, node, boundary.robin_rhs);

                            break :blk alpha * f + beta;
                        },
                        2 => blk: {
                            comptime var s0 = NodeOperator.value();
                            comptime var s1 = NodeOperator.value();
                            s0.setAxisCentered(axes[0], O, .derivative);
                            s1.setAxisCentered(axes[1], O, .derivative);

                            const sign0: f64 = signs[axes[0]];
                            const sign1: f64 = signs[axes[1]];

                            const face0 = comptime region.faceFromAxis(axes[0]);
                            const face1 = comptime region.faceFromAxis(axes[1]);
                            const boundary0 = traits.boundaryFromId(self.set, Set.boundaryIdFromFace(face0));
                            const boundary1 = traits.boundaryFromId(self.set, Set.boundaryIdFromFace(face1));

                            const alpha0 = self.fieldValue(node, boundary0.robin_value);
                            const alpha1 = self.fieldValue(node, boundary1.robin_value);
                            const beta0 = self.fieldValue(node, boundary0.robin_rhs);
                            const beta1 = self.fieldValue(node, boundary1.robin_rhs);

                            const alpha0_1 = sign1 * self.fieldEval(s1, node, boundary0.robin_value);
                            const alpha1_0 = sign0 * self.fieldEval(s0, node, boundary1.robin_value);
                            const beta0_1 = sign1 * self.fieldEval(s1, node, boundary0.robin_rhs);
                            const beta1_0 = sign0 * self.fieldEval(s0, node, boundary1.robin_rhs);

                            break :blk (alpha0_1 + alpha1_0 + alpha0 * alpha1) * f + alpha0 * beta1 + alpha1 * beta0 + beta0_1 + beta1_0;
                        },
                        else => @compileError("Robin boundary conditions only supported for N <= 2"),
                    };

                    // Current value of mixed stencil.
                    const value = sign * self.space.eval(stencil, node, bfield);
                    // Coefficient for mixed stencil.
                    const coef = sign * self.space.evalCoef(stencil, extent);
                    // Set target value.
                    self.space.setValue(target, bfield, (normal - value) / coef);
                },
                .extrapolate => {
                    // Extrapolation stencil.
                    const stencil: NodeOperator = comptime NodeOperator.extrapolate(O, extent);
                    // Apply stencil.
                    const value = self.space.eval(stencil, node, bfield);
                    // Set target value.
                    self.space.setValue(target, bfield, value);
                },
            }
        }

        // Retrieves the value of a field (either numerical or analytic) at the given vertex.
        fn fieldValue(self: @This(), vertex: [N]isize, field: anytype) f64 {
            const Field = @TypeOf(field);

            if (comptime Field == []const f64 or Field == []f64) {
                return self.space.value(vertex, field[self.range.start..self.range.end]);
            } else if (comptime traits.isAnalyticField(N, Field)) {
                const pos = self.space.position(vertex);
                return field.eval(pos);
            } else {
                @compileError("Unexpected field type");
            }
        }

        fn fieldEval(self: @This(), comptime stencil: NodeOperator, vertex: [N]isize, field: anytype) f64 {
            const Field = @TypeOf(field);

            if (comptime Field == []const f64 or Field == []f64) {
                return self.space.eval(stencil, vertex, field[self.range.start..self.range.end]);
            } else if (comptime traits.isAnalyticField(N, Field)) {
                return self.space.evalAnalytic(stencil, vertex, field);
            } else {
                @compileError("Unexpected field type");
            }
        }
    };
}

test "2d boundary filling" {
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;
    const pi = std.math.pi;

    const FaceIndex = geometry.FaceIndex(2);
    const NodeSpace = nodes_.NodeSpace(2, 2);
    // const NodeOperator = nodes_.NodeOperator(2);
    const NuemannBoundary = traits.NuemannBoundary(2);

    const NuemannSet = struct {
        pub const card: usize = 1;

        pub fn boundaryIdFromFace(_: FaceIndex) usize {
            return 0;
        }

        pub const BoundaryType0: type = NuemannBoundary;

        pub fn boundary0(_: @This()) BoundaryType0 {
            return .{};
        }
    };

    const Engine = BoundaryEngine(2, 2, NuemannSet);

    const node_space: NodeSpace = .{
        .size = [2]usize{ 100, 100 },
        .bounds = .{
            .origin = .{ 0.0, 0.0 },
            .size = .{ pi, pi },
        },
    };

    // *********************************
    // Set field values ****************

    const field: []f64 = try allocator.alloc(f64, node_space.numNodes());
    defer allocator.free(field);

    {
        var nodes = node_space.nodes(0);

        while (nodes.next()) |node| {
            const pos = node_space.position(node);
            node_space.setValue(node, field, @cos(pos[0]) * @cos(pos[1]));
        }
    }

    // ********************************
    // Set exact values ***************

    const exact: []f64 = try allocator.alloc(f64, node_space.numNodes());
    defer allocator.free(exact);

    {
        var nodes = node_space.nodes(2);

        while (nodes.next()) |node| {
            const pos = node_space.position(node);
            node_space.setValue(node, exact, @cos(pos[0]) * @cos(pos[1]));
        }
    }

    // comptime var left_operator1: NodeOperator = NodeOperator.value();
    // comptime var left_operator2: NodeOperator = NodeOperator.value();
    // comptime var right_operator1: NodeOperator = NodeOperator.value();
    // comptime var right_operator2: NodeOperator = NodeOperator.value();

    // comptime {
    //     left_operator1.setAxisBoundaryDerivative(0, 2, -1);
    //     left_operator2.setAxisBoundaryDerivative(0, 2, -2);
    //     right_operator1.setAxisBoundaryDerivative(0, 2, 1);
    //     right_operator2.setAxisBoundaryDerivative(0, 2, 2);
    // }

    // try expect(@abs(node_space.order(2).boundaryOp([2]isize{ -1, 0 }, null, [2]isize{ 0, 50 }, exact)) < 1e-10);
    // try expect(@abs(node_space.order(2).boundaryOp([2]isize{ -2, 0 }, null, [2]isize{ 0, 50 }, exact)) < 1e-10);
    // try expect(@abs(node_space.order(2).boundaryOp([2]isize{ -1, -1 }, null, [2]isize{ 0, 0 }, exact)) < 1e-10);
    // try expect(@abs(node_space.order(2).boundaryOp([2]isize{ -2, -2 }, null, [2]isize{ 0, 0 }, exact)) < 1e-10);

    // *********************************

    const engine = Engine{ .space = node_space, .set = NuemannSet{}, .range = .{
        .start = 0,
        .end = node_space.numNodes(),
    } };

    engine.fill(2, field);

    // **********************************
    // Test that boundary values are within a certain bound of the exact value

    for (0..node_space.numNodes()) |i| {
        const err = @abs(field[i] - exact[i]);
        try expect(err < 1e-8);

        // if (err > 1e-8) {
        //     std.debug.print("Failed at {} with error {}\n", .{ i, err });
        // }
    }
}

// test "mixed boundary filling" {
//     const expect = std.testing.expect;
//     const allocator = std.testing.allocator;

//     const FaceIndex = geometry.FaceIndex(1);
//     const NodeSpace = nodes_.NodeSpace(1, 2);
//     const Engine = BoundaryEngine(1, 2, 2);

//     const node_space: NodeSpace = .{
//         .size = [1]usize{100},
//         .bounds = .{
//             .origin = .{0.0},
//             .size = .{1.0},
//         },
//     };
//     const cell_space = node_space.cellSpace();

//     // *********************************
//     // Set field values ****************

//     const field: []f64 = try allocator.alloc(f64, node_space.numNodes());
//     defer allocator.free(field);

//     @memset(field, 0.0);

//     {
//         var cells = cell_space.cartesianIndices();

//         while (cells.next()) |cell| {
//             const pos = node_space.cellPosition(cell);
//             node_space.setValue(cell, field, pos[0] * pos[0]);
//         }
//     }

//     // ********************************
//     // Set exact values ***************

//     const exact: []f64 = try allocator.alloc(f64, node_space.numNodes());
//     defer allocator.free(exact);

//     {
//         var nodes = node_space.nodes(2);

//         while (nodes.next()) |node| {
//             const pos = node_space.nodePosition(node);
//             node_space.setNodeValue(node, exact, pos[0] * pos[0]);
//         }
//     }

//     const MixedBC = struct {
//         pub fn kind(_: FaceIndex) BoundaryKind {
//             return .robin;
//         }

//         pub fn robin(_: @This(), pos: [1]f64, face: FaceIndex) Robin {
//             const x = pos[0];

//             if (face.side == false) {
//                 return Robin.diritchlet(0.0);
//             } else {
//                 return Robin.nuemann(2.0 * x);
//             }
//         }
//     };

//     Engine.new(node_space).fill(MixedBC{}, field);

//     // **********************************
//     // Test that boundary values are within a certain bound of the exact value

//     for (0..node_space.numNodes()) |i| {
//         try expect(@abs(field[i] - exact[i]) < 1e-10);
//     }
// }

// test "masked boundary filling" {
//     const expect = std.testing.expect;
//     const allocator = std.testing.allocator;
//     const pi = std.math.pi;

//     const AxisMask = geometry.AxisMask(2);
//     const FaceIndex = geometry.FaceIndex(2);
//     const Region = geometry.Region(2);
//     const NodeSpace = nodes_.NodeSpace(2, 2);
//     const Engine = BoundaryEngine(2, 2, 2);

//     const node_space: NodeSpace = .{
//         .size = [2]usize{ 100, 100 },
//         .bounds = .{
//             .origin = .{ 0.0, 0.0 },
//             .size = .{ 1.0, 1.0 },
//         },
//     };

//     const engine = Engine.new(node_space);

//     // ********************************
//     // Set field values ***************

//     const exact: []f64 = try allocator.alloc(f64, node_space.numNodes());
//     defer allocator.free(exact);

//     {
//         var nodes = node_space.nodes(2);

//         while (nodes.next()) |node| {
//             const pos = node_space.nodePosition(node);
//             const v = @sin(pi * pos[0]) * pos[1];
//             node_space.setNodeValue(node, exact, v);
//         }
//     }

//     const field: []f64 = try allocator.alloc(f64, node_space.numNodes());
//     defer allocator.free(field);

//     @memcpy(field, exact);

//     // *********************************

//     const DiritchletBC = struct {
//         pub fn kind(_: FaceIndex) BoundaryKind {
//             return .robin;
//         }

//         pub fn robin(_: @This(), _: [2]f64, _: FaceIndex) Robin {
//             return Robin.diritchlet(0.0);
//         }
//     };

//     const region: Region = .{
//         .sides = .{ .right, .left },
//     };

//     const mask = comptime blk: {
//         var result = AxisMask.initEmpty();
//         result.set(0);
//         break :blk result;
//     };

//     engine.fillRegion(region, mask, DiritchletBC{}, field);

//     // **********************************
//     // Test that boundary values are within a certain bound of the exact value

//     const index_space = node_space.indexSpace();

//     var nodes = node_space.nodes(2);

//     while (nodes.next()) |node| {
//         const lin = index_space.linearFromCartesian(NodeSpace.indexFromNode(node));
//         const err = @abs(field[lin] - exact[lin]);

//         // if (err > 1e-10) {
//         //     std.debug.print("Error at {any} is {}\n", .{ node, err });
//         // }
//         try expect(err < 1e-10);
//     }
// }
