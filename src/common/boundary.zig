const std = @import("std");
const assert = std.debug.assert;

const geometry = @import("../geometry/geometry.zig");

const nodes_ = @import("nodes.zig");

/// Describes a the type of boundary condition
/// to be used to fill ghost cells on a given axis.
pub const BoundaryKind = enum {
    /// The function is symmetric across the boundary.
    even,
    /// The function is anti-symmetric across the boundary.
    odd,
    /// The function satisfies some set of robin-style boundary
    /// conditions at the boundary.
    robin,
};

/// A struct describing a robin boundary.
pub const Robin = struct {
    value: f64,
    flux: f64,
    rhs: f64,

    /// Constructs a diritchlet boundary condition.
    pub fn diritchlet(rhs: f64) Robin {
        return .{
            .value = 1.0,
            .flux = 0.0,
            .rhs = rhs,
        };
    }

    /// Constructs a nuemann boundary condition.
    pub fn nuemann(rhs: f64) Robin {
        return .{
            .value = 0.0,
            .flux = 1.0,
            .rhs = rhs,
        };
    }
};

/// An engine wrapping a node space and a field, providing routines for filling the ghost nodes of the field.
pub fn BoundaryEngine(comptime N: usize, comptime M: usize) type {
    return struct {
        space: NodeSpace,

        const AxisMask = geometry.AxisMask(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexMixin = geometry.IndexMixin(N);
        const Region = geometry.Region(N);
        const NodeSpace = nodes_.NodeSpace(N, M);

        pub fn new(space: NodeSpace) @This() {
            return .{
                .space = space,
            };
        }

        pub fn fill(self: @This(), bound: anytype, field: []f64) void {
            const regions = comptime Region.enumerateOrdered();

            inline for (comptime regions[1..]) |region| {
                self.fillRegion(region, AxisMask.initFull(), bound, field);
            }
        }

        pub fn fillRegion(self: @This(), comptime region: Region, comptime mask: AxisMask, bound: anytype, field: []f64) void {
            const Bound = @TypeOf(bound);

            if (comptime !isBoundary(N)(Bound)) {
                @compileError("Boundary must satisfy isBoundary trait.");
            }

            // Short circuit if the region does not actually have any boundary nodes
            if (comptime region.adjacency() == 0) {
                return;
            }

            // Masked version of the region. Used for wonderfully ergonomic ways of (ab)using the regions API
            // to properly handle node space corners which do not actually touch the domain corner.
            const mregion = comptime region.masked(mask);
            const oregion = comptime region.masked(mask.complement());

            // Loop over the cells touching this face.
            var inner_face_cells = region.innerFaceCells(self.space.size);

            while (inner_face_cells.next()) |cell| {
                // Some functional patterns perhaps?
                comptime var offsets = oregion.extentOffsets(M);

                inline while (comptime offsets.next()) |off| {
                    const node = IndexMixin.addSigned(off, IndexMixin.toSigned(cell));
                    // Find Position
                    const pos: [N]f64 = self.space.nodeOffsetPosition(mregion, node);

                    // Cache robin boundary conditions (if any)
                    var robin: [N]Robin = undefined;

                    inline for (0..N) |axis| {
                        const face = comptime FaceIndex{
                            .side = region.sides[axis] == .right,
                            .axis = axis,
                        };

                        if (Bound.kind(face) == .robin) {
                            robin[axis] = bound.robin(pos, face);
                        }
                    }

                    // Loop over extends
                    comptime var extents = mregion.extentOffsets(M);

                    inline while (comptime extents.next()) |extent| {
                        const target = IndexMixin.addSigned(node, extent);

                        // Set target to zero
                        self.space.setNodeValue(target, field, 0.0);

                        var result: f64 = 0.0;

                        // Accumulate result value
                        inline for (0..N) |axis| {
                            if (comptime mregion.sides[axis] == .middle) {
                                // We need not do anything
                                continue;
                            }

                            const face = comptime FaceIndex{
                                .side = mregion.sides[axis] == .right,
                                .axis = axis,
                            };

                            const kind: BoundaryKind = comptime Bound.kind(face);

                            switch (comptime kind) {
                                .odd, .even => {
                                    var source: [N]isize = target;

                                    if (comptime extent[axis] > 0) {
                                        source[axis] = node[axis] + 1 - extent[axis];
                                    } else {
                                        source[axis] = node[axis] - extent[axis] - 1;
                                    }

                                    const source_value: f64 = self.space.nodeValue(source, field);
                                    const fsign: f64 = comptime if (kind == .odd) -1.0 else 1.0;

                                    result += fsign * source_value;
                                },
                                .robin => {
                                    const vres: f64 = robin[axis].value * self.space.boundaryOp(extent, null, cell, field);
                                    const fres: f64 = robin[axis].flux * self.space.boundaryOp(extent, axis, cell, field);

                                    const vcoef: f64 = robin[axis].value * self.space.boundaryOpCoef(extent, null);
                                    const fcoef: f64 = robin[axis].flux * self.space.boundaryOpCoef(extent, axis);

                                    const rhs: f64 = robin[axis].rhs;

                                    result += (rhs - vres - fres) / (vcoef + fcoef);
                                },
                            }
                        }

                        // Take the average
                        const adj: f64 = @floatFromInt(mregion.adjacency());
                        result /= adj;

                        // Set target to result.
                        self.space.setNodeValue(target, field, result);
                    }
                }
            }
        }
    };
}

/// A trait for defining boundaries.
pub fn isBoundary(comptime N: usize) fn (comptime T: type) bool {
    const FaceIndex = geometry.FaceIndex(N);
    const hasFn = std.meta.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (!(hasFn(T, "kind") and @TypeOf(T.kind) == fn (FaceIndex) BoundaryKind)) {
                return false;
            }

            if (!(hasFn(T, "robin") and @TypeOf(T.robin) == fn (T, [N]f64, FaceIndex) Robin)) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}

test "boundary filling" {
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;
    const pi = std.math.pi;

    const FaceIndex = geometry.FaceIndex(2);
    const NodeSpace = nodes_.NodeSpace(2, 2);
    const Engine = BoundaryEngine(2, 2);

    const node_space: NodeSpace = .{
        .size = [2]usize{ 100, 100 },
        .bounds = .{
            .origin = .{ 0.0, 0.0 },
            .size = .{ pi, pi },
        },
    };
    const cell_space = node_space.cellSpace();

    // *********************************
    // Set field values ****************

    const field: []f64 = try allocator.alloc(f64, node_space.numNodes());
    defer allocator.free(field);

    {
        var cells = cell_space.cartesianIndices();

        while (cells.next()) |cell| {
            const pos = node_space.cellPosition(cell);
            node_space.setValue(cell, field, @sin(pos[0]) * @sin(pos[1]));
        }
    }

    // ********************************
    // Set exact values ***************

    const exact: []f64 = try allocator.alloc(f64, node_space.numNodes());
    defer allocator.free(exact);

    {
        var nodes = node_space.nodes(2);

        while (nodes.next()) |node| {
            const pos = node_space.nodePosition(node);
            node_space.setNodeValue(node, exact, @sin(pos[0]) * @sin(pos[1]));
        }
    }

    try expect(@abs(node_space.boundaryOp([2]isize{ -1, 0 }, null, [2]usize{ 0, 50 }, exact)) < 1e-10);
    try expect(@abs(node_space.boundaryOp([2]isize{ -2, 0 }, null, [2]usize{ 0, 50 }, exact)) < 1e-10);
    try expect(@abs(node_space.boundaryOp([2]isize{ -1, -1 }, null, [2]usize{ 0, 0 }, exact)) < 1e-10);
    try expect(@abs(node_space.boundaryOp([2]isize{ -2, -2 }, null, [2]usize{ 0, 0 }, exact)) < 1e-10);

    // *********************************

    const DiritchletBC = struct {
        pub fn kind(_: FaceIndex) BoundaryKind {
            return .robin;
        }

        pub fn robin(_: @This(), _: [2]f64, _: FaceIndex) Robin {
            return Robin.diritchlet(0.0);
        }
    };

    Engine.new(node_space).fill(DiritchletBC{}, field);

    // **********************************
    // Test that boundary values are within a certain bound of the exact value

    for (0..node_space.numNodes()) |i| {
        try expect(@abs(field[i] - exact[i]) < 1e-10);
    }
}
