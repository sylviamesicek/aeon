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

/// Provides utility functions for filling boundary regions.
pub fn BoundaryUtils(comptime N: usize, comptime M: usize) type {
    return struct {
        const FaceIndex = geometry.FaceIndex(N);
        const IndexMixin = geometry.IndexMixin(N);
        const Region = geometry.Region(N);
        const NodeSpace = nodes_.NodeSpace(N, M);

        pub fn fillBoundary(space: NodeSpace, bound: anytype, field: []f64) void {
            const regions = comptime Region.orderedRegions();

            inline for (comptime regions[1..]) |region| {
                fillBoundaryRegion(region, space, bound, field);
            }
        }

        pub fn fillBoundaryRegion(comptime region: Region, space: NodeSpace, bound: anytype, field: []f64) void {
            const Bound = @TypeOf(bound);

            if (comptime !isBoundary(N)(Bound)) {
                @compileError("Boundary must satisfy isBoundary trait.");
            }

            assert(field.len == space.numNodes());

            // Short circuit if the region does not actually have any boundary nodes
            if (comptime region.adjacency() == 0) {
                return;
            }

            // Loop over the cells touching this face.
            var inner_face_cells = region.innerFaceCells(space.size);

            while (inner_face_cells.next()) |cell| {
                // Cast to signed node index.
                const node = IndexMixin.toSigned(cell);
                // Find position of boundary
                const pos: [N]f64 = space.boundaryPosition(region, cell);

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
                    space.setNodeValue(target, field, 0.0);

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

                                const source_value: f64 = space.nodeValue(source, field);
                                const fsign: f64 = comptime if (kind == .odd) -1.0 else 1.0;

                                result += fsign * source_value;
                            },
                            .robin => {
                                const vres: f64 = robin[axis].value * space.boundaryOp(extents, null, cell, field);
                                const fres: f64 = robin[axis].flux * space.boundaryOp(extents, axis, cell, field);

                                const vcoef: f64 = robin[axis].value * space.boundaryOpCoef(extents, null);
                                const fcoef: f64 = robin[axis].flux * space.boundaryOpCoef(extents, axis);

                                const rhs: f64 = robin[axis].rhs;

                                result += (rhs - vres - fres) / (vcoef + fcoef);
                            },
                        }
                    }

                    // Take the average
                    const adj: f64 = @floatFromInt(region.adjacency());
                    result /= adj;

                    // Set target to result.
                    space.setNodeValue(target, field, result);
                }
            }
        }
    };
}

/// A trait for defining boundaries.
pub fn isBoundary(comptime N: usize) fn (comptime T: type) bool {
    const FaceIndex = geometry.FaceIndex(N);
    const hasFn = std.meta.trait.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (!(hasFn("kind")(T) and @TypeOf(T.kind) == fn (FaceIndex) BoundaryKind)) {
                return false;
            }

            if (!(hasFn("robin")(T) and @TypeOf(T.robin) == fn (T, [N]f64, FaceIndex) Robin)) {
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
    const RealBox = geometry.RealBox(2);
    const NodeSpace = nodes_.NodeSpace(2, 2);

    const domain: RealBox = .{ .origin = .{ 0.0, 0.0 }, .size = .{ pi, pi } };

    const node_space = NodeSpace.fromCellSize([2]usize{ 100, 100 });
    const cell_space = node_space.cellSpace();

    // *********************************
    // Set field values ****************

    const field: []f64 = try allocator.alloc(f64, node_space.numNodes());
    defer allocator.free(field);

    {
        var cells = cell_space.cartesianIndices();

        while (cells.next()) |cell| {
            const pos = domain.transformPos(node_space.cellPosition(cell));

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
            const pos = domain.transformPos(node_space.nodePosition(node));

            node_space.setNodeValue(node, exact, @sin(pos[0]) * @sin(pos[1]));
        }
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

    BoundaryUtils(2, 2).fillBoundary(node_space, DiritchletBC{}, field);

    // **********************************
    // Test that boundary values are within a certain bound of the exact value

    for (0..node_space.numNodes()) |i| {
        try expect(@fabs(field[i] - exact[i]) < 1e-10);
    }
}
