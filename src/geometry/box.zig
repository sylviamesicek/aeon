const std = @import("std");
const IndexSpace = @import("index.zig").IndexSpace;

const assert = std.debug.assert;

/// Identifies a face by axis and side (false is left, true is right).
pub fn FaceIndex(comptime N: usize) type {
    return struct {
        side: bool,
        axis: usize,

        /// Returns a linear index corresponding to this face.
        pub fn toLinear(self: @This()) usize {
            return if (self.side) .{ .index = self.axis + N } else .{ .index = self.axis };
        }

        /// Builds a face index from a linear index.
        pub fn fromLinear(index: usize) @This() {
            assert(index < comptime numFaces(N));

            return if (index >= N)
                .{ .side = true, .axis = index - N }
            else
                .{ .side = false, .axis = index };
        }
    };
}

/// Returns the total number of faces for a given dimension..=
pub fn numFaces(n: usize) usize {
    return 2 * n;
}

test "face index" {
    const expectEqual = std.testing.expectEqual;

    try expectEqual(numFaces(2), 4);

    const face = FaceIndex(2).fromLinear(2);

    try expectEqual(face.side, true);
    try expectEqual(face.axis, 0);
}

/// Represents an index into the 2^N subcells formed
/// when dividing a hyper box along each axis. It is essentially
/// a bitvector which can be packed and unpacked from a `[N]bool`,
/// `array[i] == false` indicates that the subcell is on the left of
/// the `i`th axis, and `array[i] == true` indicates the subcell is on the
/// right. The split index can also be converted to and from a linear index,
/// to allow for storing subcell linearly (e.g. in an octree).
pub fn SplitIndex(comptime N: usize) type {
    if (N > 16) {
        @compileError("Split index is only defined for values of N <= 16");
    }

    return struct {
        linear: u16,

        const Self = @This();

        /// Builds a `SplitIndex` for a cartesian index, ie an array
        /// of bools indicating left/right split on each axis.
        pub fn fromCartesian(cart: [N]bool) Self {
            var linear: u16 = 0x0;

            inline for (0..N) |i| {
                linear |= @as(u16, @intFromBool(cart[i])) << i;
            }

            return .{ .linear = linear };
        }

        /// Converts a linear `SplitIndex` to a cartesian implementation.
        pub fn toCartesian(self: Self) [N]bool {
            var cart: [N]bool = undefined;

            inline for (0..N) |i| {
                cart[i] = self.linear & (@as(u16, 1) << i) > 0;
            }

            return cart;
        }

        /// Finds the mirrored index along this axis.
        pub fn reverseAxis(self: Self, axis: usize) Self {
            assert(axis < N);

            return .{
                .linear = self.linear ^ (@as(u16, 1) << @as(u4, @intCast(axis))),
            };
        }
    };
}

test "split index" {
    const expectEqualDeep = std.testing.expectEqualDeep;

    const split = SplitIndex(2).fromCartesian([_]bool{ false, true });
    // Make sure that from/toCartesian are inverses
    try expectEqualDeep(split.toCartesian(), [_]bool{ false, true });
    // Test reverse axis
    try expectEqualDeep(split.reverseAxis(0).toCartesian(), [_]bool{ true, true });
    // Check linear representation
    try expectEqualDeep(split.linear, 2);
}

/// An N-dimensional subregion of some larger index space.
pub fn IndexBox(comptime N: usize) type {
    return struct {
        origin: [N]usize,
        size: [N]usize,

        /// Transforms a local index into a global one.
        pub fn globalFromLocal(self: @This(), local: [N]usize) [N]usize {
            var global: [N]usize = undefined;
            for (0..N) |axis| {
                global[axis] = self.origin[axis] + local[axis];
            }
            return global;
        }

        /// Transforms a global index into a local one
        pub fn localFromGlobal(self: @This(), global: [N]usize) [N]usize {
            var local: [N]usize = undefined;
            for (0..N) |axis| {
                local[axis] = global[axis] - self.origin[axis];
            }
            return local;
        }

        /// Checks if the given index is contained in the box.
        pub fn contains(self: @This(), index: [N]usize) bool {
            var result = true;

            inline for (0..N) |i| {
                result = result and self.origin[i] <= index[i] and index[i] <= self.origin[i] + self.size[i];
            }

            return result;
        }

        /// Refines the box by multiplying each component by 2.
        pub fn refined(self: @This()) @This() {
            var result: @This() = undefined;
            for (0..N) |axis| {
                result.origin[axis] = self.origin[axis] * 2;
                result.size[axis] = self.size[axis] * 2;
            }
            return result;
        }

        /// Coarsens the box by dividing each component by 2 (and rounding down).
        pub fn coarsened(self: @This()) @This() {
            var result: @This() = undefined;
            for (0..N) |axis| {
                result.origin[axis] = self.origin[axis] / 2;
                result.size[axis] = self.size[axis] / 2;
            }
            return result;
        }

        /// Moves the box so that its origin is measured relative to the origin of `other`.
        pub fn relativeTo(self: @This(), other: @This()) @This() {
            var result: @This() = self;

            for (0..N) |i| {
                result.origin[i] -= other.origin[i];
            }

            return result;
        }
    };
}

pub fn RealBox(comptime N: usize) type {
    return struct {
        origin: [N]f64,
        size: [N]f64,

        /// Returns the position of the center of the box.
        pub fn center(self: @This()) [N]f64 {
            var result: [N]f64 = undefined;

            for (0..N) |i| {
                result[i] = self.origin[i] + self.size[i] / 2.0;
            }

            return result;
        }

        /// Checks if the given position is contained by the box.
        pub fn contains(self: @This(), x: [N]f64) bool {
            var result = true;

            for (0..N) |i| {
                result = result and self.origin[i] <= x[i] and x[i] <= self.origin[i] + self.size[i];
            }

            return result;
        }

        pub fn transformPos(self: @This(), x: [N]f64) [N]f64 {
            var result: [N]f64 = undefined;

            for (0..N) |i| {
                result[i] = self.origin[i] + self.size[i] * x[i];
            }

            return result;
        }

        pub fn transformOp(self: @This(), comptime ranks: [N]usize, v: f64) f64 {
            var res: f64 = v;

            inline for (0..N) |i| {
                inline for (0..ranks[i]) |_| {
                    res /= self.physical_bounds.size[i];
                }
            }

            return res;
        }
    };
}

test "box" {
    const expect = std.testing.expect;
    const expectEqualDeep = std.testing.expectEqualDeep;

    const unit: RealBox(2) = .{
        .origin = [2]f64{ 0.0, 0.0 },
        .size = [2]f64{ 1.0, 1.0 },
    };

    try expectEqualDeep([_]f64{ 0.5, 0.5 }, unit.center());
    try expect(unit.contains([2]f64{ 0.5, 0.5 }));
}
