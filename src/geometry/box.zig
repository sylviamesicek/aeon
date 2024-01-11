const std = @import("std");
const IndexSpace = @import("index.zig").IndexSpace;

const assert = std.debug.assert;

fn IntegerBitSet(comptime size: usize) type {
    const n: u16 = @intCast(size);
    return std.bit_set.IntegerBitSet(n);
}

pub fn AxisMask(comptime N: usize) type {
    return struct {
        bits: IntegerBitSet(N),

        pub fn initEmpty() @This() {
            return .{
                .bits = IntegerBitSet(N).initEmpty(),
            };
        }

        pub fn initFull() @This() {
            return .{
                .bits = IntegerBitSet(N).initFull(),
            };
        }

        /// Returns the total number of set bits in this bit set.
        pub fn count(self: @This()) usize {
            return self.bits.count();
        }

        pub fn complement(self: @This()) @This() {
            return .{
                .bits = self.bits.complement(),
            };
        }

        pub fn isSet(self: @This(), axis: usize) bool {
            return self.bits.isSet(axis);
        }

        pub fn set(self: *@This(), axis: usize) void {
            self.bits.set(axis);
        }

        pub fn setValue(self: *@This(), axis: usize, val: bool) void {
            self.bits.setValue(axis, val);
        }

        pub fn unset(self: *@This(), axis: usize) void {
            self.bits.unset(axis);
        }
    };
}

/// Identifies a face by axis and side (false is left, true is right).
pub fn FaceIndex(comptime N: usize) type {
    return struct {
        side: bool,
        axis: usize,

        /// Total number of face indices.
        pub const count = 2 * N;

        /// Returns a linear index corresponding to this face.
        pub fn toLinear(self: @This()) usize {
            return if (self.side) self.axis + N else self.axis;
        }

        /// Builds a face index from a linear index.
        pub fn fromLinear(index: usize) @This() {
            assert(index < count);

            return if (index >= N)
                .{ .side = true, .axis = index - N }
            else
                .{ .side = false, .axis = index };
        }

        pub fn enumerate() [count]@This() {
            var result: [count]@This() = undefined;

            for (0..count) |i| {
                result[i] = fromLinear(i);
            }

            return result;
        }
    };
}

pub fn FaceMask(comptime N: usize) type {
    return struct {
        bits: IntegerBitSet(2 * N),

        const Index = FaceIndex(N);

        pub fn initEmpty() @This() {
            return .{
                .bits = IntegerBitSet(2 * N).initEmpty(),
            };
        }

        pub fn initFull() @This() {
            return .{
                .bits = IntegerBitSet(2 * N).initFull(),
            };
        }

        pub fn count(self: @This()) usize {
            return self.bits.count();
        }

        pub fn complement(self: @This()) @This() {
            return .{
                .bits = self.bits.complement(),
            };
        }

        pub fn isSet(self: @This(), face: Index) bool {
            return self.bits.isSet(face.toLinear());
        }

        pub fn set(self: *@This(), face: Index) void {
            self.bits.set(face.toLinear());
        }

        pub fn setValue(self: *@This(), face: Index, val: bool) void {
            self.bits.setValue(face.toLinear(), val);
        }

        pub fn unset(self: *@This(), face: Index) void {
            self.bits.unset(face.toLinear());
        }
    };
}

test "face index" {
    const expectEqual = std.testing.expectEqual;

    const Index = FaceIndex(2);
    const Mask = FaceMask(2);

    try expectEqual(Index.count, 4);

    const face = Index.fromLinear(2);

    try expectEqual(face.side, true);
    try expectEqual(face.axis, 0);

    var mask = Mask.initEmpty();
    mask.set(.{ .axis = 1, .side = false });

    try expectEqual(mask.isSet(.{ .axis = 0, .side = false }), false);
    try expectEqual(mask.isSet(.{ .axis = 1, .side = false }), true);

    mask.unset(.{ .axis = 1, .side = false });

    try expectEqual(mask.isSet(.{ .axis = 1, .side = false }), false);
}

/// Represents an index into the 2^N subcells formed
/// when dividing a hyper box along each axis. It is essentially
/// a bitvector which can be packed and unpacked from a `[N]bool`,
/// `array[i] == false` indicates that the subcell is on the left of
/// the `i`th axis, and `array[i] == true` indicates the subcell is on the
/// right. The split index can also be converted to and from a linear index,
/// to allow for storing subcell linearly (e.g. in an octree).
pub fn SplitIndex(comptime N: usize) type {
    return struct {
        bits: IntegerBitSet(N),

        const Self = @This();

        /// Number of possible split indices in this dimension.
        pub const count = blk: {
            var result: usize = 1;

            for (0..N) |_| {
                result *= 2;
            }

            break :blk result;
        };

        /// Builds a `SplitIndex` for a cartesian index, ie an array
        /// of bools indicating left/right split on each axis.
        pub fn fromCartesian(cart: [N]bool) Self {
            var bits = IntegerBitSet(N).initEmpty();

            inline for (0..N) |axis| {
                bits.setValue(axis, cart[axis]);
            }

            return .{ .bits = bits };
        }

        /// Builds a split index from a linear index.
        pub fn fromLinear(linear: usize) Self {
            return .{
                .bits = .{ .mask = @intCast(linear) },
            };
        }

        /// Converts a linear `SplitIndex` to a cartesian implementation.
        pub fn toCartesian(self: Self) [N]bool {
            var cart: [N]bool = undefined;

            inline for (0..N) |i| {
                cart[i] = self.bits.isSet(i);
            }

            return cart;
        }

        pub fn toLinear(self: @This()) usize {
            return @as(usize, self.bits.mask);
        }

        /// Finds the mirrored index along this axis.
        pub fn reverseAxis(self: Self, axis: usize) Self {
            var result = self;
            result.bits.toggle(axis);
            return result;
        }

        /// Enumerates all possible split indices.
        pub fn enumerate() [count]@This() {
            var result: [count]@This() = undefined;

            for (0..count) |linear| {
                result[linear] = fromLinear(linear);
            }

            return result;
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
    try expectEqualDeep(split.toLinear(), 2);
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

        pub fn split(self: @This(), index: SplitIndex(N)) RealBox(N) {
            const cart = index.toCartesian();

            var result: RealBox(N) = undefined;

            for (0..N) |axis| {
                result.origin[axis] = self.origin[axis];
                result.size[axis] = self.size[axis] / 2;

                if (cart[axis]) {
                    result.origin[axis] += self.size[axis] / 2;
                }
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

        pub const unit: @This() = .{
            .origin = [1]f64{0.0} ** N,
            .size = [1]f64{1.0} ** N,
        };

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

        pub fn split(self: @This(), index: SplitIndex(N)) RealBox(N) {
            const cart = index.toCartesian();

            var result: RealBox(N) = undefined;

            for (0..N) |axis| {
                result.origin[axis] = self.origin[axis];
                result.size[axis] = self.size[axis] / 2.0;

                if (cart[axis]) {
                    result.origin[axis] += self.size[axis] / 2.0;
                }
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

        // pub fn transformOp(self: @This(), ranks: [N]usize, v: f64) f64 {
        //     var res: f64 = v;

        //     for (0..N) |i| {
        //         for (0..ranks[i]) |_| {
        //             res /= self.size[i];
        //         }
        //     }

        //     return res;
        // }
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
