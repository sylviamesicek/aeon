const std = @import("std");
const IndexSpace = @import("index.zig").IndexSpace;

const assert = std.debug.assert;

fn IntegerBitSet(comptime size: usize) type {
    const n: u16 = @intCast(size);
    return std.bit_set.IntegerBitSet(n);
}

/// Represents a set of axes, stored as a bit set of N axes, where the nth bit is set if that axis is
/// included in the mask.
pub fn AxisMask(comptime N: usize) type {
    return struct {
        bits: IntegerBitSet(N),

        const MaskInt = IntegerBitSet(N).MaskInt;

        /// The total number of permutations of axis masks.
        pub const count = blk: {
            var result: usize = 1;

            for (0..N) |_| {
                result *= 2;
            }

            break :blk result;
        };

        /// Creates an empty axis mask.
        pub fn initEmpty() @This() {
            return .{
                .bits = IntegerBitSet(N).initEmpty(),
            };
        }

        /// Creates a full axis mask.
        pub fn initFull() @This() {
            return .{
                .bits = IntegerBitSet(N).initFull(),
            };
        }

        /// Creates an axis mask from a linear index.
        pub fn fromLinear(linear: usize) @This() {
            return .{ .bits = .{ .mask = @as(MaskInt, @intCast(linear)) } };
        }

        /// Converts an axis mask to a linear index.
        pub fn toLinear(self: @This()) usize {
            const result: usize = @intCast(self.bits.mask);
            return result;
        }

        /// Unpacks an axis mask into a boolean array.
        pub fn unpack(self: @This()) [N]bool {
            var result: [N]bool = undefined;

            for (0..N) |axis| {
                result[axis] = self.isSet(axis);
            }

            return result;
        }

        /// Packs an axis mask from a boolean array.
        pub fn pack(vals: [N]bool) @This() {
            var result = initEmpty();

            for (0..N) |axis| {
                result.setValue(axis, vals[axis]);
            }

            return result;
        }

        /// Computes the complement of the axis mask.
        pub fn complement(self: @This()) @This() {
            return .{
                .bits = self.bits.complement(),
            };
        }

        /// Computes the intersection of two axis masks.
        pub fn intersectWith(self: @This(), other: @This()) @This() {
            return .{
                .bits = self.bits.intersectWith(other.bits),
            };
        }

        /// Determines if the given axis is set.
        pub fn isSet(self: @This(), axis: usize) bool {
            return self.bits.isSet(axis);
        }

        /// Sets the given axis to true.
        pub fn set(self: *@This(), axis: usize) void {
            self.bits.set(axis);
        }

        /// Sets the given axis to false.
        pub fn unset(self: *@This(), axis: usize) void {
            self.bits.unset(axis);
        }

        /// Toggles the given axis.
        pub fn toggle(self: *@This(), axis: usize) void {
            self.bits.toggle(axis);
        }

        pub fn toggled(self: @This(), axis: usize) @This() {
            var result = self;
            result.toggle(axis);
            return result;
        }

        /// Sets the value of the axis to `val`.
        pub fn setValue(self: *@This(), axis: usize, val: bool) void {
            self.bits.setValue(axis, val);
        }

        /// Checks if the set is empty.
        pub fn isEmpty(self: @This()) bool {
            return self.bits.mask == 0;
        }

        /// Enumerates inner faces when an axis mask is treated as a splitting index.
        pub fn innerFaces(self: @This()) [N]FaceIndex(N) {
            var result: [N]FaceIndex(N) = undefined;

            for (0..N) |axis| {
                result[axis] = .{
                    .axis = axis,
                    .side = !self.bits.isSet(axis),
                };
            }

            return result;
        }

        /// Enumerates the outer faces when an axis mask is treated as a splitting index.
        pub fn outerFaces(self: @This()) [N]FaceIndex(N) {
            var result: [N]FaceIndex(N) = undefined;

            for (0..N) |axis| {
                result[axis] = .{
                    .axis = axis,
                    .side = self.bits.isSet(axis),
                };
            }

            return result;
        }

        /// Enumerates the possible axis masks of this dimension.
        pub fn enumerate() [count]@This() {
            var result: [count]@This() = undefined;

            for (0..count) |i| {
                result[i].bits.mask = @intCast(i);
            }

            return result;
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

/// A set of faces, used for various boundary routines.
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

// /// Represents an index into the 2^N subcells formed
// /// when dividing a hyper box along each axis. It is essentially
// /// a bitvector which can be packed and unpacked from a `[N]bool`,
// /// `array[i] == false` indicates that the subcell is on the left of
// /// the `i`th axis, and `array[i] == true` indicates the subcell is on the
// /// right. The split index can also be converted to and from a linear index,
// /// to allow for storing subcell linearly (e.g. in an octree).
// pub fn SplitIndex(comptime N: usize) type {
//     return struct {
//         bits: IntegerBitSet(N),

//         const Self = @This();

//         /// Number of possible split indices in this dimension.
//         pub const count = blk: {
//             var result: usize = 1;

//             for (0..N) |_| {
//                 result *= 2;
//             }

//             break :blk result;
//         };

//         /// Builds a `SplitIndex` for a cartesian index, ie an array
//         /// of bools indicating left/right split on each axis.
//         pub fn fromCartesian(cart: [N]bool) Self {
//             var bits = IntegerBitSet(N).initEmpty();

//             inline for (0..N) |axis| {
//                 bits.setValue(axis, cart[axis]);
//             }

//             return .{ .bits = bits };
//         }

//         /// Builds a split index from a linear index.
//         pub fn fromLinear(linear: usize) Self {
//             return .{
//                 .bits = .{ .mask = @intCast(linear) },
//             };
//         }

//         /// Converts a linear `SplitIndex` to a cartesian implementation.
//         pub fn toCartesian(self: Self) [N]bool {
//             var cart: [N]bool = undefined;

//             inline for (0..N) |i| {
//                 cart[i] = self.bits.isSet(i);
//             }

//             return cart;
//         }

//         pub fn toLinear(self: @This()) usize {
//             return @as(usize, self.bits.mask);
//         }

//         /// Finds the mirrored index along this axis.
//         pub fn reverseAxis(self: Self, axis: usize) Self {
//             var result = self;
//             result.bits.toggle(axis);
//             return result;
//         }

//         pub fn innerFaces(self: Self) [N]FaceIndex(N) {
//             var result: [N]FaceIndex(N) = undefined;

//             for (0..N) |axis| {
//                 result[axis] = .{
//                     .axis = axis,
//                     .side = !self.bits.isSet(axis),
//                 };
//             }

//             return result;
//         }

//         pub fn outerFaces(self: Self) [N]FaceIndex(N) {
//             var result: [N]FaceIndex(N) = undefined;

//             for (0..N) |axis| {
//                 result[axis] = .{
//                     .axis = axis,
//                     .side = self.bits.isSet(axis),
//                 };
//             }

//             return result;
//         }

//         /// Enumerates all possible split indices.
//         pub fn enumerate() [count]@This() {
//             var result: [count]@This() = undefined;

//             for (0..count) |linear| {
//                 result[linear] = fromLinear(linear);
//             }

//             return result;
//         }
//     };
// }

// test "split index" {
//     const expectEqualDeep = std.testing.expectEqualDeep;

//     const split = SplitIndex(2).fromCartesian([_]bool{ false, true });
//     // Make sure that from/toCartesian are inverses
//     try expectEqualDeep(split.toCartesian(), [_]bool{ false, true });
//     // Test reverse axis
//     try expectEqualDeep(split.reverseAxis(0).toCartesian(), [_]bool{ true, true });
//     // Check linear representation
//     try expectEqualDeep(split.toLinear(), 2);
// }
