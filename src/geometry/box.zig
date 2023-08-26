const std = @import("std");
const IndexSpace = @import("index.zig").IndexSpace;

// /// Represents an index into the 2^N subcells formed
// /// when dividing a hyper box along each axis.
// pub fn SplitIndex(comptime N: usize) type {
//     return struct {
//         linear: u16,

//         const Self = @This();

//         /// Builds a `SplitIndex` for a cartesian index, ie an array
//         /// of bools indicating left/right split on each axis.
//         pub fn fromCartesian(cart: [N]bool) Self {
//             if (N > 16) {
//                 @compileError("Split index is only defined for values of N <= 16");
//             }

//             var linear: u16 = 0x0;

//             for (0..N) |i| {
//                 linear |= @as(u16, cart[i]) << i;
//             }

//             return linear;
//         }

//         /// Converts a linear `SplitIndex` to a cartesian implementation.
//         pub fn toCartesian(self: Self) [N]bool {
//             var cart: [N]bool = undefined;

//             for (0..N) |i| {
//                 cart[i] = self.linear & (1 << i) > 0;
//             }

//             return cart;
//         }

//         pub fn reverseAxis(self: Self, axis: usize) Self {
//             return .{
//                 .linear = self.linear ^ (1 << axis),
//             };
//         }
//     };
// }

/// A N-dimensional axis aligned bounding box over a field T.
pub fn Box(comptime N: usize, comptime T: type) type {
    return struct {
        origin: [N]T,
        size: [N]T,

        const Self = @This();

        /// Returns the position of the center of the box.
        pub fn center(self: Self) [N]T {
            var result: [N]T = undefined;

            for (0..N) |i| {
                result[i] = self.origin[i] + self.size[i] / @as(T, 2);
            }

            return result;
        }

        /// Checks if the given position is contained by the box.
        pub fn contains(self: Self, x: [N]T) bool {
            var result = true;

            for (0..N) |i| {
                result = result and self.origin[i] <= x[i] and x[i] <= self.origin[i] + self.size[i];
            }

            return result;
        }

        /// Returns the index space over the interior of the box.
        pub fn space(self: Self) IndexSpace(N) {
            return .{ .size = self.size };
        }

        pub fn globalFromLocal(self: Self, local: [N]usize) [N]usize {
            var global: [N]usize = undefined;
            for (0..N) |axis| {
                global[axis] = self.origin[axis] + local[axis];
            }
            return global;
        }

        pub fn refine(self: *Self) void {
            for (0..N) |axis| {
                self.origin[axis] *= 2;
                self.size[axis] *= 2;
            }
        }

        pub fn coarsen(self: *Self) !void {
            for (0..N) |axis| {
                self.origin[axis] = try std.math.divFloor(usize, self.origin[axis], 2);
                self.size[axis] = try std.math.divCeil(usize, self.size[axis], 2);
            }
        }
    };
}

test "box" {
    const expect = std.testing.expect;
    const eql = std.meta.eql;

    const unit: Box(2, f64) = .{
        .origin = [2]f64{ 0.0, 0.0 },
        .size = [2]f64{ 1.0, 1.0 },
    };

    try expect(eql(unit.center(), [2]f64{ 0.5, 0.5 }));
    try expect(unit.contains([2]f64{ 0.5, 0.5 }));
}
