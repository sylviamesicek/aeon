const std = @import("std");

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
        widths: [N]T,

        const Self = @This();

        pub fn center(self: Self) [N]T {
            var result: [N]T = undefined;

            for (0..N) |i| {
                result[i] = self.origin[i] + self.widths[i] / @as(T, 2);
            }

            return result;
        }

        pub fn contains(self: Self, x: [N]T) bool {
            var result = true;

            for (0..N) |i| {
                result = result and self.origin[i] <= x[i] and x[i] <= self.origin[i] + self.widths[i];
            }

            return result;
        }
    };
}

test "box" {
    const expect = std.testing.expect;
    const eql = std.meta.eql;

    const unit: Box(2, f64) = .{
        .origin = [2]f64{ 0.0, 0.0 },
        .widths = [2]f64{ 1.0, 1.0 },
    };

    try expect(eql(unit.center(), [2]f64{ 0.5, 0.5 }));
    try expect(unit.contains([2]f64{ 0.5, 0.5 }));
}
