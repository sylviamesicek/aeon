/// Describes an abstract index space, ie an N-dimensional space with
/// size[i] discrete cells on each axis. Contains helpers for converting
/// between cartesian and linear indices.
pub fn IndexSpace(comptime N: usize) type {
    return struct {
        size: [N]usize,

        const Self = @This();

        /// Converts a cartesian index to a linear index.
        pub fn cartesianToLinear(self: Self, cartesian: [N]usize) usize {
            var stride: usize = 1;
            var linear: usize = 0;

            for (0..N) |i| {
                // Iterate in reverse order (as last index is most significant).
                const axis = N - 1 - i;

                linear += stride * cartesian[axis];
                stride *= self.size[axis];
            }

            return linear;
        }

        /// Converts a linear index to a cartesian index (as a rule of thumb, this is much
        /// slower than the opposite conversion).
        pub fn linearToCartesian(self: Self, linear: usize) [N]usize {
            var cartesian: [N]usize = undefined;
            var index = linear;

            for (0..N) |i| {
                const axis = N - 1 - i;

                cartesian[axis] = index % self.size[axis];
                index /= self.size[axis];
            }

            return cartesian;
        }

        /// Computes the total number of indices in this space (ie, the product of
        /// indices on each axis).
        pub fn total(self: Self) usize {
            var result: usize = 1;

            for (0..N) |i| {
                result *= self.size[i];
            }

            return result;
        }
    };
}

test "index space" {
    const std = @import("std");
    const expect = std.testing.expect;
    const eql = std.mem.eql;
    const space: IndexSpace(3) = .{
        .size = [3]usize{ 10, 10, 10 },
    };

    try expect(space.cartesianToLinear([_]usize{ 4, 4, 4 }) == 444);
    try expect(eql(usize, &space.linearToCartesian(444), &[_]usize{ 4, 4, 4 }));
}
