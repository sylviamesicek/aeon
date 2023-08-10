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

        /// An iterator over the cartesian indices of the index space.
        pub const CartesianIterator = struct {
            size: [N]usize,
            cursor: [N]usize,

            pub fn next(self: *CartesianIterator) ?[N]usize {
                // Last index was incremented, iteration is complete.
                if (self.cursor[0] == self.size[0]) {
                    return null;
                }

                // Store current cursor value (this is what we will return).
                var result = self.cursor;

                // Set an increment array. In 3d this looks like [false, false, true].
                var increment = [1]bool{false} ** N;
                increment[N - 1] = true;

                // Loop over axes in reverse order.
                for (0..N) |i| {
                    const axis = N - 1 - i;

                    if (increment[axis]) {
                        // If we need to increment this axis, we add to the cursor value.
                        self.cursor[axis] += 1;
                        // If cursor is equal to the size, we need to perform wrapping.
                        // However, if we have reached the final axis, this indicates we
                        // are at the end of iteration, and will return null on the next
                        // call to next.
                        if (self.cursor[axis] == self.size[axis]) {
                            if (axis != 0) {
                                self.cursor[axis] = 0;
                                increment[axis - 1] = true;
                            }
                        }
                    }
                }

                // Return old cursor position.
                return result;
            }
        };

        /// Iterates the cartesian indices of an index space.
        pub fn cartesianIndices(self: Self) CartesianIterator {
            return .{
                .size = self.size,
                .cursor = [1]usize{0} ** N,
            };
        }
    };
}

test "index space" {
    const std = @import("std");
    const expect = std.testing.expect;
    const eql = std.mem.eql;

    const space: IndexSpace(3) = .{
        .size = [3]usize{ 2, 1, 3 },
    };

    try expect(space.cartesianToLinear([_]usize{ 1, 0, 2 }) == 5);
    try expect(eql(usize, &space.linearToCartesian(5), &[_]usize{ 1, 0, 2 }));

    var iterator = space.cartesianIndices();

    try expect(eql(usize, &iterator.next().?, &[_]usize{ 0, 0, 0 }));
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 0, 0, 1 }));
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 0, 0, 2 }));
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 1, 0, 0 }));
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 1, 0, 1 }));
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 1, 0, 2 }));
    try expect(iterator.next() == null);
}
