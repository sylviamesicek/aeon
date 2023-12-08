const std = @import("std");
const assert = std.debug.assert;

/// A mixin for common manipulation indices (`[N]usize`).
pub fn IndexMixin(comptime N: usize) type {
    return struct {
        pub fn splat(v: usize) [N]usize {
            return [1]usize{v} ** N;
        }

        pub fn add(v: [N]usize, u: [N]usize) [N]usize {
            var res: [N]usize = undefined;

            inline for (0..N) |i| {
                res[i] = v[i] + u[i];
            }

            return res;
        }

        pub fn addSigned(v: [N]isize, u: [N]isize) [N]isize {
            var res: [N]isize = undefined;

            inline for (0..N) |i| {
                res[i] = v[i] + u[i];
            }

            return res;
        }

        pub fn sub(v: [N]usize, u: [N]usize) [N]usize {
            var res: [N]usize = v;

            inline for (0..N) |i| {
                res[i] -= u[i];
            }

            return res;
        }

        pub fn scaled(v: [N]usize, u: usize) [N]usize {
            var res: [N]usize = v;

            inline for (0..N) |i| {
                res[i] *= u;
            }

            return res;
        }

        pub fn addWithSign(v: [N]usize, u: [N]isize) [N]usize {
            var res: [N]usize = undefined;

            for (0..N) |i| {
                const vf: isize = @intCast(v[i]);
                const uf: isize = u[i];
                res[i] = @intCast(vf + uf);
            }

            return res;
        }

        pub fn offsetFromOrigin(origin: [N]usize, offset: [N]isize) [N]isize {
            var result: [N]isize = undefined;

            for (0..N) |i| {
                result[i] = @as(isize, @intCast(origin[i])) + offset[i];
            }

            return result;
        }

        pub fn refined(self: [N]usize) [N]usize {
            var result: [N]usize = undefined;
            inline for (0..N) |i| {
                result[i] = self[i] * 2;
            }
            return result;
        }

        pub fn coarsened(self: [N]usize) [N]usize {
            var result: [N]usize = undefined;
            inline for (0..N) |i| {
                result[i] = self[i] / 2;
            }
            return result;
        }

        pub fn toSigned(self: [N]usize) [N]isize {
            var result: [N]isize = undefined;
            for (0..N) |i| {
                result[i] = @intCast(self[i]);
            }
            return result;
        }

        pub fn toUnsigned(self: [N]isize) [N]usize {
            var result: [N]usize = undefined;
            for (0..N) |i| {
                result[i] = @intCast(self[i]);
            }
            return result;
        }
    };
}

/// Describes an abstract index space, ie an N-dimensional space with
/// size[i] discrete cells on each axis. Contains helpers for converting
/// between cartesian and linear indices, as well as iterator over the space.
/// This interprets multidimensional arrays as being in row-major (last most index
/// should change the fastest for optimal cache efficiency).
pub fn IndexSpace(comptime N: usize) type {
    return struct {
        size: [N]usize,

        const Self = @This();
        const IndexBox = @import("box.zig").IndexBox(N);

        /// Constructs an IndexSpace with the given size.
        pub fn fromSize(size: [N]usize) Self {
            return .{ .size = size };
        }

        /// Constructs an IndexSpace with the same size as an IndexBox.
        pub fn fromBox(box: IndexBox) Self {
            return .{ .size = box.size };
        }

        /// Converts a cartesian index to a linear index.
        pub fn linearFromCartesian(self: Self, cartesian: [N]usize) usize {
            var stride: usize = 1;
            var linear: usize = 0;

            for (0..N) |i| {
                // Iterate in reverse order (as last index is least significant).
                const axis = N - 1 - i;

                linear += stride * cartesian[axis];
                stride *= self.size[axis];
            }

            return linear;
        }

        /// Converts a linear index to a cartesian index (as a rule of thumb, this is much
        /// slower than the opposite conversion).
        pub fn cartesianFromLinear(self: Self, linear: usize) [N]usize {
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

        /// Returns the index of the largest axis in the space.
        pub fn longestAxis(self: Self) usize {
            var axis: usize = 0;

            for (1..N) |i| {
                if (self.size[i] > self.size[axis]) {
                    axis = i;
                }
            }

            return axis;
        }

        /// Builds a window, ie a smaller array that represents a subspace within the index space.
        pub fn copyWindow(self: Self, bounds: IndexBox, comptime T: type, dest: []T, src: []const T) void {
            const space = IndexSpace(N).fromBox(bounds);

            assert(dest.len == space.total());
            assert(src.len == self.total());

            // Set new tags for patch
            var indices = space.cartesianIndices();

            var i: usize = 0;

            while (indices.next()) |local| {
                const global: [N]usize = bounds.globalFromLocal(local);
                dest[i] = src[self.linearFromCartesian(global)];

                i += 1;
            }
        }

        /// Fills a subspace of an array representing values defined over the index space with `val`.
        pub fn fillWindow(self: Self, bounds: IndexBox, comptime T: type, dest: []T, val: T) void {
            const space = IndexSpace(N).fromBox(bounds);

            assert(dest.len == self.total());

            // Set new tags for patch
            var indices = space.cartesianIndices();

            while (indices.next()) |local| {
                const global = bounds.globalFromLocal(local);
                dest[self.linearFromCartesian(global)] = val;
            }
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

        /// An iterator over the cartesian indices of a slice of the index space.
        pub const CartesianSliceIterator = struct {
            indices: CartesianIterator,
            axis: usize,
            slice: usize,

            pub fn next(self: *CartesianSliceIterator) ?[N]usize {
                var index = self.indices.next() orelse return null;
                index[self.axis] = self.slice;
                return index;
            }
        };

        /// Iterates the cartesian indices of a slice of the index space.
        pub fn cartesianSliceIndices(self: Self, axis: usize, slice: usize) CartesianSliceIterator {
            var size = self.size;
            size[axis] = 1;

            return .{
                .indices = .{
                    .size = size,
                    .cursor = [1]usize{0} ** N,
                },
                .axis = axis,
                .slice = slice,
            };
        }
    };
}

test "index space" {
    const expect = std.testing.expect;
    const eql = std.mem.eql;

    const space: IndexSpace(3) = .{
        .size = [3]usize{ 2, 1, 3 },
    };

    try expect(space.linearFromCartesian([_]usize{ 1, 0, 2 }) == 5);
    try expect(eql(usize, &space.cartesianFromLinear(5), &[_]usize{ 1, 0, 2 }));

    var iterator = space.cartesianIndices();
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 0, 0, 0 }));
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 0, 0, 1 }));
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 0, 0, 2 }));
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 1, 0, 0 }));
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 1, 0, 1 }));
    try expect(eql(usize, &iterator.next().?, &[_]usize{ 1, 0, 2 }));
    try expect(iterator.next() == null);

    var iterator_slice = space.cartesianSliceIndices(0, 1);
    try expect(eql(usize, &iterator_slice.next().?, &[_]usize{ 1, 0, 0 }));
    try expect(eql(usize, &iterator_slice.next().?, &[_]usize{ 1, 0, 1 }));
    try expect(eql(usize, &iterator_slice.next().?, &[_]usize{ 1, 0, 2 }));
    try expect(iterator_slice.next() == null);

    var indices = space.cartesianIndices();
    var linear: usize = 0;

    while (indices.next()) |index| : (linear += 1) {
        try expect(linear == space.linearFromCartesian(index));
    }
}
