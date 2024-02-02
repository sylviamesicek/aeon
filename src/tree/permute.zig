const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUmanaged = std.ArrayListUnmanaged;
const assert = std.debug.assert;
const panic = std.debug.panic;

const geometry = @import("../geometry/geometry.zig");

/// A class for handling permutations between uniform squares in index space and tree space.
pub fn CellPermutation(comptime N: usize) type {
    return struct {
        buffer: []const usize,
        offsets: []const usize,

        const AxisMask = geometry.AxisMask(N);
        const IndexMixin = geometry.IndexMixin(N);
        const IndexSpace = geometry.IndexSpace(N);

        pub fn init(allocator: Allocator, max_refinement: usize) !@This() {
            // Compute offsets
            var offsets: []usize = try allocator.alloc(usize, max_refinement + 1);
            errdefer allocator.free(offsets);

            var offset: usize = 0;
            var total: usize = 1;

            offsets[0] = offset;

            for (1..max_refinement + 1) |i| {
                offset += total;
                offsets[i] = offset;
                total *= AxisMask.count;
            }

            // Compute buffer
            var buffer: []usize = try allocator.alloc(usize, offsets[max_refinement]);
            errdefer allocator.free(buffer);

            buffer[0] = 0;

            var size: usize = 1;

            for (0..(max_refinement - 1)) |i| {
                const src: []const usize = buffer[offsets[i]..offsets[i + 1]];
                const dest: []usize = buffer[offsets[i + 1]..offsets[i + 2]];

                const sspace = IndexSpace.fromSize(IndexMixin.splat(size));
                const dspace = IndexSpace.fromSize(IndexMixin.splat(size * 2));

                var indices = sspace.cartesianIndices();
                var linear: usize = 0;

                while (indices.next()) |sindex| : (linear += 1) {
                    for (AxisMask.enumerate()) |split| {
                        var dindex: [N]usize = undefined;

                        for (0..N) |axis| {
                            if (split.isSet(axis)) {
                                dindex[axis] = 2 * sindex[axis] + 1;
                            } else {
                                dindex[axis] = 2 * sindex[axis];
                            }
                        }

                        const dlinear = dspace.linearFromCartesian(dindex);
                        dest[dlinear] = AxisMask.count * src[linear] + split.toLinear();
                    }
                }

                size *= 2;
            }

            return .{
                .buffer = buffer,
                .offsets = offsets,
            };
        }

        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.buffer);
            allocator.free(self.offsets);
        }

        pub fn permutation(self: @This(), refinement: usize) []const usize {
            return self.buffer[self.offsets[refinement]..self.offsets[refinement + 1]];
        }

        pub fn maxRefinement(self: @This()) usize {
            return self.offsets.len - 1;
        }
    };
}

test "cell permutation" {
    const expect = std.testing.expect;
    const expectEqualSlices = std.testing.expectEqualSlices;
    const allocator = std.testing.allocator;

    const permute = try CellPermutation(2).init(allocator, 4);
    defer permute.deinit(allocator);

    try expect(permute.maxRefinement() == 4);
    try expect(permute.offsets[0] == 0);
    try expect(permute.offsets[1] == 1);
    try expect(permute.offsets[2] == 5);

    try expectEqualSlices(usize, permute.permutation(0), &.{0});
    try expectEqualSlices(usize, permute.permutation(1), &.{
        0, 1,
        2, 3,
    });
    try expectEqualSlices(usize, permute.permutation(2), &.{
        0,  1,  4,  5,
        2,  3,  6,  7,
        8,  9,  12, 13,
        10, 11, 14, 15,
    });

    try expectEqualSlices(usize, permute.permutation(3), &.{
        0,  1,  4,  5,  16, 17, 20, 21,
        2,  3,  6,  7,  18, 19, 22, 23,
        8,  9,  12, 13, 24, 25, 28, 29,
        10, 11, 14, 15, 26, 27, 30, 31,
        32, 33, 36, 37, 48, 49, 52, 53,
        34, 35, 38, 39, 50, 51, 54, 55,
        40, 41, 44, 45, 56, 57, 60, 61,
        42, 43, 46, 47, 58, 59, 62, 63,
    });
}
