const std = @import("std");
const insertion = std.sort.insertion;
const powi = std.math.powi;

const geometry = @import("geometry.zig");

/// Defines an extended region around a block that can optionally include ghost nodes. In 2D, a region subdivision looks like
///
///     ---------------
///     | NW | N | NE |
///     ---------------
///     | W  |   | E  |
///     ---------------
///     | SW | S | SE |
///     ---------------
///
/// And allows one to write algorithms that traverse the edges and corners in an ordered way and iterate over the
/// cells within a given region. Most functions and iterators accept an extent parameter `E` which determines the
/// additional width of these buffer regions and a `block` parameter which describes the size of the central block.
pub fn Region(comptime N: usize) type {
    return struct {
        sides: [N]Side,

        const Self = @This();
        const AxisMask = @import("axis.zig").AxisMask(N);
        const IndexBox = @import("box.zig").IndexBox(N);
        const IndexSpace = @import("index.zig").IndexSpace(N);

        pub const count = blk: {
            var result: usize = 1;

            for (0..N) |_| {
                result *= 3;
            }

            break :blk result;
        };

        // Which side the region is on for each axis.
        pub const Side = enum(u2) {
            left = 0,
            middle = 1,
            right = 2,
        };

        /// Converts a region into a linear index into an array.
        pub fn linear(self: @This()) usize {
            const rspace = IndexSpace.fromSize([1]usize{3} ** N);

            var cart: [N]usize = undefined;

            for (0..N) |axis| {
                cart[axis] = @intFromEnum(self.sides[axis]);
            }

            return rspace.linearFromCartesian(cart);
        }

        /// How many steps of adjacency to reach the middle region.
        pub fn adjacency(self: Self) usize {
            var res: usize = 0;

            for (0..N) |i| {
                if (self.sides[i] != .middle) {
                    res += 1;
                }
            }

            return res;
        }

        /// Returns a space corresponding to the size of the region.
        pub fn space(self: Self, comptime E: usize, block: [N]usize) IndexSpace {
            var size: [N]usize = undefined;

            for (0..N) |i| {
                if (self.sides[i] != .middle) {
                    size[i] = E;
                } else {
                    size[i] = block[i];
                }
            }

            return IndexSpace.fromSize(size);
        }

        pub const NodeIterator = struct {
            inner: IndexSpace.CartesianIterator,
            block: [N]usize,
            sides: [N]Side,

            pub fn next(self: *NodeIterator) ?[N]isize {
                if (self.inner.next()) |cart| {
                    var result: [N]isize = undefined;

                    for (0..N) |i| {
                        switch (self.sides[i]) {
                            .left => result[i] = -1 - @as(isize, @intCast(cart[i])),
                            .right => result[i] = @intCast(self.block[i] + cart[i]),
                            else => result[i] = @intCast(cart[i]),
                        }
                    }

                    return result;
                } else {
                    return null;
                }
            }
        };

        /// Iterates all ghost nodes in this region out to a given extent.
        pub fn nodes(self: Self, comptime E: usize, block: [N]usize) NodeIterator {
            return .{
                .inner = self.space(E, block).cartesianIndices(),
                .block = block,
                .sides = self.sides,
            };
        }

        pub const InnerFaceIterator = struct {
            inner: IndexSpace.CartesianIterator,
            block: [N]usize,
            sides: [N]Side,

            pub fn next(self: *InnerFaceIterator) ?[N]usize {
                if (self.inner.next()) |cart| {
                    var result: [N]usize = undefined;

                    for (0..N) |i| {
                        switch (self.sides[i]) {
                            .left => result[i] = 0,
                            .right => result[i] = self.block[i] - 1,
                            else => result[i] = cart[i],
                        }
                    }

                    return result;
                } else {
                    return null;
                }
            }
        };

        /// Iterates over all cells on the inner face.
        pub fn innerFaceCells(self: Self, block: [N]usize) InnerFaceIterator {
            var size: [N]usize = undefined;

            for (0..N) |i| {
                if (self.sides[i] == .left or self.sides[i] == .right) {
                    size[i] = 1;
                } else {
                    size[i] = block[i];
                }
            }

            return .{
                .inner = self.space(1, block).cartesianIndices(),
                .block = block,
                .sides = self.sides,
            };
        }

        pub const ExtentIterator = struct {
            inner: IndexSpace.CartesianIterator,
            sides: [N]Side,

            pub fn next(self: *ExtentIterator) ?[N]isize {
                if (self.inner.next()) |cart| {
                    var result: [N]isize = undefined;

                    for (0..N) |i| {
                        const off: isize = @intCast(cart[i]);

                        switch (self.sides[i]) {
                            .left => result[i] = -1 - off,
                            .right => result[i] = 1 + off,
                            else => result[i] = 0,
                        }
                    }

                    return result;
                } else {
                    return null;
                }
            }
        };

        /// Iterates outwards from the inner face. Provides offsets that can be added to the inner face
        /// index to find the global index.
        pub fn extentOffsets(self: Self, E: usize) ExtentIterator {
            // Determine size along each direction
            var size: [N]usize = undefined;

            for (0..N) |i| {
                if (self.sides[i] == .left or self.sides[i] == .right) {
                    size[i] = E;
                } else {
                    size[i] = 1;
                }
            }

            return .{
                .inner = IndexSpace.fromSize(size).cartesianIndices(),
                .sides = self.sides,
            };
        }

        pub fn extentDir(self: Self) [N]isize {
            var dir: [N]isize = undefined;

            for (0..N) |i| {
                switch (self.sides[i]) {
                    .left => dir[i] = -1,
                    .right => dir[i] = 1,
                    .middle => dir[i] = 0,
                }
            }

            return dir;
        }

        pub fn toMask(self: Self) AxisMask {
            var result = AxisMask.initEmpty();

            for (0..N) |axis| {
                result.setValue(axis, self.sides[axis] != .middle);
            }

            return result;
        }

        pub fn masked(self: Self, mask: AxisMask) @This() {
            var sides: [N]Side = self.sides;

            for (0..N) |i| {
                if (!mask.isSet(i)) {
                    sides[i] = .middle;
                }
            }

            return .{
                .sides = sides,
            };
        }

        pub fn maskedBySplit(self: Self, split: AxisMask) @This() {
            var mask = AxisMask.initEmpty();

            for (0..N) |axis| {
                const side = self.sides[axis];
                switch (side) {
                    .middle => mask.unset(axis),
                    .left => mask.setValue(axis, split.isSet(axis) == false),
                    .right => mask.setValue(axis, split.isSet(axis)),
                }
            }

            return self.masked(mask);
        }

        // ************************
        // Constructors ***********
        // ************************

        pub fn central() @This() {
            return .{
                .sides = [1]Side{.middle} ** N,
            };
        }

        /// Assembles an array of all valid regions.
        pub fn enumerate() [count]Region(N) {
            var regs: [count]Region(N) = undefined;

            const rspace = IndexSpace.fromSize([1]usize{3} ** N);

            var indices = rspace.cartesianIndices();

            var i: usize = 0;

            while (indices.next()) |cart| {
                var sides: [N]Side = undefined;

                for (0..N) |axis| {
                    sides[axis] = @enumFromInt(cart[axis]);
                }

                regs[i] = .{ .sides = sides };
                i += 1;
            }

            return regs;
        }

        /// Constructs an array of regions ordered by adjacency.
        pub fn enumerateOrdered() [count]Region(N) {
            var regs: [count]Region(N) = enumerate();
            insertion(Region(N), &regs, {}, lessThanFn);
            return regs;
        }

        fn lessThanFn(_: void, lhs: Region(N), rhs: Region(N)) bool {
            return lhs.adjacency() < rhs.adjacency();
        }
    };
}

test "region adjacency" {
    const expectEqualSlices = std.testing.expectEqualSlices;

    const Side = Region(2).Side;

    const regions = Region(2).enumerateOrdered();

    try expectEqualSlices(Side, &regions[0].sides, &[_]Side{ .middle, .middle });
    try expectEqualSlices(Side, &regions[1].sides, &[_]Side{ .middle, .left });
    try expectEqualSlices(Side, &regions[2].sides, &[_]Side{ .left, .middle });
    try expectEqualSlices(Side, &regions[3].sides, &[_]Side{ .right, .middle });
    try expectEqualSlices(Side, &regions[4].sides, &[_]Side{ .middle, .right });
    try expectEqualSlices(Side, &regions[5].sides, &[_]Side{ .left, .left });
    try expectEqualSlices(Side, &regions[6].sides, &[_]Side{ .right, .left });
    try expectEqualSlices(Side, &regions[7].sides, &[_]Side{ .left, .right });
    try expectEqualSlices(Side, &regions[8].sides, &[_]Side{ .right, .right });
}

test "region indices" {
    const expect = std.testing.expect;
    const expectEqualSlices = std.testing.expectEqualSlices;

    const AxisMask = Region(2).AxisMask;
    _ = AxisMask; // autofix
    const Side = Region(2).Side;

    const block: [2]usize = [_]usize{ 2, 2 };

    const region = Region(2){
        .sides = [_]Side{ .right, .left },
    };

    try expect(region.adjacency() == 2);

    var indices = region.nodes(2, block);

    try expectEqualSlices(isize, &[_]isize{ 2, -1 }, &indices.next().?);
    try expectEqualSlices(isize, &[_]isize{ 3, -1 }, &indices.next().?);
    try expectEqualSlices(isize, &[_]isize{ 2, -2 }, &indices.next().?);
    try expectEqualSlices(isize, &[_]isize{ 3, -2 }, &indices.next().?);
    try expect(indices.next() == null);

    var inner_cells = region.innerFaceCells(block);

    try expectEqualSlices(usize, &[_]usize{ 1, 0 }, &inner_cells.next().?);
    try expect(inner_cells.next() == null);

    var offsets = region.extentOffsets(2);

    try expectEqualSlices(isize, &[_]isize{ 1, -1 }, &offsets.next().?);
    try expectEqualSlices(isize, &[_]isize{ 2, -1 }, &offsets.next().?);
    try expectEqualSlices(isize, &[_]isize{ 1, -2 }, &offsets.next().?);
    try expectEqualSlices(isize, &[_]isize{ 2, -2 }, &offsets.next().?);
    try expect(offsets.next() == null);

    const other_region = Region(2){
        .sides = [_]Side{ .right, .right },
    };

    var other_offsets = other_region.extentOffsets(2);

    try expectEqualSlices(isize, &[_]isize{ 1, 1 }, &other_offsets.next().?);
    try expectEqualSlices(isize, &[_]isize{ 2, 1 }, &other_offsets.next().?);
    try expectEqualSlices(isize, &[_]isize{ 1, 2 }, &other_offsets.next().?);
    try expectEqualSlices(isize, &[_]isize{ 2, 2 }, &other_offsets.next().?);
    try expect(offsets.next() == null);
}
