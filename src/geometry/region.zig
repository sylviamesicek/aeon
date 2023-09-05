const std = @import("std");
const heap = std.sort.heap;

const geometry = @import("../geometry/geometry.zig");

/// Defines a region of a block when ghost cells are included.
pub fn Region(comptime N: usize, comptime O: usize) type {
    return struct {
        sides: [N]Side,

        // Which side the region is on for each axis.
        pub const Side = enum(u2) {
            left = 0,
            middle = 1,
            right = 2,
        };

        const Self = @This();
        const IndexSpace = geometry.IndexSpace(N);
        const IndexBox = geometry.Box(N, usize);

        /// How many steps of adjacency to reach the middle.
        pub fn adjacency(self: Self) usize {
            var res: usize = 0;

            for (0..N) |i| {
                if (self.sides[i] == .left or self.sides[i] == .right) {
                    res += 1;
                }
            }

            return res;
        }

        pub const CartesianIterator = struct {
            inner: IndexSpace.CartesianIterator,
            block: [N]usize,
            sides: [N]Side,

            pub fn next(self: CartesianIterator) ?[N]usize {
                if (self.inner.next()) |cart| {
                    var result: [N]usize = undefined;

                    for (0..N) |i| {
                        switch (self.sides[i]) {
                            .left => result[i] = O - 1 - cart[i],
                            .right => result[i] = O + self.block[i] + cart[i],
                            else => result[i] = O + cart[i],
                        }
                    }

                    return result;
                } else {
                    return null;
                }
            }
        };

        /// Iterates all cell indices (in ghost space) in this region.
        pub fn cartesianIndices(self: Self, block: [N]usize) CartesianIterator {
            var size: [N]usize = undefined;

            for (0..N) |i| {
                if (self.sides[i] == .left or self.sides[i] == .right) {
                    size[i] = O;
                } else {
                    size[i] = block[i];
                }
            }

            const space = IndexSpace.fromSize(size);

            return .{
                .inner = space.cartesianIndices(),
                .block = block,
                .sides = self.sides,
            };
        }

        pub const InnerFaceIterator = struct {
            inner: IndexSpace.CartesianIterator,
            block: [N]usize,
            sides: [N]Side,

            pub fn next(self: InnerFaceIterator) ?[N]usize {
                if (self.inner.next()) |cart| {
                    var result: [N]usize = undefined;

                    for (0..N) |i| {
                        switch (self.sides[i]) {
                            .left => result[i] = O,
                            .right => result[i] = O + self.block[i],
                            else => result[i] = O + cart[i],
                        }
                    }

                    return result;
                } else {
                    return null;
                }
            }
        };

        /// Iterates over all indices on the inner face.
        pub fn innerFaceIndices(self: Self, block: [N]usize) InnerFaceIterator {
            var size: [N]usize = undefined;

            for (0..N) |i| {
                if (self.sides[i] == .left or self.sides[i] == .right) {
                    size[i] = 1;
                } else {
                    size[i] = block[i];
                }
            }

            const space = IndexSpace.fromSize(size);

            return .{
                .inner = space.cartesianIndices(),
                .block = block,
                .sides = self.sides,
            };
        }

        pub const ExtentIterator = struct {
            inner: IndexSpace.CartesianIterator,
            block: [N]usize,
            sides: [N]Side,

            pub fn next(self: ExtentIterator) ?[N]isize {
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
        pub fn extentOffsets(self: Self, block: [N]usize) ExtentIterator {
            var size: [N]usize = undefined;

            for (0..N) |i| {
                if (self.sides[i] == .left or self.sides[i] == .right) {
                    size[i] = O;
                } else {
                    size[i] = 1;
                }
            }

            const space = IndexSpace.fromSize(size);

            return .{
                .inner = space.cartesianIndices(),
                .block = block,
                .sides = self.sides,
            };
        }

        // ************************
        // Constructors ***********
        // ************************

        pub fn regions() [3 ^ N]Region(N) {
            var regs: [3 ^ N]Region(N) = undefined;

            const space = IndexSpace.fromSize([1]usize{3} ** N);

            var indices = space.cartesianIndices();

            var i: usize = 0;

            while (indices.next()) |cart| {
                comptime var sides: [N]Side = undefined;

                inline for (0..N) |axis| {
                    sides[axis] = @enumFromInt(cart[axis]);
                }

                regs[i] = .{ .sides = sides };
                i += 1;
            }

            return regs;
        }

        pub fn orderedRegions() [3 ^ N]Region(N) {
            var regs: [3 ^ N]Region(N) = regions();
            heap(Region(N), &regs, void, lessThanFn);
            return regs;
        }

        fn lessThanFn(_: void, lhs: Region(N), rhs: Region(N)) bool {
            return lhs.adjacency() < rhs.adjacency();
        }
    };
}
