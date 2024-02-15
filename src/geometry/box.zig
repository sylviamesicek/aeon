const std = @import("std");
const IndexSpace = @import("index.zig").IndexSpace;

const assert = std.debug.assert;

const axis_ = @import("axis.zig");

/// An N-dimensional subregion of some larger index space.
pub fn IndexBox(comptime N: usize) type {
    return struct {
        origin: [N]usize,
        size: [N]usize,

        const AxisMask = axis_.AxisMask(N);

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

        pub fn split(self: @This(), mask: AxisMask) RealBox(N) {
            const cart = mask.unpack();

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

/// An n-dimension AABB in physical space, often used to denote rectangular domains.
pub fn RealBox(comptime N: usize) type {
    return struct {
        origin: [N]f64,
        size: [N]f64,

        const AxisMask = axis_.AxisMask(N);

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

        /// Computes the hyperquadrant assocaited with the mask.
        pub fn split(self: @This(), mask: AxisMask) RealBox(N) {
            const cart = mask.unpack();

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

        /// Transforms a local position in the box to a global position.
        pub fn localToGlobal(self: @This(), x: [N]f64) [N]f64 {
            var result: [N]f64 = undefined;

            for (0..N) |i| {
                result[i] = self.origin[i] + self.size[i] * x[i];
            }

            return result;
        }
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
