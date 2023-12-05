const std = @import("std");

const geometry = @import("../geometry/geometry.zig");
const nodes = @import("../nodes/nodes.zig");

/// An interface for taking values, gradients, and hessians of functions in physical space
/// while being mesh agnostic.
pub fn Engine(comptime N: usize, comptime M: usize) type {
    return struct {
        bounds: RealBox,
        space: NodeSpace,
        cell: [N]usize,
        offset: usize,
        total: usize,
        diag: bool = false,

        const NodeSpace = nodes.NodeSpace(N, M);
        const RealBox = geometry.RealBox(N);

        pub fn position(self: @This()) [N]f64 {
            const pos = self.space.cellPosition(self.cell);
            return self.bounds.transformPos(pos);
        }

        pub fn value(self: @This(), field: []const f64) f64 {
            if (self.diag) {
                return 1.0;
            } else {
                return self.space.value(self.cell, field[self.offset .. self.offset + self.total]);
            }
        }

        pub fn op(self: @This(), comptime ranks: [N]usize, field: []const usize) f64 {
            const v = if (self.diag) self.space.opDiagonal(ranks) else self.space.op(ranks, self.cell, field[self.offset .. self.offset + self.total]);

            return self.bounds.transformOp(ranks, v);
        }

        pub fn gradient(self: @This(), field: []const usize) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] += 1;

                result[i] = self.op(ranks, field);
            }

            return result;
        }

        /// Computes the hessian of the given field.
        pub fn hessian(self: @This(), field: []const f64) [N][N]f64 {
            var result: [N][N]f64 = undefined;

            inline for (0..N) |i| {
                inline for (0..N) |j| {
                    comptime var ranks: [N]usize = [1]usize{0} ** N;
                    ranks[i] += 1;
                    ranks[j] += 1;

                    result[i][j] = self.op(ranks, field);
                }
            }

            return result;
        }

        /// Computes the laplacian of the given field.
        pub fn laplacian(self: @This(), field: []const f64) f64 {
            var result: f64 = 0.0;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] = 2;

                result += self.op(ranks, field);
            }

            return result;
        }
    };
}

/// A trait for defining operators.
pub fn isOperator(comptime N: usize, comptime M: usize) fn (type) bool {
    const hasFn = std.meta.trait.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (comptime !(hasFn("apply")(T) and @TypeOf(T.apply) == fn (T, Engine(N, M), []const f64) f64)) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}
