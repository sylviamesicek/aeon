const std = @import("std");

const geometry = @import("../geometry/geometry.zig");

const nodes = @import("nodes.zig");

/// An interface extending a `NodeSpace` that allows one to take properly transformed
/// gradients, hessians, and laplacians of some larger node vector.
pub fn Engine(comptime N: usize, comptime M: usize) type {
    return struct {
        space: NodeSpace,
        bounds: RealBox,
        cell: [N]usize,
        start: usize,
        end: usize,

        const NodeSpace = nodes.NodeSpace(N, M);
        const RealBox = geometry.RealBox(N);

        pub fn position(self: @This()) [N]f64 {
            const pos = self.space.cellPosition(self.cell);
            return self.bounds.transformPos(pos);
        }

        pub fn op(self: @This(), comptime ranks: [N]usize, field: []const f64) f64 {
            const v = self.space.op(ranks, self.cell, field[self.start..self.end]);
            return self.bounds.transformOp(ranks, v);
        }

        pub fn opDiag(self: @This(), comptime ranks: [N]usize) f64 {
            const v = self.space.opDiagonal(ranks);
            return self.bounds.transformOp(ranks, v);
        }

        pub fn value(self: @This(), field: []const f64) f64 {
            return self.space.value(self.cell, field[self.start..self.end]);
        }

        pub fn valueDiag(self: @This()) f64 {
            _ = self;
            return 1.0;
        }

        pub fn gradient(self: @This(), field: []const f64) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] += 1;

                result[i] = self.op(ranks, field);
            }

            return result;
        }

        pub fn gradientDiag(self: @This()) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] += 1;

                result[i] = self.opDiag(ranks);
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

        pub fn hessianDiag(self: @This()) [N][N]f64 {
            var result: [N][N]f64 = undefined;

            inline for (0..N) |i| {
                inline for (0..N) |j| {
                    comptime var ranks: [N]usize = [1]usize{0} ** N;
                    ranks[i] += 1;
                    ranks[j] += 1;

                    result[i][j] = self.opDiag(ranks);
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

        pub fn laplacianDiag(self: @This()) f64 {
            var result: f64 = 0.0;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] = 2;

                result += self.opDiag(ranks);
            }

            return result;
        }
    };
}

/// A trait for defining operators.
pub fn isOperator(comptime N: usize, comptime M: usize) fn (type) bool {
    const hasFn = std.meta.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (comptime !(hasFn("apply")(T) and @TypeOf(T.apply) == fn (T, Engine(N, M), []const f64) f64)) {
                return false;
            }

            if (comptime !(hasFn("applyDiag")(T) and @TypeOf(T.applyDiag) == fn (T, Engine(N, M)) f64)) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}

pub fn isProjection(comptime N: usize, comptime M: usize) fn (type) bool {
    const hasFn = std.meta.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (comptime !(hasFn("project")(T) and @TypeOf(T.project) == fn (T, Engine(N, M)) f64)) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}
