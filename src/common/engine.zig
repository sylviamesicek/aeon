const std = @import("std");

const geometry = @import("../geometry/geometry.zig");
const utils = @import("../utils.zig");
const Range = utils.Range;

const nodes = @import("nodes.zig");

/// An interface extending a `NodeSpace` that allows one to take properly transformed
/// gradients, hessians, and laplacians of some larger node vector.
pub fn Engine(comptime N: usize, comptime M: usize, comptime O: usize) type {
    return struct {
        space: NodeSpace,
        node: [N]isize,
        range: Range,

        const NodeSpace = nodes.NodeSpace(N, M);
        const Operator = nodes.NodeOperator(N);

        const RealBox = geometry.RealBox(N);

        /// Retrieves the position of the current vertex.
        pub fn position(self: @This()) [N]f64 {
            return self.space.nodePosition(self.node);
        }

        pub fn eval(self: @This(), comptime ranks: [N]usize, field: []const f64) f64 {
            return self.space.eval(Operator.centered(O, ranks), self.node, field[self.range.start..self.range.end]);
        }

        pub fn evalDiag(self: @This(), comptime ranks: [N]usize) f64 {
            return self.space.evalCoef(Operator.centered(O, ranks), [1]isize{0} ** N);
        }

        pub fn value(self: @This(), field: []const f64) f64 {
            return self.space.value(self.node, field[self.range.start..self.range.end]);
        }

        pub fn valueDiag(_: @This()) f64 {
            return 1.0;
        }

        pub fn gradient(self: @This(), field: []const f64) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] += 1;

                result[i] = self.eval(ranks, field);
            }

            return result;
        }

        pub fn gradientDiag(self: @This()) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] += 1;

                result[i] = self.evalDiag(ranks);
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

                    result[i][j] = self.eval(ranks, field);
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

                    result[i][j] = self.evalDiag(ranks);
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

                result += self.eval(ranks, field);
            }

            return result;
        }

        pub fn laplacianDiag(self: @This()) f64 {
            var result: f64 = 0.0;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] = 2;

                result += self.evalDiag(ranks);
            }

            return result;
        }
    };
}
