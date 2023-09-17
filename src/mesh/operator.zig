const std = @import("std");

const basis = @import("../basis/basis.zig");
const system = @import("system.zig");

/// Wraps a stencil space, output field, input system, and cell, to provide a consistent
/// interface to write operators.
pub fn ApproxEngine(comptime N: usize, comptime O: usize, comptime Input: type) type {
    if (!system.isConstSystem(Input)) {
        @compileError("Input type must be a const system");
    }

    return struct {
        space: StencilSpace,
        output: []const f64,
        inputs: Input,
        cell: [N]usize,

        pub const FieldEnum = system.SystemFieldEnum(Input);

        const Self = @This();
        const StencilSpace = basis.StencilSpace(N, O);

        pub fn position(self: Self) [N]f64 {
            return self.space.position(self.cell);
        }

        pub fn value(self: Self, comptime field: FieldEnum) f64 {
            const f: []const f64 = system.systemField(Input, self.inputs, field);
            return self.valueField(f);
        }

        pub fn gradient(self: Self, comptime field: FieldEnum) [N]f64 {
            const f: []const f64 = system.systemField(Input, self.inputs, field);
            return self.gradientField(f);
        }

        pub fn hessian(self: Self, comptime field: FieldEnum) [N][N]f64 {
            const f: []const f64 = system.systemField(Input, self.inputs, field);
            return self.hessianField(f);
        }

        pub fn laplacian(self: Self, comptime field: FieldEnum) f64 {
            const f: []const f64 = system.systemField(Input, self.inputs, field);
            return self.laplacianField(f);
        }

        pub fn valueOp(self: Self) f64 {
            return self.valueField(self.output);
        }

        pub fn gradientOp(self: Self) [N]f64 {
            return self.gradientField(self.output);
        }

        pub fn hessianOp(self: Self) [N][N]f64 {
            return self.hessianField(self.output);
        }

        pub fn laplacianOp(self: Self) f64 {
            return self.laplacianField(self.output);
        }

        pub fn valueDiagonal(self: Self) f64 {
            return self.space.valueDiagonal();
        }

        pub fn gradientDiagonal(self: Self) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] += 1;

                result[i] = self.space.derivativeDiagonal(ranks);
            }

            return result;
        }

        pub fn hessianDiagonal(self: Self) [N]f64 {
            var result: [N][N]f64 = undefined;

            inline for (0..N) |i| {
                inline for (0..N) |j| {
                    comptime var ranks: [N]usize = [1]usize{0} ** N;
                    ranks[i] += 1;
                    ranks[j] += 1;

                    result[i][j] = self.space.derivativeDiagonal(ranks);
                }
            }

            return result;
        }

        pub fn laplacianDiagonal(self: Self) [N]f64 {
            var result: f64 = 0.0;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] = 2;

                result += self.space.derivativeDiagonal(ranks);
            }

            return result;
        }

        fn valueField(self: Self, field: []const f64) f64 {
            return self.space.value(self.cell, field);
        }

        fn gradientField(self: Self, field: []const f64) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] += 1;

                result[i] = self.space.derivative(ranks, self.cell, field);
            }

            return result;
        }

        fn hessianField(self: Self, field: []const f64) [N][N]f64 {
            var result: [N][N]f64 = undefined;

            inline for (0..N) |i| {
                inline for (0..N) |j| {
                    comptime var ranks: [N]usize = [1]usize{0} ** N;
                    ranks[i] += 1;
                    ranks[j] += 1;

                    result[i][j] = self.space.derivative(ranks, self.cell, field);
                }
            }

            return result;
        }

        fn laplacianField(self: Self, field: []const f64) f64 {
            var result: f64 = 0.0;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] = 2;

                result += self.space.derivative(ranks, self.cell, field);
            }

            return result;
        }
    };
}

const is = std.meta.trait.is;
const hasFn = std.meta.trait.hasFn;
const TraitFn = std.meta.trait.TraitFn;

pub fn isOperator(comptime N: usize, comptime O: usize) TraitFn {
    const Closure = struct {
        fn t(comptime T: type) bool {
            if (!(@hasDecl(T, "Input") and T.Input == type and system.isConstSystem(T.Input))) {
                return false;
            }

            if (!(hasFn("apply")(T) and @TypeOf(T.apply) == fn (T, ApproxEngine(N, O, T.input)) f64)) {
                return false;
            }

            if (!(hasFn("applyDiagonal")(T) and @TypeOf(T.applyDiagonal) == fn (T, ApproxEngine(N, O, T.input)) f64)) {
                return false;
            }

            return true;
        }
    };
    return Closure.t;
}
