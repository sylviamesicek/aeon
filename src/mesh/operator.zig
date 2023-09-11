const std = @import("std");
const meta = std.meta;
const trait = meta.trait;

const basis = @import("../basis/basis.zig");

pub fn ApproxEngine(comptime N: usize, comptime O: usize, comptime Input: type) type {
    std.debug.assert(isInputStruct(Input));

    const ILen: usize = meta.fields(Input).len;

    return struct {
        space: StencilSpace,
        output: []const f64,
        inputs: [ILen][]const f64,
        cell: [N]usize,

        pub const Field = meta.FieldEnum(Input);

        const Self = @This();
        const StencilSpace = basis.StencilSpace(N, O);

        pub fn position(self: Self) [N]f64 {
            return self.space.position(self.cell);
        }

        pub fn value(self: Self, comptime field: Field) f64 {
            const field_index = @intFromEnum(field);
            return self.valueField(self.inputs[field_index]);
        }

        pub fn gradient(self: Self, comptime field: Field) [N]f64 {
            const field_index = @intFromEnum(field);
            return self.gradientField(self.inputs[field_index]);
        }

        pub fn hessian(self: Self, comptime field: Field) [N][N]f64 {
            const field_index = @intFromEnum(field);
            return self.hessianField(self.inputs[field_index]);
        }

        pub fn laplacian(self: Self, comptime field: Field) f64 {
            const field_index = @intFromEnum(field);
            return self.laplacianField(self.inputs[field_index]);
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

const is = trait.is;
const hasFn = trait.hasFn;
const TraitFn = std.meta.trait.TraitFn;

pub fn isInputStruct(comptime T: type) bool {
    switch (@typeInfo(T)) {
        .Struct => |info| {
            for (info.fields) |field| {
                if (field.type != []const f64) {
                    return false;
                }
            }

            return true;
        },
        else => return false,
    }
}

pub fn isOperator(comptime N: usize, comptime O: usize) TraitFn {
    const Closure = struct {
        fn t(comptime T: type) bool {
            if (!(@hasDecl(T, "Input") and T.Input == type and isInputStruct(T.Input))) {
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
