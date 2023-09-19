const std = @import("std");

const basis = @import("../basis/basis.zig");
const system = @import("system.zig");

/// Wraps a stencil space, output field, input system, and cell, to provide a consistent
/// interface to write operators.
pub fn ApproxEngine(comptime N: usize, comptime O: usize, comptime Input: type) type {
    if (!system.isSystem(Input)) {
        @compileError("Input type must be a system");
    }

    return struct {
        space: StencilSpace,
        output: []const f64,
        input: Input,
        cell: [N]usize,

        // Aliases
        const Self = @This();
        const StencilSpace = basis.StencilSpace(N, 2 * O, O);

        // Public types
        pub const FieldEnum = system.SystemFieldEnum(Input);

        /// Computes the position of the cell.
        pub fn position(self: Self) [N]f64 {
            return self.space.position(self.cell);
        }

        /// Returns the value of the field at the current cell.
        pub fn value(self: Self, comptime field: FieldEnum) f64 {
            const f: []const f64 = system.systemField(Input, self.input, field);
            return self.valueField(f);
        }

        /// Returns the gradient of the field at the current cell.
        pub fn gradient(self: Self, comptime field: FieldEnum) [N]f64 {
            const f: []const f64 = system.systemField(Input, self.input, field);
            return self.gradientField(f);
        }

        /// Returns the hessian of the field at the current cell.
        pub fn hessian(self: Self, comptime field: FieldEnum) [N][N]f64 {
            const f: []const f64 = system.systemField(Input, self.input, field);
            return self.hessianField(f);
        }

        /// Returns the laplacian of the field at the current cell.
        pub fn laplacian(self: Self, comptime field: FieldEnum) f64 {
            const f: []const f64 = system.systemField(Input, self.input, field);
            return self.laplacianField(f);
        }

        /// Returns the value of the output field at the current cell.
        pub fn valueOp(self: Self) f64 {
            return self.valueField(self.output);
        }

        /// Returns the gradient of the output field at the current cell.
        pub fn gradientOp(self: Self) [N]f64 {
            return self.gradientField(self.output);
        }

        /// Returns the hessian of the output field at the current cell.
        pub fn hessianOp(self: Self) [N][N]f64 {
            return self.hessianField(self.output);
        }

        /// Returns the laplacian of the output field at the current cell.
        pub fn laplacianOp(self: Self) f64 {
            return self.laplacianField(self.output);
        }

        /// Returns the value diagonal coefficient.
        pub fn valueDiagonal(self: Self) f64 {
            return self.space.valueDiagonal();
        }

        /// Returns the gradient diagonal coefficients.
        pub fn gradientDiagonal(self: Self) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] += 1;

                result[i] = self.space.derivativeDiagonal(ranks);
            }

            return result;
        }

        /// Returns the hessian diagonal coefficients.
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

        /// Returns the laplacian diagonal coefficients.
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

/// A trait which checks if a type is a mesh operator. Such a type follows the following set of declarations.
/// ```
/// const Operator = struct {
///     pub const Input = struct {
///         field1: []const f64,
///         field2: []const f64,
///         // ...
///     };
///
///     pub fn apply(self: Operator, engine: ApproxEngine(2, 2, Input)) f64 {
///         // ...
///     }
///
///     pub fn applyDiagonal(self: Operator, engine: ApproxEngine(2, 2, Input)) f64 {
///         // ...
///     }
/// };
/// ```
pub fn isMeshOperator(comptime N: usize, comptime O: usize) fn (type) bool {
    const hasFn = std.meta.trait.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (!(@hasDecl(T, "Input") and T.Input == type and system.isConstSystem(T.Input))) {
                return false;
            }

            if (!(hasFn("apply")(T) and @TypeOf(T.apply) == fn (T, ApproxEngine(N, O, T.Input)) f64)) {
                return false;
            }

            if (!(hasFn("applyDiagonal")(T) and @TypeOf(T.applyDiagonal) == fn (T, ApproxEngine(N, O, T.Input)) f64)) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}
