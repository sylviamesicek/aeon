const std = @import("std");

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const system = @import("../system.zig");

const boundary = @import("boundary.zig");
const SystemBoundaryCondition = boundary.SystemBoundaryCondition;

/// Wraps a stencil space, output field, input system, and cell, to provide a consistent
/// interface to write operators.
pub fn Engine(comptime N: usize, comptime O: usize) type {
    return struct {
        space: StencilSpace,
        cell: [N]isize,

        // Aliases
        const Self = @This();
        const StencilSpace = basis.StencilSpace(N, O);

        /// Computes the position of the cell.
        pub fn position(self: Self) [N]f64 {
            return self.space.position(self.cell);
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

        /// Computes the value of the given field.
        pub fn value(self: Self, field: []const f64) f64 {
            return self.space.value(self.cell, field);
        }

        /// Computes the gradient of the given field.
        pub fn gradient(self: Self, field: []const f64) [N]f64 {
            var result: [N]f64 = undefined;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] += 1;

                result[i] = self.space.derivative(ranks, self.cell, field);
            }

            return result;
        }

        /// Computes the hessian of the given field.
        pub fn hessian(self: Self, field: []const f64) [N][N]f64 {
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

        /// Computes the laplacian of the given field.
        pub fn laplacian(self: Self, field: []const f64) f64 {
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

/// An `Engine` which can compute operations on elements of a given context system.
pub fn FunctionEngine(comptime N: usize, comptime O: usize, comptime Input: type) type {
    if (!system.isSystem(Input)) {
        @compileError("Input must satisfy isSystem trait.");
    }

    return struct {
        inner: Engine(N, O),
        input: system.SystemSliceConst(Input),

        // Aliases
        const Self = @This();

        pub fn position(self: Self) [N]f64 {
            return self.inner.position();
        }

        /// Returns the value of the field at the current cell.
        pub fn value(self: Self, comptime field: Input) f64 {
            const f: []const f64 = @field(self.input, @tagName(field));
            return self.inner.value(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn gradient(self: Self, comptime field: Input) [N]f64 {
            const f: []const f64 = @field(self.input, @tagName(field));
            return self.inner.gradient(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn hessian(self: Self, comptime field: Input) [N][N]f64 {
            const f: []const f64 = @field(self.input, @tagName(field));
            return self.inner.hessian(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn laplacian(self: Self, comptime field: Input) f64 {
            const f: []const f64 = @field(self.input, @tagName(field));
            return self.inner.laplacian(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn valueDiagonal(self: Self) f64 {
            return self.inner.valueDiagonal();
        }

        /// Returns the value of the field at the current cell.
        pub fn gradientDiagonal(self: Self) [N]f64 {
            return self.inner.gradientDiagonal();
        }

        /// Returns the value of the field at the current cell.
        pub fn hessianDiagonal(self: Self) [N][N]f64 {
            return self.inner.hessianDiagonal();
        }

        pub fn laplacianDiagonal(self: Self) f64 {
            return self.inner.laplacianDiagonal();
        }
    };
}

/// An `Engine` which can compute operations on elements of a given context system, as well as on
/// a given operated system
pub fn OperatorEngine(comptime N: usize, comptime O: usize, comptime Context: type, comptime Operated: type) type {
    if (!system.isSystem(Context)) {
        @compileError("Context must satisfy isSystem trait.");
    }

    if (!system.isSystem(Operated)) {
        @compileError("Operated must satisfy isSystem trait.");
    }

    return struct {
        inner: Engine(N, O),
        context: system.SystemSliceConst(Context),
        operated: system.SystemSliceConst(Operated),

        // Aliases
        const Self = @This();

        pub fn position(self: Self) [N]f64 {
            return self.inner.position();
        }

        /// Returns the value of the field at the current cell.
        pub fn value(self: Self, comptime field: Context) f64 {
            const f: []const f64 = @field(self.context, @tagName(field));
            return self.inner.value(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn gradient(self: Self, comptime field: Context) [N]f64 {
            const f: []const f64 = @field(self.context, @tagName(field));
            return self.inner.gradient(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn hessian(self: Self, comptime field: Context) [N][N]f64 {
            const f: []const f64 = @field(self.context, @tagName(field));
            return self.inner.hessian(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn laplacian(self: Self, comptime field: Context) f64 {
            const f: []const f64 = @field(self.context, @tagName(field));
            return self.inner.laplacian(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn valueOp(self: Self, comptime field: Operated) f64 {
            const f: []const f64 = @field(self.operated, @tagName(field));
            return self.inner.value(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn gradientOp(self: Self, comptime field: Operated) [N]f64 {
            const f: []const f64 = @field(self.operated, @tagName(field));
            return self.inner.gradient(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn hessianOp(self: Self, comptime field: Operated) [N][N]f64 {
            const f: []const f64 = @field(self.operated, @tagName(field));
            return self.inner.hessian(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn laplacianOp(self: Self, comptime field: Operated) f64 {
            const f: []const f64 = @field(self.operated, @tagName(field));
            return self.inner.laplacian(f);
        }

        /// Returns the value of the field at the current cell.
        pub fn valueDiagonal(self: Self) f64 {
            return self.inner.valueDiagonal();
        }

        /// Returns the value of the field at the current cell.
        pub fn gradientDiagonal(self: Self) [N]f64 {
            return self.inner.gradientDiagonal();
        }

        /// Returns the value of the field at the current cell.
        pub fn hessianDiagonal(self: Self) [N][N]f64 {
            return self.inner.hessianDiagonal();
        }

        /// Returns the value of the field at the current cell.
        pub fn laplacianDiagonal(self: Self) f64 {
            return self.inner.laplacianDiagonal();
        }
    };
}

pub fn EngineType(comptime N: usize, comptime O: usize, comptime T: type) type {
    if (comptime isMeshOperator(N, O)(T)) {
        return OperatorEngine(N, O, T.Context, T.System);
    } else if (comptime isMeshFunction(N, O)(T)) {
        return FunctionEngine(N, O, T.Input);
    } else {
        @compileError("EngineType may only be called on types which satisfy isMeshOperator or isMeshFunction traits.");
    }
}

/// A trait which checks if a type is a mesh operator. Such a type follows the following set of declarations.
/// ```
/// const Operator = struct {
///     pub const Context = enum {
///         field1,
///         field2,
///         // ...
///     };
///
///     pub const System = enum {
///         result,
///     };
///
///     pub fn apply(self: Operator, engine: OperatorEngine(2, 2, Context, System)) SystemValue(System) {
///         // ...
///     }
///
///     pub fn applyDiagonal(self: Operator, engine: OperatorEngine(2, 2, Context, System)) SystemValue(System) {
///         // ...
///     }
/// };
/// ```
pub fn isMeshOperator(comptime N: usize, comptime O: usize) fn (type) bool {
    const hasFn = std.meta.trait.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (comptime !(@hasDecl(T, "Context") and @TypeOf(T.Context) == type and system.isSystem(T.Context))) {
                return false;
            }

            if (comptime !(@hasDecl(T, "System") and @TypeOf(T.System) == type and system.isSystem(T.System))) {
                return false;
            }

            if (comptime !(hasFn("apply")(T) and @TypeOf(T.apply) == fn (T, OperatorEngine(N, O, T.Context, T.System)) system.SystemValue(T.System))) {
                return false;
            }

            if (comptime !(hasFn("applyDiagonal")(T) and @TypeOf(T.applyDiagonal) == fn (T, OperatorEngine(N, O, T.Context, T.System)) system.SystemValue(T.System))) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}

/// A trait which checks if a type is a mesh function. Such a type follows the following set of declarations.
/// ```
/// const Function = struct {
///     pub const Input = enum {
///         field1,
///         field2,
///         // ...
///     };
///
///     pub const Output = enum {
///         x,
///         y,
///     }
///
///     pub fn value(self: Operator, engine: FunctionEngine(2, 2, Input)) SystemValue(Output) {
///         // ...
///     }
/// };
/// ```
pub fn isMeshFunction(comptime N: usize, comptime O: usize) fn (type) bool {
    const hasFn = std.meta.trait.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (comptime !(@hasDecl(T, "Input") and @TypeOf(T.Input) == type and system.isSystem(T.Input))) {
                return false;
            }

            if (comptime !(@hasDecl(T, "Output") and @TypeOf(T.Output) == type and system.isSystem(T.Output))) {
                return false;
            }

            if (comptime !(hasFn("value")(T) and @TypeOf(T.value) == fn (T, FunctionEngine(N, O, T.Input)) system.SystemValue(T.Output))) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}

/// A trait which checks if a type is a mesh boundary. Such a type follows the following set of declarations.
/// ```
/// const Boundary = struct {
///     pub const System = enum {
///         field1,
///         field2,
///     }
///
///     pub fn condition(self: Boundary, pos: [2]f64, face: Face) SystemStruct(System, BoundaryCondition) {
///         // ...
///     }
/// }
/// ```
pub fn isMeshBoundary(comptime N: usize) fn (type) bool {
    const hasFn = std.meta.trait.hasFn;
    const Face = geometry.Face(N);

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (comptime !(@hasDecl(T, "System") and @TypeOf(T.System) == type and system.isSystem(T.System))) {
                return false;
            }

            if (comptime !(hasFn("condition")(T) and @TypeOf(T.condition) == fn (T, [N]f64, Face) SystemBoundaryCondition(T.System))) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}
