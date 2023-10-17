const std = @import("std");

const basis = @import("../basis/basis.zig");

const geometry = @import("../geometry/geometry.zig");
const Face = geometry.Face;

const system = @import("../system.zig");
const SystemSlice = system.SystemSlice;
const SystemSliceConst = system.SystemSliceConst;
const isSystem = system.isSystem;

const boundary = @import("boundary.zig");
const SystemBoundaryCondition = boundary.SystemBoundaryCondition;

pub const EngineSetting = enum {
    normal,
    diagonal,
};

/// An `Engine` which can compute operations on elements of a given context system, as well as on
/// a given operated system. This is an API for bridging the gap between systems (consisting of
/// multiple fields and stencil spaces (which only operates on one field at a time).
pub fn Engine(
    comptime N: usize,
    comptime O: usize,
    comptime Setting: EngineSetting,
    comptime System: type,
    comptime Context: type,
) type {
    if (!isSystem(System)) {
        @compileError("Operated must satisfy isSystem trait.");
    }

    if (!isSystem(Context)) {
        @compileError("Context must satisfy isSystem trait.");
    }

    return struct {
        inner: EngineImpl(N, O),
        sys: SystemSliceConst(System),
        ctx: SystemSliceConst(Context),

        // Aliases
        const Self = @This();
        const StencilSpace = basis.StencilSpace(N, O);

        pub fn new(space: StencilSpace, cell: [N]isize, sys: system.SystemSliceConst(System), ctx: system.SystemSliceConst(Context)) Self {
            return .{
                .inner = .{
                    .space = space,
                    .cell = cell,
                },
                .sys = sys,
                .ctx = ctx,
            };
        }

        pub fn position(self: Self) [N]f64 {
            return self.inner.position();
        }

        /// Returns the value of the field at the current cell.
        pub fn valueCtx(self: Self, comptime field: Context) f64 {
            return self.inner.value(self.ctx.field(field));
        }

        /// Returns the value of the field at the current cell.
        pub fn gradientCtx(self: Self, comptime field: Context) [N]f64 {
            return self.inner.gradient(self.ctx.field(field));
        }

        /// Returns the value of the field at the current cell.
        pub fn hessianCtx(self: Self, comptime field: Context) [N][N]f64 {
            return self.inner.hessian(self.ctx.field(field));
        }

        /// Returns the value of the field at the current cell.
        pub fn laplacianCtx(self: Self, comptime field: Context) f64 {
            return self.inner.laplacian(self.ctx.field(field));
        }

        /// Returns the value of the field at the current cell.
        pub fn valueSys(self: Self, comptime field: System) f64 {
            return switch (Setting) {
                .normal => self.inner.value(self.sys.field(field)),
                .diagonal => self.inner.valueDiagonal(),
            };
        }

        /// Returns the value of the field at the current cell.
        pub fn gradientSys(self: Self, comptime field: System) [N]f64 {
            return switch (Setting) {
                .normal => self.inner.gradient(self.sys.field(field)),
                .diagonal => self.inner.gradientDiagonal(),
            };
        }

        /// Returns the value of the field at the current cell.
        pub fn hessianSys(self: Self, comptime field: System) [N][N]f64 {
            return switch (Setting) {
                .normal => self.inner.hessian(self.sys.field(field)),
                .diagonal => self.inner.hessianDiagonal(),
            };
        }

        /// Returns the value of the field at the current cell.
        pub fn laplacianSys(self: Self, comptime field: System) f64 {
            return switch (Setting) {
                .normal => self.inner.laplacian(self.sys.field(field)),
                .diagonal => self.inner.laplacianDiagonal(),
            };
        }
    };
}

fn EngineImpl(comptime N: usize, comptime O: usize) type {
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

                result[i] = self.space.derivativeDiagonal(O, ranks);
            }

            return result;
        }

        /// Returns the hessian diagonal coefficients.
        pub fn hessianDiagonal(self: Self) [N][N]f64 {
            var result: [N][N]f64 = undefined;

            inline for (0..N) |i| {
                inline for (0..N) |j| {
                    comptime var ranks: [N]usize = [1]usize{0} ** N;
                    ranks[i] += 1;
                    ranks[j] += 1;

                    result[i][j] = self.space.derivativeDiagonal(O, ranks);
                }
            }

            return result;
        }

        /// Returns the laplacian diagonal coefficients.
        pub fn laplacianDiagonal(self: Self) f64 {
            var result: f64 = 0.0;

            inline for (0..N) |i| {
                comptime var ranks: [N]usize = [1]usize{0} ** N;
                ranks[i] = 2;

                result += self.space.derivativeDiagonal(O, ranks);
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

                result[i] = self.space.derivative(O, ranks, self.cell, field);
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

                    result[i][j] = self.space.derivative(O, ranks, self.cell, field);
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

                result += self.space.derivative(O, ranks, self.cell, field);
            }

            return result;
        }
    };
}
