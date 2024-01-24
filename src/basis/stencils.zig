const std = @import("std");
const geometry = @import("../geometry/geometry.zig");

const lagrange = @import("lagrange.zig");

/// This namespace provides several functions for computing cell centered interpolation, differentiation,
/// prolongation, restriction, and boundary stencils. Here `M` corresponds to the number of support points
/// to either side of the center of the stencil (or the number of interior support points in the case of
/// boundary stencils).
pub fn Stencils(comptime M: usize) type {
    return struct {
        const Lagrange = lagrange.Lagrange(f128);

        /// Computes a value stencil of order `2M` (for lagrange polynomials, the stencil is always 1.0 at the central
        /// support point and 0.0 everywhere else).
        pub fn value() [2 * M + 1]f64 {
            var result: [2 * M + 1]f128 = undefined;
            Lagrange.value(&centeredCellGrid(), 0.0, &result);
            return floatCastSlice(2 * M + 1, result);
        }

        /// Computes a first derivative stencil of order `2M`.
        pub fn derivative() [2 * M + 1]f64 {
            var result: [2 * M + 1]f128 = undefined;
            Lagrange.derivative(&centeredCellGrid(), 0.0, &result);
            return floatCastSlice(2 * M + 1, result);
        }

        /// Computes a second derivative stencil of order `2M`.
        pub fn secondDerivative() [2 * M + 1]f64 {
            var result: [2 * M + 1]f128 = undefined;
            Lagrange.secondDerivative(&centeredCellGrid(), 0.0, &result);
            return floatCastSlice(2 * M + 1, result);
        }

        /// Computes an interpolation stencil of order `M + E - 1` for the boundary
        /// of a cell. `E` denotes the number of exterior support points.
        pub fn boundaryValue(comptime E: usize) [M + E]f64 {
            if (comptime E > M) {
                @compileError("E must be <= M when computing boundary stencils.");
            }

            var result: [M + E]f128 = undefined;
            Lagrange.value(&boundaryGrid(E), 0.0, &result);
            return floatCastSlice(M + E, result);
        }

        /// Computes an differentiation stencil of order `M + E - 1` for the boundary
        /// of a cell. `E` denotes the number of exterior support points.
        pub fn boundaryFlux(comptime E: usize) [M + E]f64 {
            if (comptime E > M) {
                @compileError("E must be <= M when computing boundary stencils.");
            }

            var result: [M + E]f128 = undefined;
            Lagrange.derivative(&boundaryGrid(E), 0.0, &result);
            return floatCastSlice(M + E, result);
        }

        /// Computes a prolongation stencil which is centered on a cell. `side` denotes which
        /// subcell to prolong to.
        pub fn prolongCell(comptime side: bool) [2 * M + 1]f64 {
            var result: [2 * M + 1]f128 = undefined;
            Lagrange.value(&centeredCellGrid(), if (side) 0.25 else -0.25, &result);
            return floatCastSlice(2 * M + 1, result);
        }

        /// Computes a prolongation stencil which is centered on a vertex.
        pub fn prolongVertex(comptime side: bool) [2 * M]f64 {
            var result: [2 * M]f128 = undefined;
            Lagrange.value(&centeredVertexGrid(), if (side) 0.25 else -0.25, &result);
            return floatCastSlice(2 * M, result);
        }

        /// Computes a restriction stencil.
        pub fn restrict() [2 * M]f64 {
            var result: [2 * M]f128 = undefined;
            Lagrange.value(&centeredVertexGrid(), 0.0, &result);
            return floatCastSlice(2 * M, result);
        }

        // ************************
        // Helpers ****************
        // ************************

        fn centeredCellGrid() [2 * M + 1]f128 {
            return cellGrid(M, M);
        }

        fn centeredVertexGrid() [2 * M]f128 {
            return vertexGrid(M, M);
        }

        fn boundaryGrid(comptime E: usize) [M + E]f128 {
            return vertexGrid(M, E);
        }

        fn cellGrid(comptime L: usize, comptime R: usize) [L + R + 1]f128 {
            var grid: [L + R + 1]f128 = undefined;

            for (0..(L + R + 1)) |i| {
                grid[i] = @as(f128, @floatFromInt(i)) - @as(f128, @floatFromInt(L));
            }

            return grid;
        }

        fn vertexGrid(comptime L: usize, comptime R: usize) [L + R]f128 {
            var grid: [L + R]f128 = undefined;

            for (0..(L + R)) |i| {
                grid[i] = @as(f128, @floatFromInt(i)) + 0.5 - @as(f128, @floatFromInt(L));
            }

            return grid;
        }

        /// Casts arrays of `[Len]f128` -> `[Len]f64` using the `@floatCast` builtin.
        fn floatCastSlice(comptime Len: usize, slice: [Len]f128) [Len]f64 {
            var result: [Len]f64 = undefined;

            for (0..Len) |i| {
                result[i] = @floatCast(slice[i]);
            }

            return result;
        }
    };
}

test "stencil grids" {
    const S2 = Stencils(2);

    const expectEqualDeep = std.testing.expectEqualDeep;

    try expectEqualDeep([_]f128{ -1.0, 0.0, 1.0, 2.0 }, S2.cellGrid(1, 2));
    try expectEqualDeep([_]f128{ -1.5, -0.5, 0.5 }, S2.vertexGrid(2, 1));
    try expectEqualDeep([_]f128{ -2.0, -1.0, 0.0, 1.0, 2.0 }, S2.centeredCellGrid());
    try expectEqualDeep([_]f128{ -1.5, -0.5, 0.5, 1.5 }, S2.centeredVertexGrid());
}

test "stencils" {
    const S0 = Stencils(0);
    const S1 = Stencils(1);
    const S2 = Stencils(2);

    const expectEqualDeep = std.testing.expectEqualDeep;

    try expectEqualDeep([_]f64{1.0}, S0.prolongCell(false));
    try expectEqualDeep([_]f64{1.0}, S0.prolongCell(true));
    try expectEqualDeep([_]f64{ 1.0 / 2.0, 1.0 / 2.0 }, S1.restrict());

    try expectEqualDeep([_]f64{ 0.0, 1.0, 0.0 }, S1.value());
    try expectEqualDeep([_]f64{ -1.0 / 2.0, 0.0, 1.0 / 2.0 }, S1.derivative());
    try expectEqualDeep([_]f64{ 1.0, -2.0, 1.0 }, S1.secondDerivative());
    try expectEqualDeep([_]f64{ 1.0 / 2.0, 1.0 / 2.0 }, S1.restrict());

    try expectEqualDeep([_]f64{ 0.0, -0.0, 1.0, 0.0, -0.0 }, S2.value());
    try expectEqualDeep([_]f64{ 1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0 }, S2.derivative());
    try expectEqualDeep([_]f64{ -1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0 }, S2.secondDerivative());
    try expectEqualDeep([_]f64{ -1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0 }, S2.restrict());
}
