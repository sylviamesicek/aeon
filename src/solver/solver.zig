const std = @import("std");
const bicgstabl = @import("bicgstabl.zig");

/// Represents an operator which is simply a (linear) function taking a input vector and writing a transformation of this
/// vector into an output vector.
pub fn MatrixFreeOperator(comptime Ctx: type) type {
    return fn ([]f64, Ctx, []const f64) void;
}

pub const BiCGStablSolver = bicgstabl.BiCGStablSolver;

test {
    _ = bicgstabl;
}
