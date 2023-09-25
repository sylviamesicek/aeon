const std = @import("std");

// Submodules
const bicgstab = @import("bicgstab.zig");
const bicgstabl = @import("bicgstabl.zig");

// ************************
// Linear Solvers *********
// ************************

pub const BiCGStabSolver = bicgstab.BiCGStabSolver;
pub const BiCGStablSolver = bicgstabl.BiCGStablSolver;

// ************************
// Core traits and types **
// ************************

/// A trait which checks if a type is an linear map. Such a type follows the following set of declarations.
/// ```
/// const LinearMap = struct {
///     pub fn apply(self: LinearMap, output: []f64, input: []const f64) void {
///         // ...
///     }
/// };
/// ```
pub fn isLinearMap(comptime T: type) bool {
    const hasFn = std.meta.trait.hasFn;

    if (!(hasFn("apply")(T) and @TypeOf(T.apply) == fn (*const T, []f64, []const f64) void)) {
        return false;
    }

    return true;
}

/// A trait which checks if a type is a linear solver. Such a type follows the following set of declarations.
///
/// ```
/// const LinearSolver = struct {
///     pub fn solve(self: LinearSolver, operator: anytype, x: []f64, b: []const f64) void {
///         // ...
///     }
/// }
/// ```
pub fn isLinearSolver(comptime T: type) bool {
    const hasFn = std.meta.trait.hasFn;

    if (!(hasFn("solve")(T) and @TypeOf(T.solve) == fn (*const T, anytype, []f64, []const f64) void)) {
        return false;
    }

    return true;
}

/// Represents an identity map which copies the input to the output.
pub const IdentityMap = struct {
    pub fn apply(_: *const IdentityMap, output: []f64, input: []const f64) void {
        // Simply copy input to output
        @memcpy(output, input);
    }
};

// *********************
// Propogate testing ***
// *********************

test {}
