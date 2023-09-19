const std = @import("std");

// Submodules
const bicgstab = @import("bicgstab.zig");

// ************************
// Public Exports *********
// ************************

/// A trait which checks if a type is an operator. Such a type follows the following set of declarations.
/// ```
/// const Operator = struct {
///     pub fn apply(self: Operator, output: []f64, input: []const f64) f64 {
///         // ...
///     }
/// };
/// ```
pub fn isOperator(comptime T: type) bool {
    const hasFn = std.meta.trait.hasFn;

    if (!(hasFn("apply")(T) and @TypeOf(T.apply) == fn (T, []f64, []const f64) void)) {
        return false;
    }

    return true;
}

pub const BiCGStabSolver = bicgstab.BiCGStabSolver;

test {
    _ = bicgstab;
}
