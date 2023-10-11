//! Handles elliptic equation solving and hyperbolic intergration.

const std = @import("std");
const system = @import("../system.zig");

const multigrid = @import("multigrid.zig");
const rk4 = @import("rk4.zig");

pub const MultigridMethod = multigrid.MultigridMethod;
pub const RungeKutta4Integrator = rk4.RungeKutta4Integrator;

pub fn isSystemDerivative(comptime T: type) bool {
    const hasFn = std.meta.trait.hasFn;

    if (comptime !(@hasDecl(T, "System") and @TypeOf(T.System) == type and system.isSystem(T.System))) {
        return false;
    }

    if (comptime !(hasFn("derivative")(T) and @TypeOf(T.derivative) == fn (T, system.SystemSlice(T.System), system.SystemSliceConst(T.System), f64) void)) {
        return false;
    }

    return true;
}
