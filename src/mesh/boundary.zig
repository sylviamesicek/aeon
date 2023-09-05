const std = @import("std");

const Face = @import("../geometry/geometry.zig").Face;

/// Represents a boundary condition as returned by a BoundaryOperator. Specifies a robin boundary condition along each
/// face as a function of position.
pub const BoundaryCondition = struct {
    value: f64,
    normal: f64,
    rhs: f64,

    const Self = @This();

    /// Constructs a diritchlet boundary condition.
    pub fn diritchlet(rhs: f64) Self {
        return .{
            .value = 1.0,
            .normal = 0.0,
            .rhs = rhs,
        };
    }

    /// Constructs a nuemann boundary condition.
    pub fn nuemann(rhs: f64) Self {
        return .{
            .value = 0.0,
            .normal = 1.0,
            .rhs = rhs,
        };
    }

    /// Constructs a robin boundary condition.
    pub fn robin(value: f64, normal: f64, rhs: f64) Self {
        return .{
            .value = value,
            .normal = normal,
            .rhs = rhs,
        };
    }
};

const is = std.meta.trait.is;
const hasFn = std.meta.trait.hasFn;
const TraitFn = std.meta.trait.TraitFn;

pub fn isBoundaryFunction(comptime N: usize) TraitFn {
    return is(fn ([N]f64, Face(N)) BoundaryCondition);
}

pub fn hasDiritchletDecl(comptime N: usize) bool {
    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (!hasFn("diritchlet")(T)) {
                return false;
            }

            return isBoundaryFunction(N)(@TypeOf(T.diritchlet));
        }
    };
    return Closure.trait;
}

pub fn hasNuemannDecl(comptime N: usize) bool {
    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (!hasFn("diritchlet")(T)) {
                return false;
            }

            return isBoundaryFunction(N)(@TypeOf(T.nuemann));
        }
    };
    return Closure.trait;
}

pub fn hasRobinDecl(comptime N: usize) bool {
    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (!hasFn("robin")(T)) {
                return false;
            }

            return isBoundaryFunction(N)(@TypeOf(T.robin));
        }
    };
    return Closure.trait;
}
