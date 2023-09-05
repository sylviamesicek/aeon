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

/// A trait to check if a type is a boundary function (ie a function which takes in a position and a face
/// and returns a boundary condition).
pub fn isBoundaryFunction(comptime N: usize) TraitFn {
    return is(fn ([N]f64, Face(N)) BoundaryCondition);
}

/// Checks whether a type has a function named `condition` that satisfied the `isBoundaryFunction(N)` trait.
pub fn hasConditionDecl(comptime N: usize) TraitFn {
    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (!hasFn("condition")(T)) {
                return false;
            }

            return isBoundaryFunction(N)(@TypeOf(T.condition));
        }
    };
    return Closure.trait;
}
