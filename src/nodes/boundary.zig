const std = @import("std");

const geometry = @import("../geometry/geometry.zig");

/// Describes a the type of boundary condition
/// to be used to fill ghost cells on a given axis.
pub const BoundaryKind = enum {
    /// The function is symmetric across the boundary.
    even,
    /// The function is anti-symmetric across the boundary.
    odd,
    /// The function satisfies some set of robin-style boundary
    /// conditions at the boundary.
    robin,
};

/// A struct describing a robin boundary.
pub const Robin = struct {
    value: f64,
    flux: f64,
    rhs: f64,

    /// Constructs a diritchlet boundary condition.
    pub fn diritchlet(rhs: f64) Robin {
        return .{
            .value = 1.0,
            .flux = 0.0,
            .rhs = rhs,
        };
    }

    /// Constructs a nuemann boundary condition.
    pub fn nuemann(rhs: f64) Robin {
        return .{
            .value = 0.0,
            .flux = 1.0,
            .rhs = rhs,
        };
    }
};

/// A trait for defining boundaries.
pub fn isBoundary(comptime N: usize) fn (comptime T: type) bool {
    const FaceIndex = geometry.FaceIndex(N);
    const hasFn = std.meta.trait.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (!(hasFn("kind")(T) and @TypeOf(T.kind) == fn (FaceIndex) BoundaryKind)) {
                return false;
            }

            if (!(hasFn("robin")(T) and @TypeOf(T.robin) == fn (T, [N]f64, FaceIndex) Robin)) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}
