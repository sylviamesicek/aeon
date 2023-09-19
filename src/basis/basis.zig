//! This submodule provides interfaces for interacting with basis functions,
//! building numerical stencils, approximating derivatives, interpolating values,
//! and extrapolating from boundary conditions.

const boundary = @import("boundary.zig");
const lagrange = @import("lagrange.zig");
const space = @import("space.zig");

// ***************************
// Public exports ************
// ***************************

pub const StencilSpace = space.StencilSpace;
pub const InterpolationSpace = space.InterpolationSpace;
pub const BoudnaryCondition = boundary.BoundaryCondition;
pub const isBoundaryOperator = boundary.isBoundaryOperator;

test {
    _ = boundary;
    _ = lagrange;
    _ = space;
}
