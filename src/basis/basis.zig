//! This submodule provides interfaces for interacting with basis functions,
//! building numerical stencils, approximating derivatives, interpolating values,
//! and extrapolating from boundary conditions.

const lagrange = @import("lagrange.zig");
const space = @import("space.zig");

// ***************************
// Public exports ************
// ***************************

pub const CellSpace = space.CellSpace;
pub const StencilSpace = space.StencilSpace;

test {
    _ = lagrange;
    _ = space;
}
