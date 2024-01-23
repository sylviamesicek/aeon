//! This submodule provides interfaces for interacting with basis functions,
//! building numerical stencils, approximating derivatives, interpolating values,
//! and extrapolating from boundary conditions.

const lagrange = @import("lagrange.zig");
const stencils = @import("stencils.zig");

// ***************************
// Public exports ************
// ***************************

pub const Lagrange = lagrange.Lagrange;
pub const Stencils = stencils.Stencils;

test {
    _ = lagrange;
    _ = stencils;
}
