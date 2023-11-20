//! This submodule provides interfaces for interacting with basis functions,
//! building numerical stencils, approximating derivatives, interpolating values,
//! and extrapolating from boundary conditions.

const lagrange = @import("lagrange.zig");
const nodes = @import("nodes.zig");
const stencils = @import("stencils.zig");

// ***************************
// Public exports ************
// ***************************

pub const Lagrange = lagrange.Lagrange;
pub const Stencils = stencils.Stencils;

pub const NodeSpace = nodes.NodeSpace;
pub const StencilSpace = stencils.StencilSpace;

test {
    _ = lagrange;
    _ = nodes;
    _ = stencils;
}
