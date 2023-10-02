//! This submodule provides interfaces for interacting with basis functions,
//! building numerical stencils, approximating derivatives, interpolating values,
//! and extrapolating from boundary conditions.

const lagrange = @import("lagrange.zig");
const space = @import("space.zig");

// ***************************
// Public exports ************
// ***************************

pub const CellSpaceWithExtent = space.CellSpaceWithExtent;

pub fn CellSpace(comptime N: usize, comptime O: usize) type {
    return CellSpaceWithExtent(N, 2 * O, O);
}

pub const StencilSpaceWithExtent = space.StencilSpaceWithExtent;

pub fn StencilSpace(comptime N: usize, comptime O: usize) type {
    return StencilSpaceWithExtent(N, 2 * O, O);
}

test {
    _ = lagrange;
    _ = space;
}
