//! This modules provides the primary interface for block structured finite difference
//! meshs, as well as adaptively refining those meshes, transforming cell representations
//! to nodes, and solving differential equations on these domains.

const dofs = @import("dofs.zig");
const mesh = @import("mesh.zig");
const regrid = @import("regrid.zig");

pub const DofManager = dofs.DofManager;

pub const Block = mesh.Block;
pub const Level = mesh.Level;
pub const Mesh = mesh.Mesh;
pub const Patch = mesh.Patch;

pub const RegridManager = regrid.RegridManager;

test {
    _ = dofs;
    _ = mesh;
    _ = regrid;
}
