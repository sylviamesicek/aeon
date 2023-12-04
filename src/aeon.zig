// Submodules

pub const basis = @import("basis/basis.zig");
pub const bsamr = @import("bsamr/bsamr.zig");
pub const geometry = @import("geometry/geometry.zig");
pub const lac = @import("lac/lac.zig");
pub const mesh = @import("mesh/mesh.zig");
pub const methods = @import("methods/methods.zig");
pub const nodes = @import("nodes/nodes.zig");

// Global exports.

// Propogate testing

test {
    _ = basis;
    _ = bsamr;
    _ = geometry;
    _ = mesh;
    _ = nodes;
    _ = lac;
}
