// Submodules

pub const basis = @import("basis/basis.zig");
pub const bsamr = @import("bsamr/bsamr.zig");
pub const dofs = @import("dofs/dofs.zig");
pub const geometry = @import("geometry/geometry.zig");
pub const lac = @import("lac/lac.zig");
pub const mesh = @import("mesh/mesh.zig");
pub const methods = @import("methods/methods.zig");
pub const nodes = @import("nodes/nodes.zig");

// Global exports.

const io = @import("io/io.zig");
pub const DataOut = io.DataOut;

const system = @import("system.zig");
pub const isSystem = system.isSystem;
pub const SystemSlice = system.SystemSlice;
pub const SystemSliceConst = system.SystemSliceConst;
pub const SystemValue = system.SystemValue;
pub const EmptySystem = system.EmptySystem;

test {
    _ = basis;
    _ = dofs;
    _ = geometry;
    _ = io;
    _ = mesh;
    _ = nodes;
    _ = lac;
    _ = system;
}
