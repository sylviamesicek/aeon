// Submodules

pub const basis = @import("basis/basis.zig");
pub const dofs = @import("dofs/dofs.zig");
pub const geometry = @import("geometry/geometry.zig");
pub const lac = @import("lac/lac.zig");
pub const mesh = @import("mesh/mesh.zig");
pub const methods = @import("methods/methods.zig");
pub const vtkio = @import("vtkio.zig");

// Global exports.

const system = @import("system.zig");
const index = @import("index.zig");

pub const Index = index.Index;

pub const isSystem = system.isSystem;
pub const SystemSlice = system.SystemSlice;
pub const SystemSliceConst = system.SystemSliceConst;
pub const SystemValue = system.SystemValue;

test {
    _ = basis;
    _ = dofs;
    _ = geometry;
    _ = index;
    _ = mesh;
    _ = vtkio;
    _ = lac;
    _ = system;
}
