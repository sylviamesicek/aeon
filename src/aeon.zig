// Subdirectories
pub const basis = @import("basis/basis.zig");
pub const dofs = @import("dofs/dofs.zig");
pub const geometry = @import("geometry/geometry.zig");
pub const index = @import("index.zig");
pub const lac = @import("lac/lac.zig");
pub const mesh = @import("mesh/mesh.zig");
pub const methods = @import("methods/methods.zig");
pub const system = @import("system.zig");
pub const vtkio = @import("vtkio.zig");

test {
    _ = basis;
    _ = geometry;
    _ = dofs;
    _ = mesh;
    _ = vtkio;
    _ = lac;
}
