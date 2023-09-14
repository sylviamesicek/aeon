const std = @import("std");
const Allocator = std.mem.Allocator;

// Subdirectories
const array = @import("array.zig");
const basis = @import("basis/basis.zig");
const geometry = @import("geometry/geometry.zig");
const mesh = @import("mesh/mesh.zig");
const solver = @import("solver/solver.zig");
const vtkio = @import("vtkio.zig");

/// Main function, using a universal base allocator.
fn mainChecked(allocator: Allocator) !void {
    // Aliases
    const VtkUnstructuredGrid = vtkio.VtkUnstructuredGrid;

    // Setup vtk grid object
    var grid = try VtkUnstructuredGrid.init(allocator, .{ .cell_type = .quad, .points = &[_]f64{}, .vertices = &[_]i64{} });
    defer grid.deinit();

    const stdout = std.io.getStdOut().writer();
    try grid.write(stdout);
}

/// Actual main function (with allocator and leak detection boilerplate)
pub fn main() !void {
    // Setup Allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();

        if (deinit_status == .leak) {
            std.debug.print("Runtime data leak detected\n", .{});
        }
    }

    // Run main
    try mainChecked(gpa.allocator());
}

test {
    _ = array;
    _ = basis;
    _ = geometry;
    _ = mesh;
    _ = solver;
    _ = vtkio;
}
