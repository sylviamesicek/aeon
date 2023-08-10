const std = @import("std");

// Subdirectories
const basis = @import("basis/basis.zig");
const geometry = @import("geometry/geometry.zig");
const vtkio = @import("vtkio.zig");

// Aliases
const VtkCellType = vtkio.VtkCellType;
const VtkUnstructuredGrid = vtkio.VtkUnstructuredGrid;

// Main function
pub fn main() !void {
    // Setup Allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();

        if (deinit_status == .leak) {
            std.debug.print("Runtime data leak detected\n", .{});
        }
    }

    const allocator = gpa.allocator();

    // Setup vtk grid object
    var grid = try VtkUnstructuredGrid.init(allocator, .{ .cell_type = .quad, .points = &[_]f64{}, .vertices = &[_]i64{} });

    defer grid.deinit();

    const stdout = std.io.getStdOut().writer();
    try grid.write(stdout);
}

test {
    _ = geometry;
    _ = basis;
    _ = vtkio;
}
