const std = @import("std");
const vtkio = @import("vtkio.zig");
const VtkCellType = vtkio.VtkCellType;
const VtkGrid = vtkio.VtkGrid;

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
    var grid = VtkGrid.init(allocator);
    defer grid.deinit();

    const stdout = std.io.getStdOut().writer();
    try grid.write_unstructured(.quad, &[_]f64{}, &[_]i64{}, stdout);
}
