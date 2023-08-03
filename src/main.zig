const std = @import("std");
const vtkio = @import("vtkio.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();

        if (deinit_status == .leak) {
            std.debug.print("Runtime data leak detected\n", .{});
        }
    }

    const allocator = gpa.allocator();

    var vec = std.ArrayList(f64).init(allocator);
    defer vec.deinit();

    try vec.append(10.0);
    try vec.append(20.0);
    try vec.append(30.0);

    std.debug.print("{}", .{vec});
}
