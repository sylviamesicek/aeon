const std = @import("std");

pub fn Face(comptime N: usize) type {
    if (N > 16) {
        @compileError("Face only supports N <= 16");
    }

    return struct {
        side: bool,
        axis: usize,

        const Self = @This();

        pub const Count: usize = 2 * N;

        pub fn index(self: Self) usize {
            return if (self.side) .{ .index = self.axis + N } else .{ .index = self.axis };
        }
    };
}
