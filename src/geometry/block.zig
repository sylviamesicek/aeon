const std = @import("std");

/// An N-dimensional array with element T.
pub fn Block(comptime N: usize, comptime T: type) type {
    return struct {
        size: [N]usize,
        data: std.ArrayList(T),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, size: [N]usize) !Self {
            var length: usize = 1;

            for (0..N) |i| {
                length *= size[i];
            }

            var data = std.ArrayList(T).init(allocator);
            errdefer data.deinit();

            try data.resize(length);

            return .{
                .size = size,
                .data = data,
            };
        }

        pub fn deinit(self: Self) void {
            self.data.deinit();
        }
    };
}
