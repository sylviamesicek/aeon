const std = @import("std");

/// Identifies a face by axis and side (false is left, true is right).
pub fn Face(comptime N: usize) type {
    return struct {
        side: bool,
        axis: usize,

        const Self = @This();

        pub const Count: usize = 2 * N;

        /// Returns an index into an array of size `Count`.
        pub fn index(self: Self) usize {
            return if (self.side) .{ .index = self.axis + N } else .{ .index = self.axis };
        }
    };
}
