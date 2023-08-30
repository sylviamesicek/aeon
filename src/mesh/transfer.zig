const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayListUnmanaged = std.ArrayListUnmanaged;

pub const Transfer = struct {
    gpa: Allocator,
    base: Base,

    pub const Base = struct {
        tile_total: usize,
        cell_total: usize,
    };

    pub const Level = struct {
        tile_total: usize,
        cell_total: usize,

        tile_offset: usize,
        cell_offset: usize,
    };
};
