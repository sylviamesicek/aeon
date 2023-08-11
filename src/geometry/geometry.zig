const std = @import("std");

pub const IndexSpace = @import("index.zig").IndexSpace;
pub const Box = @import("box.zig").Box;
pub const Tiles = @import("tiles.zig").Tiles;

/// Represents one level of a mesh. This includes an array of
/// rectangular blocks that make up that particular level (which needs
/// not extend over the whole domain), and information for transforming
/// between index space and physical space.
///
/// TODO: Rework init/deinit/partition API to be more zig-ish.
pub fn Geometry(comptime N: usize) type {
    return struct {
        // Fields
        allocator: std.mem.Allocator,
        physical_bounds: Box(N, f64),
        index_space: IndexSpace(N),
        tile_width: usize,
        ghost_width: usize,
        blocks: BlockList,
        dof_offsets: OffsetList,

        // Aliases
        const Self = @This();
        const BlockList = std.ArrayListUnmanaged(Box(N, usize));
        const OffsetList = std.ArrayListUnmanaged(usize);

        /// Geometry
        pub fn init(allocator: std.mem.Allocator, physical_bounds: Box(N, f64)) Self {
            return .{
                .allocator = allocator,
                .physical_bounds = physical_bounds,
                .index_space = .{ .size = [1]usize{0} ** N },
                .tile_width = 1,
                .ghost_width = 1,
                .blocks = BlockList{},
                .dof_offsets = OffsetList{},
            };
        }

        pub fn deinit(self: *Self) void {
            self.blocks.deinit(self.allocator);
            self.dof_offsets.deinit(self.allocator);
        }
    };
}

test {
    _ = @import("box.zig");
    _ = @import("index.zig");
    _ = @import("tiles.zig");
}

test "geometry" {
    var geometry = Geometry(2).init(std.testing.allocator, .{
        .origin = [2]f64{ 0.0, 0.0 },
        .widths = [2]f64{ 1.0, 1.0 },
    });
    defer geometry.deinit();
}
