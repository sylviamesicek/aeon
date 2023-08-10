const std = @import("std");

pub const IndexSpace = @import("index.zig").IndexSpace;
pub const Box = @import("box.zig").Box;
pub const Block = @import("block.zig").Block;

/// Represents one level of a mesh. This includes an array of
/// rectangular blocks that make up that particular level (which needs
/// not extend over the whole domain), and information for transforming
/// between index space and physical space.
pub fn Geometry(comptime N: usize) type {
    return struct {
        // Fields
        physical_bounds: Box(N, f64),
        index_space: IndexSpace(N),
        tile_width: usize,
        ghost_width: usize,
        blocks: BlockList,
        dof_offsets: OffsetList,

        // Aliases
        const Self = @This();
        const BlockList = std.ArrayList(Box(N, usize));
        const OffsetList = std.ArrayList(usize);

        /// Geometry
        pub const Config = struct {
            physical_bounds: Box(N, f64),
            tile_width: usize,
            ghost_width: usize,
        };

        pub fn init(allocator: std.mem.Allocator, config: Config) Self {
            return .{
                .physical_bounds = config.physical_bounds,
                .index_space = undefined,
                .tile_width = config.tile_width,
                .ghost_width = config.ghost_width,
                .blocks = BlockList.init(allocator),
                .dof_offsets = OffsetList.init(allocator),
            };
        }

        pub fn deinit(self: Self) void {
            self.blocks.deinit();
            self.dof_offsets.deinit();
        }

        pub fn regrid(self: *Self, tagged: Block(N, bool)) !void {
            self.cells = tagged.size;
        }
    };
}

test {
    _ = @import("block.zig");
    _ = @import("box.zig");
    _ = @import("index.zig");
}
