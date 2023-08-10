const std = @import("std");

pub const IndexSpace = @import("index.zig").IndexSpace;
pub const Box = @import("box.zig").Box;
pub const Block = @import("block.zig").Block;

/// Represents one level of a mesh. This includes an array of
/// rectangular blocks that make up that particular level (which needs
/// not extend over the whole domain), and information for transforming
/// between index space and physical space.
///
/// TODO: Rework init/deinit/partition API to be more zig-ish.
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
        pub fn init(allocator: std.mem.Allocator, physical_bounds: Box(N, f64)) Self {
            return .{
                .physical_bounds = physical_bounds,
                .index_space = .{ .size = [1]usize{0} ** N },
                .tile_width = 1,
                .ghost_width = 1,
                .blocks = BlockList.init(allocator),
                .dof_offsets = OffsetList.init(allocator),
            };
        }

        pub fn deinit(self: Self) void {
            self.blocks.deinit();
            self.dof_offsets.deinit();
        }

        pub const PartitionConfig = struct {
            tile_width: usize,
            ghost_width: usize,
            max_size: usize,
            efficiency: f64 = 0.7,
        };

        pub fn partition(self: *Self, tagged: Block(N, bool), config: PartitionConfig) !void {
            _ = config;
            self.cells = tagged.size;
        }

        fn compute_efficiency(tagged: Block(N, bool), part: Box(N, usize)) f64 {
            _ = part;
            _ = tagged;
        }
    };
}

test {
    _ = @import("block.zig");
    _ = @import("box.zig");
    _ = @import("index.zig");
}
