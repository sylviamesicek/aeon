const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;
const pow = std.math.pow;
const maxInt = std.math.maxInt;

pub const IndexSpace = @import("index.zig").IndexSpace;
pub const Box = @import("box.zig").Box;
pub const Tiles = @import("tiles.zig").Tiles;
pub const Partitions = @import("tiles.zig").Partitions;

/// Stores data for the geometry of a problem on one level. A domain is uniformly
/// divided into a number of tiles determined by `index_size`. A tile can be active
/// (which means the physical space covered by that tile is part of the problem space).
/// An active tile is further divded into cells as specified by `tile_width`,
/// where the value of a function is known at the center of each cell.
///
/// Cells are grouped (using point clustering) into a set of blocks, which allows
/// for better uniformity and cache efficiency when applying operators to functions.
/// Blocks are further divided between patches, to allow for O(1) block indexing (that
/// is, finding the block a tile belongs to) while still providing minimal storage
/// overhead.
pub fn Geometry(comptime N: usize) type {
    return struct {
        /// A general allocator used by `Geometry` internals.
        gpa: Allocator,
        /// The physical bounds of the problem domain.
        physical_bounds: RealBox,
        /// The size of the problem domain in index space.
        index_size: [N]usize,
        /// The number of cells on the edge of each tile.
        tile_width: usize,
        /// The number of additional ghost cells around each block.
        ghost_width: usize,
        /// A list of patches on this level.
        patches: MultiArrayList(Patch),
        /// A list of blocks on this level.
        blocks: MultiArrayList(Block),
        /// An array mapping tiles in patches to the block
        /// that contains that tile. `intMax(usize)` indicates
        /// that there is no block that contains that tile.
        tiles_to_blocks: ArrayListUnmanaged(usize),
        /// The total number of cells on this level, including
        /// ghost cells.
        total_cells: usize,

        // Aliases
        const Self = @This();
        const RealBox = Box(N, f64);
        const IndexBox = Box(N, usize);

        // Subtypes
        pub const Config = struct {
            /// The physical bounds of the problem domain.
            physical_bounds: RealBox,
            /// The size of the problem domain in index space.
            index_size: [N]usize,
            /// The number of cells on the edge of each tile.
            tile_width: usize,
            /// The number of additional ghost cells around each block.
            ghost_width: usize,

            pub fn check(config: Config) void {
                assert(config.tile_width >= 1);
                assert(config.tile_width >= config.ghost_width);
                for (0..N) |i| {
                    assert(config.index_size[i] > 0);
                    assert(config.physical_bounds.size[i] > 0.0);
                }
            }
        };

        pub const Patch = struct {
            /// Bounds of the patch in index space.
            bounds: IndexBox,
            /// Offset of the patch into tile array.
            offset: usize,
        };

        pub const Block = struct {
            /// Bounds of the block in index space.
            bounds: IndexBox,
            /// Patch this block is in.
            patch: usize,
            /// Offset into an array of cells.
            offset: usize,
        };

        // Methods

        pub fn init(gpa: Allocator, config: Config) Self {
            config.check();
            return .{
                .gpa = gpa,
                .physical_bounds = config.physical_bounds,
                .index_size = config.index_size,
                .tile_width = config.tile_width,
                .ghost_width = config.ghost_width,
                .patches = MultiArrayList(IndexBox){},
                .blocks = MultiArrayList(Block){},
                .tiles_to_blocks = ArrayListUnmanaged(usize){},
                .total_cells = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.patches.deinit(self.gpa);
            self.blocks.deinit(self.gpa);
            self.tiles_to_blocks.deinit(self.gpa);
        }

        pub fn clearRetainingCapacity(self: *Self) void {
            self.patches.shrinkRetainingCapacity(0);
            self.blocks.shrinkRetainingCapacity(0);
            self.tiles_to_blocks.shrinkRetainingCapacity(0);
        }

        pub fn addPatch(self: *Self, patch: IndexBox, blocks: []IndexBox) !void {
            const space: IndexSpace(N) = .{ .size = patch.size };
            const total: usize = space.total();

            const tile_offset = self.tiles_to_blocks.items.len;
            const patch_index: usize = self.patches.len;

            try self.patches.append(self.gpa, .{
                .bounds = patch,
                .offset = tile_offset,
            });

            try self.tiles_to_blocks.appendNTimes(self.gpa, maxInt(usize), total);

            var cell_offset: usize = self.total_cells;

            for (blocks, 0..) |block, i| {
                try self.blocks.append(self.gpa, .{
                    .bounds = block,
                    .parent = patch_index,
                    .offset = cell_offset,
                });

                space.fillSubspace(block, usize, self.tiles_to_blocks.items, i);

                const block_space = block.space();
                cell_offset += block_space.totalWithGhost(self.tile_width, self.ghost_width);
            }

            self.total_cells = cell_offset;
        }

        /// Returns the number of tiles in a given patch.
        pub fn tilesInPatch(self: Self, patch: usize) usize {
            const bounds = self.patches.items(.bounds)[patch];
            const space: IndexSpace(N) = .{ .size = bounds.widths };
            return space.total();
        }

        /// Gives the total number of tiles in all patches.
        pub fn tileTotal(self: Self) usize {
            return self.tiles_to_blocks.items.len;
        }

        pub fn tileOffset(self: Self, patch: usize) usize {
            return self.patches.items(.offset)[patch];
        }

        pub fn tileToBlock(self: Self, patch: usize, tile: [N]usize) usize {
            const bounds = self.patches.items(.bounds)[patch];
            const offset = self.patches.items(.offset)[patch];
            const space: IndexSpace(N) = .{ .size = bounds.widths };
            const linear = space.linearFromCartesian(tile);
            return self.tiles_to_blocks.items[offset + linear];
        }

        pub fn tilesToBlocks(self: Self, patch: usize) []const usize {
            const bounds = self.patches.items(.bounds)[patch];
            const offset = self.patches.items(.offset)[patch];
            const space: IndexSpace(N) = .{ .size = bounds.widths };
            return self.tiles_to_blocks.items[offset..(offset + space.total())];
        }

        pub fn cellTotal(self: Self) usize {
            return self.total_cells;
        }

        pub fn cellOffset(self: Self, block: usize) usize {
            return self.blocks.items(.offset)[block];
        }
    };
}

/// A geometry where all tiles are active.
pub fn UniformGeometry(comptime N: usize) type {
    return struct {
        /// The physical bounds of the problem domain.
        physical_bounds: RealBox,
        /// The size of the problem domain in index space.
        index_size: [N]usize,
        /// The number of cells on the edge of each tile.
        tile_width: usize,
        /// The number of additional ghost cells around the domain.
        ghost_width: usize,

        // Aliases
        const Self = @This();
        const RealBox = Box(N, f64);
        const IndexBox = Box(N, usize);

        /// Returns the total number of tiles on this geometry
        pub fn tileTotal(self: Self) usize {
            const space: IndexSpace(N) = .{ .size = self.index_size };
            return space.total();
        }

        /// Returns the total number of cells (including ghost cells) on this geometry.
        pub fn cellTotal(self: Self) usize {
            var total: usize = 1;

            for (0..N) |i| {
                total *= self.index_size[i] * self.tile_width[i] + 2 * self.ghost_width;
            }

            return total;
        }
    };
}

test {
    _ = @import("box.zig");
    _ = @import("index.zig");
    _ = @import("tiles.zig");
}
