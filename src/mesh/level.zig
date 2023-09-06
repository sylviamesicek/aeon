const std = @import("std");
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const Allocator = std.mem.Allocator;

const geometry = @import("../geometry/geometry.zig");
const Box = geometry.Box;

pub fn Block(comptime N: usize) type {
    return struct {
        bounds: Box(N, usize),
        patch: usize,
        cell_total: usize = 0,
        cell_offset: usize = 0,
    };
}

pub fn Patch(comptime N: usize) type {
    return struct {
        /// Bounds of this patch in level index space.
        bounds: Box(N, usize),
        /// Offset into blocks in this patch.
        block_offset: usize,
        /// Total number of blocks in this patch.
        block_total: usize,
        /// Offset into children array
        children_offset: usize,
        /// Number of child patches
        children_total: usize,
        /// The total number of tiles on this patch.
        tile_total: usize = 0,
        /// The offset into a level-wide array of tiles.
        tile_offset: usize = 0,
    };
}

pub fn Level(comptime N: usize, comptime O: usize) type {
    return struct {
        index_size: [N]usize,

        /// The blocks belonging to this level.
        blocks: MultiArrayList(Block(N)),
        /// The patches belonging to this level.
        patches: MultiArrayList(Patch(N)),
        /// Index buffer for children of patches
        children: ArrayListUnmanaged(usize),
        /// The parent of each child.
        parents: ArrayListUnmanaged(usize),
        // Total number of tiles in this level
        tile_total: usize,
        // Total number of cells in this level.
        cell_total: usize,
        // Offset in tile array for this level
        tile_offset: usize,
        // Offset into cell array for this level
        cell_offset: usize,

        /// Allocates a new level with no data.
        fn init(index_size: [N]usize) Level {
            return .{
                .index_size = index_size,

                .blocks = .{},
                .patches = .{},
                .children = .{},
                .parents = .{},
                .tile_offset = 0,
                .cell_offset = 0,
                .tile_total = 0,
                .cell_total = 0,
            };
        }

        /// Frees a level
        fn deinit(self: *Level, allocator: Allocator) void {
            self.blocks.deinit(allocator);
            self.patches.deinit(allocator);
            self.children.deinit(allocator);
            self.parents.deinit(allocator);
            self.transfer_blocks.deinit(allocator);
            self.transfer_patches.deinit(allocator);
        }

        /// Gets a slice of children indices for each patch
        fn childrenSlice(self: *Level, patch: usize) []const usize {
            const offset = self.patches.items(.children_offset)[patch];
            const total = self.patches.items(.children_total)[patch];
            return self.children.items[offset..(offset + total)];
        }

        fn tileSlice(self: *Level, patch: usize, level_slice: anytype) @TypeOf(level_slice) {
            if (@TypeOf(level_slice) != []f64 and @TypeOf(level_slice) != []const f64) {
                @compileError("tileSlice only defined for []f64 and []const f64");
            }

            const offset = self.patches.items(.tile_offset)[patch];
            const total = self.patches.items(.tile_total)[patch];
            return level_slice[offset..(offset + total)];
        }

        /// Computes patch and block offsets and level totals for tiles and cells.
        fn computeOffsets(self: *Level, tile_width: usize) void {
            var tile_offset: usize = 0;

            for (self.patches.items(.bounds), self.patches.items(.tile_total), self.patches.items(.tile_offset)) |bounds, *total, *offset| {
                offset.* = tile_offset;
                total.* = bounds.space().total();
                tile_offset += total.*;
            }

            var cell_offset: usize = 0;

            for (self.blocks.items(.bounds), self.blocks.items(.cell_total), self.blocks.items(.cell_offset)) |bounds, *total, *offset| {
                offset.* = cell_offset;
                total.* = bounds.space().scale(tile_width).extendUniform(2 * O).total();
                cell_offset += total.*;
            }

            self.tile_total = tile_offset;
            self.cell_total = cell_offset;

            var transfer_patch_offset: usize = 0;

            for (self.transfer_patches.items(.patch_total), self.transfer_patches.items(.patch_offset)) |total, *offset| {
                offset.* = transfer_patch_offset;
                transfer_patch_offset += total;
            }

            var transfer_block_offset: usize = 0;

            for (self.transfer_patches.items(.block_total), self.transfer_patches.items(.block_offset)) |total, *offset| {
                offset.* = transfer_block_offset;
                transfer_block_offset += total;
            }

            var transfer_tile_offset: usize = 0;

            for (self.transfer_patches.items(.bounds), self.transfer_patches.items(.tile_total), self.transfer_patches.items(.tile_offset)) |bounds, *total, *offset| {
                offset.* = transfer_tile_offset;
                total.* = bounds.space().total();
                transfer_tile_offset += total.*;
            }

            var transfer_cell_offset: usize = 0;

            for (self.transfer_blocks.items(.bounds), self.transfer_blocks.items(.cell_total), self.transfer_blocks.items(.cell_offset)) |bounds, *total, *offset| {
                offset.* = transfer_cell_offset;
                total.* = bounds.space().scale(tile_width).extendUniform(2 * O).total();
                transfer_cell_offset += total.*;
            }

            self.tile_total = transfer_tile_offset;
            self.cell_total = transfer_cell_offset;
        }

        /// Refines every patch and block on this level.
        fn refine(self: *Level) void {
            for (self.patches.items(.bounds)) |*bounds| {
                bounds.refine();
            }

            for (self.blocks.items(.bounds)) |*bounds| {
                bounds.refine();
            }

            for (self.transfer_blocks.items(.bounds)) |*bounds| {
                bounds.refine();
            }

            for (self.transfer_patches.items(.bounds)) |*bounds| {
                bounds.refine();
            }
        }
    };
}
