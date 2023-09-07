const std = @import("std");
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const Allocator = std.mem.Allocator;

const geometry = @import("../geometry/geometry.zig");
const Box = geometry.Box;

pub fn Block(comptime N: usize) type {
    return struct {
        /// Bounds of this block in level index space.
        bounds: IndexBox,
        /// Patch this block belongs to
        patch: usize,
        /// Total number of cells in this block (including ghost cells)
        cell_total: usize = 0,
        /// Offset into the level wide cell array.
        cell_offset: usize = 0,

        const IndexBox = geometry.Box(N, usize);
    };
}

pub fn Patch(comptime N: usize) type {
    return struct {
        /// Bounds of this patch in level index space.
        bounds: IndexBox,
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

        const IndexBox = geometry.Box(N, usize);
    };
}

pub fn Level(comptime N: usize, comptime O: usize) type {
    return struct {
        /// Number of indices on each side in the level index space.
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
        // Offset in mesh tile array for this level
        tile_offset: usize,
        // Offset into mesh cell array for this level
        cell_offset: usize,

        const Self = @This();
        const IndexBox = geometry.Box(N, usize);

        /// Allocates a new level with no data.
        pub fn init(index_size: [N]usize) Self {
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
        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.blocks.deinit(allocator);
            self.patches.deinit(allocator);
            self.children.deinit(allocator);
            self.parents.deinit(allocator);
            self.transfer_blocks.deinit(allocator);
            self.transfer_patches.deinit(allocator);
        }

        pub fn clearRetainingCapacity(self: *Self) void {
            self.blocks.shrinkRetainingCapacity(0);
            self.patches.shrinkRetainingCapacity(0);
            self.children.shrinkRetainingCapacity(0);
            self.parents.shrinkRetainingCapacity(0);
        }

        pub fn setTotalChildren(self: *Self, allocator: Allocator, total: usize) !void {
            try self.parents.resize(allocator, total);
        }

        pub fn addPatch(self: *Self, allocator: Allocator, bounds: IndexBox, blocks: []const IndexBox, children: []const usize) !void {
            const patch_id: usize = self.patches.len;
            const block_offset: usize = self.blocks.len;
            const children_offset: usize = self.children.len;

            const patch: Patch(N) = .{
                .bounds = bounds,
                .block_offset = block_offset,
                .block_total = blocks.len,
                .children_offset = children_offset,
                .children_total = children.len,
            };

            try self.patches.append(allocator, patch);

            try self.blocks.ensureUnusedCapacity(allocator, blocks.len);

            for (blocks) |block| {
                self.blocks.appendAssumeCapacity(Block(N){
                    .bounds = block,
                    .patch = patch_id,
                });
            }

            try self.children.ensureUnusedCapacity(allocator, children.len);

            for (children) |child| {
                self.parents.items[child] = patch_id;
                self.children.appendAssumeCapacity(child);
            }
        }

        /// Gets a slice of children indices for each patch
        pub fn childrenSlice(self: *const Self, patch: usize) []const usize {
            const offset = self.patches.items(.children_offset)[patch];
            const total = self.patches.items(.children_total)[patch];
            return self.children.items[offset..(offset + total)];
        }

        pub fn levelTileSlice(self: *const Self, mesh_slice: anytype) @TypeOf(mesh_slice) {
            return mesh_slice[self.tile_offset..(self.tile_offset + self.tile_total)];
        }

        pub fn levelCellSlice(self: *const Self, mesh_slice: anytype) @TypeOf(mesh_slice) {
            return mesh_slice[self.cell_offset..(self.cell_offset + self.cell_total)];
        }

        pub fn patchTileSlice(self: *const Self, patch: usize, level_slice: anytype) @TypeOf(level_slice) {
            const offset = self.patches.items(.tile_offset)[patch];
            const total = self.patches.items(.tile_total)[patch];
            return level_slice[offset..(offset + total)];
        }

        pub fn blockCellSlice(self: *const Self, block: usize, level_slice: anytype) @TypeOf(level_slice) {
            const total = self.blocks.items(.cell_total)[block];
            const offset = self.blocks.items(.cell_offset)[block];
            return level_slice[offset..(offset + total)];
        }

        /// Computes patch and block offsets and level totals for tiles and cells.
        pub fn computeOffsets(self: *Self, tile_width: usize) void {
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

            // var transfer_patch_offset: usize = 0;

            // for (self.transfer_patches.items(.patch_total), self.transfer_patches.items(.patch_offset)) |total, *offset| {
            //     offset.* = transfer_patch_offset;
            //     transfer_patch_offset += total;
            // }

            // var transfer_block_offset: usize = 0;

            // for (self.transfer_patches.items(.block_total), self.transfer_patches.items(.block_offset)) |total, *offset| {
            //     offset.* = transfer_block_offset;
            //     transfer_block_offset += total;
            // }

            // var transfer_tile_offset: usize = 0;

            // for (self.transfer_patches.items(.bounds), self.transfer_patches.items(.tile_total), self.transfer_patches.items(.tile_offset)) |bounds, *total, *offset| {
            //     offset.* = transfer_tile_offset;
            //     total.* = bounds.space().total();
            //     transfer_tile_offset += total.*;
            // }

            // var transfer_cell_offset: usize = 0;

            // for (self.transfer_blocks.items(.bounds), self.transfer_blocks.items(.cell_total), self.transfer_blocks.items(.cell_offset)) |bounds, *total, *offset| {
            //     offset.* = transfer_cell_offset;
            //     total.* = bounds.space().scale(tile_width).extendUniform(2 * O).total();
            //     transfer_cell_offset += total.*;
            // }

            // self.tile_total = transfer_tile_offset;
            // self.cell_total = transfer_cell_offset;
        }

        /// Refines every patch and block on this level.
        pub fn refine(self: *Self) void {
            for (self.patches.items(.bounds)) |*bounds| {
                bounds.refine();
            }

            for (self.blocks.items(.bounds)) |*bounds| {
                bounds.refine();
            }

            // for (self.transfer_blocks.items(.bounds)) |*bounds| {
            //     bounds.refine();
            // }

            // for (self.transfer_patches.items(.bounds)) |*bounds| {
            //     bounds.refine();
            // }
        }
    };
}
