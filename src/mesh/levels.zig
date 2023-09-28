const std = @import("std");
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const Allocator = std.mem.Allocator;

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");

/// Represents a block in a mesh. A block exists in some patch, and is composed of a set of
/// tiles which are further subdivided into cells.
pub fn Block(comptime N: usize) type {
    return struct {
        /// Bounds of this block in level index space.
        bounds: IndexBox,
        /// Patch this block belongs to.
        patch: usize,
        /// Total number of cells in this block (including ghost cells)
        cell_total: usize = 0,
        /// Offset into the level wide cell array.
        cell_offset: usize = 0,

        const IndexBox = geometry.Box(N, usize);
    };
}

/// A set of tiles on a patch which are more sparesly laid out than blocks. Patches form
/// a higherarchical structure between levels on a mesh, and are essentially an optimisation
/// which allows for O(1) lookups of intersections and underlying blocks.
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

/// A single level of a mesh.
pub fn Level(comptime N: usize) type {
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
        const IndexSpace = geometry.IndexSpace(N);
        const Index = @import("../index.zig").Index(N);

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
        }

        pub fn clearRetainingCapacity(self: *Self) void {
            self.blocks.shrinkRetainingCapacity(0);
            self.patches.shrinkRetainingCapacity(0);
            self.children.shrinkRetainingCapacity(0);
            self.parents.shrinkRetainingCapacity(0);
        }

        /// Returns the total number of patches in this level.
        pub fn patchTotal(self: *const Self) usize {
            return self.patches.len;
        }

        pub fn blockTotal(self: *const Self) usize {
            return self.blocks.len;
        }

        /// Preallocates a number of total children.
        pub fn setTotalChildren(self: *Self, allocator: Allocator, total: usize) !void {
            try self.parents.resize(allocator, total);
        }

        /// All blocks should be given relative two the bounds of the patch.
        pub fn addPatch(self: *Self, allocator: Allocator, bounds: IndexBox, blocks: []const IndexBox, children: []const usize) !void {
            const patch_id: usize = self.patches.len;
            const block_offset: usize = self.blocks.len;
            const children_offset: usize = self.children.items.len;

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
                var bbounds: IndexBox = block;

                for (0..N) |i| {
                    bbounds.origin[i] += bounds.origin[i];
                }

                self.blocks.appendAssumeCapacity(Block(N){
                    .bounds = bbounds,
                    .patch = patch_id,
                });
            }

            try self.children.ensureUnusedCapacity(allocator, children.len);

            for (children) |child| {
                self.parents.items[child] = patch_id;
                self.children.appendAssumeCapacity(child);
            }
        }

        pub fn buildBase(self: *Self, allocator: Allocator, index_size: [N]usize, children: usize) !void {
            self.blocks.shrinkRetainingCapacity(0);
            self.patches.shrinkRetainingCapacity(0);
            try self.children.resize(allocator, children);
            try self.parents.resize(allocator, children);

            for (0..self.children.items.len) |i| {
                self.children.items[i] = i;
            }

            @memset(self.parents.items, 0);

            try self.blocks.append(allocator, .{
                .bounds = .{
                    .origin = [1]usize{0} ** N,
                    .size = index_size,
                },
                .patch = 0,
            });

            try self.patches.append(allocator, .{
                .bounds = .{
                    .origin = [1]usize{0} ** N,
                    .size = index_size,
                },
                .block_offset = 0,
                .block_total = 1,
                .children_offset = 0,
                .children_total = children,
            });
        }

        /// Gets a slice of children indices for each patch
        pub fn childrenSlice(self: *const Self, patch: usize) []const usize {
            const offset = self.patches.items(.children_offset)[patch];
            const total = self.patches.items(.children_total)[patch];
            return self.children.items[offset..(offset + total)];
        }

        /// Computes patch and block offsets and level totals for tiles and cells.
        pub fn computeOffsets(self: *Self, tile_width: usize) void {
            var tile_offset: usize = 0;

            for (self.patches.items(.bounds), self.patches.items(.tile_total), self.patches.items(.tile_offset)) |bounds, *total, *offset| {
                offset.* = tile_offset;
                total.* = IndexSpace.fromSize(bounds.size).total();
                tile_offset += total.*;
            }

            var cell_offset: usize = 0;

            for (self.blocks.items(.bounds), self.blocks.items(.cell_total), self.blocks.items(.cell_offset)) |bounds, *total, *offset| {
                offset.* = cell_offset;
                total.* = IndexSpace.fromSize(Index.scaled(bounds.size, tile_width)).total();
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
