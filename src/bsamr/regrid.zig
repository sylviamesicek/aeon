const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const ArenaAllocator = std.heap.ArenaAllocator;
const assert = std.debug.assert;
const exp2 = std.math.exp2;
const maxInt = std.math.maxInt;

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const mesh = @import("../mesh/mesh.zig");

const mesh_ = @import("mesh.zig");

pub fn RegridManager(comptime N: usize) type {
    return struct {
        max_levels: usize,
        patch_efficiency: f64,
        patch_max_tiles: usize = std.math.maxInt(usize),
        block_efficiency: f64,
        block_max_tiles: usize = std.math.maxInt(usize),

        const ClusterSpace = geometry.ClusterSpace(N);
        const IndexBox = geometry.IndexBox(N);
        const IndexSpace = geometry.IndexSpace(N);

        const IndexMixin = geometry.IndexMixin(N);
        const splat = IndexMixin.splat;

        const Block = mesh_.Block(N);
        const Level = mesh_.Level(N);
        const Mesh = mesh_.Mesh(N);
        const Patch = mesh_.Patch(N);

        const TileMap = mesh.TileMap;

        pub fn regrid(
            self: @This(),
            allocator: Allocator,
            tags: []const bool,
            grid: *Mesh,
            tile_map: TileMap,
        ) !void {

            // The total levels must be >= 1, but may not be greater than self.levels.len + 1.
            const total_levels = @max(1, @min(self.max_levels, grid.levels.len + 1));

            // **********************************
            // Allocate Per level data

            var data: MeshHeirarchy = try MeshHeirarchy.init(allocator, total_levels);
            defer data.deinit(allocator);

            const IdxSlice = struct { offset: usize, total: usize };

            // A map from old patches on l to new patches on l+1.
            var patch_map: []IdxSlice = try allocator.alloc(IdxSlice, grid.patches.len);
            defer allocator.free(patch_map);

            // ************************************
            // Iterate from total_levels - 1 to 0

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(allocator);
            defer arena.deinit();

            const scratch: Allocator = arena.allocator();

            for (0..total_levels - 1) |rev_level_id| {
                const level_id: usize = total_levels - 2 - rev_level_id;
                const target_id: usize = level_id + 1;

                const level = grid.levels[level_id];

                data.blocks[target_id].clearRetainingCapacity();
                data.patches[target_id].clearRetainingCapacity();
                data.children[target_id].clearRetainingCapacity();

                // *******************************
                // Loop over every patch on l

                for (level.patch_offset..level.patch_offset + level.patch_total) |patch_id| {
                    // Reset arena for new "frame"
                    defer _ = arena.reset(.retain_capacity);

                    const patch = grid.patches[patch_id];
                    const patch_space: IndexSpace = IndexSpace.fromBox(patch.bounds);

                    // *****************************************************
                    // Find all grandchildren of this patch (on l + 2)

                    const grandchild_count: usize = blk: {
                        var result: usize = 0;
                        for (patch.children_offset..patch.children_offset + patch.children_total) |child_id| {
                            result += patch_map[child_id].total;
                        }
                        break :blk result;
                    };

                    // Now fill grandchildren and cluster to patch maps

                    // A collection of l+2 patches in l patch space
                    const grandchildren = try scratch.alloc(IndexBox, grandchild_count);
                    defer scratch.free(grandchildren);

                    // A map from granchildren to index into patches array on l+2 in `data`
                    const grandchild_to_patch = try scratch.alloc(usize, grandchild_count);
                    defer scratch.free(grandchild_to_patch);

                    {
                        var cur: usize = 0;

                        for (patch.children_offset..patch.children_offset + patch.children_total) |child_id| {
                            const slice = patch_map[child_id];
                            for (slice.offset..slice.offset + slice.total) |grandchild_id| {
                                const bounds = data.blocks[target_id + 1].items[grandchild_id].bounds;

                                grandchildren[cur] = bounds.coarsened().coarsened().relativeTo(patch.bounds);
                                grandchild_to_patch[cur] = grandchild_id;

                                cur += 1;
                            }
                        }
                    }

                    // ********************************
                    // Preprocess tags ****************

                    const source_tags: []const bool = tile_map.slice(patch_id, tags);

                    // Using new patches on l + 2

                    const patch_tags = try scratch.alloc(bool, tile_map.total(patch_id));
                    defer scratch.free(patch_tags);

                    @memcpy(patch_tags, source_tags);

                    for (grandchildren) |grandchild| {
                        var cluster: IndexBox = grandchild;

                        for (0..N) |i| {
                            if (cluster.origin[i] > 0) {
                                cluster.origin[i] -= 1;
                                cluster.size[i] += 1;
                            }

                            if (cluster.origin[i] + cluster.size[i] < patch_space.size[i]) {
                                cluster.size[i] += 1;
                            }
                        }

                        patch_space.fillWindow(cluster, bool, patch_tags, true);
                    }

                    // *************************************************************
                    // Run point clustering algorithm to find new blocks on l + 1

                    var cluster_space: ClusterSpace = .{
                        .size = patch_space.size,
                        .min_efficiency = self.block_efficiency,
                        .max_tiles = self.block_max_tiles,
                    };

                    // New blocks on l+1 (in old l space).
                    const blocks = try cluster_space.points(scratch, patch_tags);
                    defer scratch.free(blocks);

                    // *************************************************************
                    // Run block clustering algorithm to find new patchs on l + 1

                    // Combined blocks + grandchildren
                    const combined_blocks = try scratch.alloc(IndexBox, blocks.len + grandchildren.len);
                    defer scratch.free(combined_blocks);

                    @memcpy(combined_blocks[0..blocks.len], blocks);
                    @memcpy(combined_blocks[blocks.len..], grandchildren);

                    cluster_space.min_efficiency = self.patch_efficiency;
                    cluster_space.max_tiles = self.patch_max_tiles;

                    const patches = try cluster_space.blocks(scratch, combined_blocks);
                    defer patches.deinit(scratch);

                    // Update patch map
                    patch_map[patch_id].offset = data.patches[target_id].items.len;
                    patch_map[patch_id].total = patches.clusters.len;

                    // *****************************************
                    // Update Mesh Heirarchy

                    try data.patches[target_id].ensureUnusedCapacity(allocator, patches.clusters.len);
                    try data.blocks[target_id].ensureUnusedCapacity(allocator, blocks.len);
                    try data.children[target_id].ensureUnusedCapacity(allocator, grandchildren.len);

                    const patch_offset = data.patches[target_id].items.len;

                    for (0..patches.clusters.len) |idx| {
                        // Find new l + 1 patch bounds in l global space
                        var bounds: IndexBox = undefined;
                        bounds.origin = patch.bounds.globalFromLocal(patches.clusters[idx].origin);
                        bounds.size = patches.clusters[idx].size;

                        const children_offset: usize = data.children[target_id].items.len;
                        var children_total: usize = 0;

                        const block_offset: usize = data.blocks[target_id].items.len;
                        var block_total: usize = 0;

                        for (patches.children[idx]) |child| {
                            if (child >= blocks.len) {
                                // This is a child patch
                                data.children[target_id].appendAssumeCapacity(grandchild_to_patch[child - blocks.len]);

                                children_total += 1;
                            } else {
                                // This is a block that is in this patch
                                // Find new l + 1 block bounds in l global space
                                var block_bounds: IndexBox = undefined;
                                block_bounds.origin = patch.bounds.globalFromLocal(blocks[child].origin);
                                block_bounds.size = blocks[child].size;

                                data.blocks[target_id].appendAssumeCapacity(.{
                                    .bounds = block_bounds.refined(),
                                    .patch = patch_offset + idx,
                                });

                                block_total += 1;
                            }
                        }

                        data.patches[target_id].appendAssumeCapacity(.{
                            .bounds = bounds.refined(),
                            .level = target_id,
                            .parent = null,
                            .children_offset = children_offset,
                            .children_total = children_total,
                            .block_offset = block_offset,
                            .block_total = block_total,
                        });
                    }
                }
            }

            // **************************
            // Fill base level of data

            const base_child_offset: usize = if (total_levels > 1) 1 else 0;
            const base_child_total = if (total_levels > 1) data.patches[1].items.len else 0;

            const base_block: Block = .{
                .bounds = .{ .origin = splat(0), .size = grid.index_size },
                .patch = 0,
            };

            var base_patch: Patch = .{
                .bounds = .{ .origin = splat(0), .size = grid.index_size },
                .level = 0,
                .parent = null,
                .children_offset = 0,
                .children_total = base_child_total,
                .block_offset = 0,
                .block_total = 1,
            };

            const base_level: Level = .{
                .tile_size = grid.index_size,
                .patch_offset = 0,
                .patch_total = 1,
                .block_offset = 0,
                .block_total = 1,
            };

            try data.patches[0].append(allocator, base_patch);
            try data.children[0].ensureUnusedCapacity(allocator, base_child_total);

            for (0..base_child_total) |id| {
                data.children[0].appendAssumeCapacity(id);
            }

            base_patch.children_offset = base_child_offset;

            // **************************
            // Flatten

            var blocks: ArrayListUnmanaged(Block) = .{ .capacity = grid.block_capacity, .items = grid.blocks };
            var patches: ArrayListUnmanaged(Patch) = .{ .capacity = grid.patch_capacity, .items = grid.patches };
            var levels: ArrayListUnmanaged(Level) = .{ .capacity = grid.level_capacity, .items = grid.levels };

            blocks.clearRetainingCapacity();
            patches.clearRetainingCapacity();
            levels.clearRetainingCapacity();

            try blocks.ensureUnusedCapacity(grid.gpa, data.blockTotal() + 2);
            try patches.ensureUnusedCapacity(grid.gpa, data.patchTotal() + 2);
            try levels.ensureUnusedCapacity(grid.gpa, total_levels);

            // *****************************
            // Base level

            blocks.appendAssumeCapacity(base_block);
            patches.appendAssumeCapacity(base_patch);
            levels.appendAssumeCapacity(base_level);

            // **********************
            // Higher levels

            for (0..total_levels - 1) |level_id| {
                const target_id = level_id + 1;

                const global_patch_offset = patches.items.len;
                const global_block_offset = blocks.items.len;

                const prev_patch_offset: usize = global_patch_offset - data.patches[level_id].items.len;
                const next_patch_offset: usize = global_patch_offset + data.patches[target_id].items.len;

                // Loop over patches in child order (so we can set parent id)
                for (data.patches[level_id].items, 0..) |patch, patch_id| {
                    for (patch.children_offset..patch.children_offset + patch.children_total) |child_id| {
                        const child = data.children[level_id].items[child_id];

                        const target_patch = data.patches[target_id].items[child];

                        const patch_offset = patches.items.len;

                        patches.appendAssumeCapacity(.{
                            .bounds = target_patch.bounds,
                            .level = target_id,
                            .parent = patch_id + prev_patch_offset,
                            .children_offset = next_patch_offset + target_patch.children_offset,
                            .children_total = target_patch.children_total,
                            .block_offset = global_block_offset + target_patch.block_offset,
                            .block_total = target_patch.block_total,
                        });

                        for (data.blocks[target_id].items[target_patch.block_offset .. target_patch.block_offset + target_patch.block_total]) |block| {
                            blocks.appendAssumeCapacity(.{
                                .bounds = block.bounds,
                                .patch = patch_offset,
                            });
                        }
                    }
                }

                levels.appendAssumeCapacity(.{
                    .tile_size = IndexMixin.scaled(levels.items[level_id].tile_size, 2),
                    .patch_offset = global_patch_offset,
                    .patch_total = data.patches[target_id].items.len,
                    .block_offset = global_block_offset,
                    .block_total = data.blocks[target_id].items.len,
                });
            }

            grid.block_capacity = blocks.capacity;
            grid.patch_capacity = patches.capacity;
            grid.level_capacity = levels.capacity;
            grid.blocks = blocks.items;
            grid.patches = patches.items;
            grid.levels = levels.items;
        }

        /// An unflattened representation of a block structured mesh.
        const MeshHeirarchy = struct {
            blocks: []ArrayListUnmanaged(Block),
            patches: []ArrayListUnmanaged(Patch),
            children: []ArrayListUnmanaged(usize),

            fn init(allocator: Allocator, total_levels: usize) !@This() {
                const blocks: []ArrayListUnmanaged(Block) = try allocator.alloc(ArrayListUnmanaged(Block), total_levels);
                @memset(blocks, .{});

                errdefer {
                    for (blocks) |*level_block| {
                        level_block.deinit(allocator);
                    }

                    allocator.free(blocks);
                }

                const patches: []ArrayListUnmanaged(Patch) = try allocator.alloc(ArrayListUnmanaged(Patch), total_levels);
                @memset(patches, .{});

                errdefer {
                    for (patches) |*level_patch| {
                        level_patch.deinit(allocator);
                    }

                    allocator.free(patches);
                }

                const children: []ArrayListUnmanaged(usize) = try allocator.alloc(ArrayListUnmanaged(usize), total_levels);
                @memset(children, .{});

                errdefer {
                    for (children) |*level_child| {
                        level_child.deinit(allocator);
                    }

                    allocator.free(children);
                }

                return .{
                    .blocks = blocks,
                    .patches = patches,
                    .children = children,
                };
            }

            fn deinit(self: *@This(), allocator: Allocator) void {
                for (self.blocks) |*level_block| {
                    level_block.deinit(allocator);
                }

                for (self.patches) |*level_patch| {
                    level_patch.deinit(allocator);
                }

                for (self.children) |*level_child| {
                    level_child.deinit(allocator);
                }

                allocator.free(self.blocks);
                allocator.free(self.patches);
                allocator.free(self.children);
            }

            fn blockTotal(self: @This()) usize {
                var result: usize = 0;

                for (self.blocks) |block_list| {
                    result += block_list.items.len;
                }

                return result;
            }

            fn patchTotal(self: @This()) usize {
                var result: usize = 0;

                for (self.patches) |patch_list| {
                    result += patch_list.items.len;
                }
                return result;
            }
        };
    };
}
