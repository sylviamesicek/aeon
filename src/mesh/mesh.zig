//! This module provides the core functionality for defining systems across meshes,
//! adaptively refining those meshes, filling boundaries, and solving partial
//! differential equations on these domains.

// std imports

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const ArenaAllocator = std.heap.ArenaAllocator;
const assert = std.debug.assert;
const exp2 = std.math.exp2;
const maxInt = std.math.maxInt;

// Root imports

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");

// Submodules

// const levels = @import("levels.zig");

// ************************
// Mesh *******************
// ************************

/// A contigious and uniformly rectangular block of data on the mesh.
pub fn Block(comptime N: usize) type {
    return struct {
        /// Bounds of this block
        bounds: IndexBox,
        /// Patch this block belongs to
        patch: usize,
        /// Offset into global cell vector
        cell_offset: usize = 0,
        /// Total number of cells in this block
        cell_total: usize = 0,

        const IndexBox = geometry.Box(N, usize);

        // pub fn format(
        //     self: @This(),
        //     comptime fmt: []const u8,
        //     options: std.fmt.FormatOptions,
        //     writer: anytype,
        // ) !void {
        //     _ = options;
        //     _ = fmt;
        //     _ = self;

        //     try writer.print(
        //         \\Block:
        //         \\    Origin: {any}, Size: {any},
        //         \\    Patch: {}
        //         \\    Cell
        //     , .{});
        // }
    };
}

pub fn Patch(comptime N: usize) type {
    return struct {
        bounds: IndexBox,
        level: usize,
        parent: ?usize,
        children_offset: usize,
        children_total: usize,
        block_offset: usize,
        block_total: usize,
        tile_offset: usize = 0,
        tile_total: usize = 0,

        const IndexBox = geometry.Box(N, usize);
    };
}

pub fn Level(comptime N: usize) type {
    return struct {
        index_size: [N]usize,
        patch_offset: usize,
        patch_total: usize,
        block_offset: usize,
        block_total: usize,

        const IndexBox = geometry.Box(N, usize);
    };
}

/// Represents a block structured mesh, ie a numerical discretisation of a physical domain into a number of levels,
/// each of which consists of patches of tiles, and blocks of cells. This class handles accessing and
/// storing data for each cell/tile, allocating ghost cells, and running regridding and
/// transfer algorithms.
pub fn Mesh(comptime N: usize) type {
    return struct {
        /// Allocator used for various arraylists stored in this struct.
        gpa: Allocator,
        /// THe physical box that this mesh covers.
        physical_bounds: RealBox,
        /// The dimensions of the mesh in index space.
        index_size: [N]usize,
        /// The number of cells along each time edge.
        tile_width: usize,
        /// Total number of tiles in mesh
        tile_total: usize = 0,
        /// Total number of cells in mesh
        cell_total: usize = 0,

        block_capacity: usize,
        patch_capacity: usize,
        level_capacity: usize,

        blocks: []Block(N),
        patches: []Patch(N),
        levels: []Level(N),

        block_map: []usize,

        /// Configuration for a mesh.
        pub const Config = struct {
            physical_bounds: RealBox,
            index_size: [N]usize,
            tile_width: usize,

            /// Checks that the given mesh config is valid.
            pub fn check(self: Config) void {
                assert(self.tile_width >= 1);
                for (0..N) |i| {
                    assert(self.index_size[i] > 0);
                    assert(self.physical_bounds.size[i] > 0.0);
                }
            }
        };

        // Aliases
        const Self = @This();
        const IndexBox = geometry.Box(N, usize);
        const RealBox = geometry.Box(N, f64);
        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Partitions = geometry.Partitions(N);
        const Region = geometry.Region(N);

        const BlockList = ArrayListUnmanaged(Block(N));
        const PatchList = ArrayListUnmanaged(Patch(N));
        const LevelList = ArrayListUnmanaged(Level(N));

        // Mixins
        const Index = geometry.Index(N);

        const add = Index.add;
        const sub = Index.sub;
        const scaled = Index.scaled;
        const splat = Index.splat;
        const toSigned = Index.toSigned;

        /// Initialises a new mesh with an general purpose allocator, subject to the given configuration.
        pub fn init(allocator: Allocator, config: Config) !Self {
            // Check config
            config.check();

            var blocks: BlockList = .{};
            errdefer blocks.deinit(allocator);

            var patches: PatchList = .{};
            errdefer patches.deinit(allocator);

            var levels: LevelList = .{};
            errdefer levels.deinit(allocator);

            try blocks.append(allocator, .{
                .bounds = .{
                    .origin = splat(0),
                    .size = config.index_size,
                },
                .patch = 0,
            });

            try patches.append(allocator, .{
                .bounds = .{
                    .origin = splat(0),
                    .size = config.index_size,
                },
                .level = 0,
                .parent = null,
                .children_offset = 0,
                .children_total = 0,
                .block_offset = 0,
                .block_total = 1,
            });

            try levels.append(allocator, .{
                .index_size = config.index_size,
                .patch_offset = 0,
                .patch_total = 1,
                .block_offset = 0,
                .block_total = 1,
            });

            var self: Self = .{
                .gpa = allocator,
                .index_size = config.index_size,
                .physical_bounds = config.physical_bounds,
                .tile_width = config.tile_width,
                .block_capacity = blocks.capacity,
                .patch_capacity = patches.capacity,
                .level_capacity = levels.capacity,
                .blocks = blocks.items,
                .patches = patches.items,
                .levels = levels.items,
                .block_map = &.{},
            };

            self.computeOffsets();

            // Reset and free block map
            self.block_map = try self.gpa.alloc(usize, self.tile_total);
            errdefer self.gpa.free(self.block_map);

            self.buildBlockMap();

            return self;
        }

        /// Deinitalises a mesh.
        pub fn deinit(self: *Self) void {
            var blocks: BlockList = .{
                .capacity = self.block_capacity,
                .items = self.blocks,
            };

            var patches: PatchList = .{
                .capacity = self.patch_capacity,
                .items = self.patches,
            };

            var levels: LevelList = .{
                .capacity = self.level_capacity,
                .items = self.levels,
            };

            levels.deinit(self.gpa);
            patches.deinit(self.gpa);
            blocks.deinit(self.gpa);

            self.gpa.free(self.block_map);
        }

        // *************************
        // Block Map ***************
        // *************************

        pub fn buildTagsFromCells(self: *const Self, dest: []bool, src: []const bool) void {
            @memset(dest, false);

            for (self.blocks) |block| {
                const patch = self.patches[block.patch];

                const block_src = src[block.cell_offset .. block.cell_offset + block.cell_total];
                const block_dest = dest[patch.tile_offset .. patch.tile_offset + patch.tile_total];

                const patch_space = IndexSpace.fromBox(patch.bounds);
                const block_space = IndexSpace.fromBox(block.bounds);
                const cell_space = IndexSpace.fromSize(Index.scaled(block.bounds.size, self.tile_width));

                var tiles = block_space.cartesianIndices();

                while (tiles.next()) |tile| {
                    const origin = Index.scaled(tile, self.tile_width);

                    const patch_tile = patch.bounds.localFromGlobal(block.bounds.globalFromLocal(tile));
                    const patch_linear = patch_space.linearFromCartesian(patch_tile);

                    var cells = IndexSpace.fromSize(splat(self.tile_width)).cartesianIndices();

                    while (cells.next()) |cell| {
                        const global_cell = Index.add(origin, cell);

                        if (block_src[cell_space.linearFromCartesian(global_cell)]) {
                            block_dest[patch_linear] = true;
                            break;
                        }
                    }
                }
            }
        }

        // /// Builds a block child map. ie a map from each tile in the mesh to the id of the block that covers it
        // /// on the next level, if any.
        // pub fn buildBlockChildMap(self: *const Self, map: []usize) void {
        //     assert(map.len == self.tile_total);

        //     @memset(map, maxInt(usize));

        //     for (0..self.active_levels - 1) |l| {
        //         const level = self.getLevel(l);
        //         const refined = self.getLevel(l + 1);

        //         for (0..level.patchTotal()) |patch| {
        //             const patch_bounds: IndexBox = level.patches.items(.bounds)[patch];
        //             const patch_map: []usize = self.patchTileSlice(l, patch, map);
        //             for (level.childrenSlice(patch)) |child_patch| {
        //                 const offset = refined.patches.items(.block_offset)[child_patch];
        //                 const total = refined.patches.items(.block_total)[child_patch];
        //                 for (offset..offset + total) |child_block| {
        //                     var bounds = refined.blocks.items(.bounds)[child_block];
        //                     bounds.coarsen();

        //                     IndexSpace.fromBox(patch_bounds).fillWindow(bounds.relativeTo(patch), usize, patch_map, child_block);
        //                 }
        //             }
        //         }

        //         for (level.blocks.items(.bounds), level.blocks.items(.patch), 0..) |bounds, patch, id| {
        //             const pbounds: IndexBox = level.patches.items(.bounds)[patch];
        //             const tile_to_block: []usize = self.patchTileSlice(l, patch, map);
        //             IndexSpace.fromBox(pbounds).fillWindow(bounds.relativeTo(pbounds), usize, tile_to_block, id);
        //         }
        //     }
        // }

        // pub fn buildTransferMap(self: *const Self, map: []usize) !void {
        //     assert(map.len == self.transferTotal());

        //     @memset(map, std.math.maxInt(usize));
        //     @memset(self.baseTransferSlice(usize, map), 0);

        //     for (self.levels.items, 0..) |*level, l| {
        //         const level_map: []usize = self.levelTransferSlice(l, usize, map);

        //         for (level.transfer_blocks.items(.bounds), level.transfer_blocks.items(.patch), 0..) |bounds, parent, id| {
        //             const pbounds: IndexBox = level.transfer_patches.items(.bounds)[parent];
        //             const tile_offset: usize = level.transfer_patches.items(.tile_offset)[parent];
        //             const tile_total: usize = level.transfer_patches.items(.tile_total)[parent];
        //             const tile_to_block: []usize = level_map[tile_offset..(tile_offset + tile_total)];

        //             pbounds.space().fillSubspace(bounds.relativeTo(pbounds), usize, tile_to_block, id);
        //         }
        //     }
        // }

        // ***************************
        // Physical Bounds ***********
        // ***************************

        /// Gets the physical bounds of a higher level
        pub fn blockPhysicalBounds(self: *const Self, block: usize) RealBox {
            const patch = self.blocks[block].patch;
            const level = self.patches[patch].level;
            const index_size: [N]usize = self.levels[level].index_size;

            // Get bounds of this block
            const bounds = self.blocks[block].bounds;

            var physical_bounds: RealBox = undefined;

            for (0..N) |i| {
                const sratio: f64 = @as(f64, @floatFromInt(bounds.size[i])) / @as(f64, @floatFromInt(index_size[i]));
                const oratio: f64 = @as(f64, @floatFromInt(bounds.origin[i])) / @as(f64, @floatFromInt(index_size[i]));

                physical_bounds.size[i] = self.physical_bounds.size[i] * sratio;
                physical_bounds.origin[i] = self.physical_bounds.origin[i] + self.physical_bounds.size[i] * oratio;
            }

            return physical_bounds;
        }

        // *************************
        // Regridding **************
        // *************************

        pub const RegridConfig = struct {
            max_levels: usize,
            patch_efficiency: f64,
            patch_max_tiles: usize,
            block_efficiency: f64,
            block_max_tiles: usize,
        };

        pub fn regrid(
            self: *Self,
            allocator: Allocator,
            tags: []const bool,
            config: RegridConfig,
        ) !void {
            // The total levels must be >= 1, but may not be greater than self.levels.len + 1.
            const total_levels = @max(1, @min(config.max_levels, self.levels.len + 1));

            // **********************************
            // Allocate Per level data

            var data: MeshByLevel = try MeshByLevel.init(allocator, total_levels);
            defer data.deinit(allocator);

            const IdxSlice = struct { offset: usize, total: usize };

            // A map from old patches on l to new patches on l+1.
            var patch_map: []IdxSlice = try allocator.alloc(IdxSlice, self.patches.len);
            defer allocator.free(patch_map);

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(allocator);
            defer arena.deinit();

            var scratch: Allocator = arena.allocator();

            // Iterate from total_levels - 1 to 0
            for (0..total_levels - 1) |rev_level_id| {
                const level_id: usize = total_levels - 2 - rev_level_id;
                const target_id: usize = level_id + 1;

                const level = self.levels[level_id];

                data.blocks[target_id].clearRetainingCapacity();
                data.patches[target_id].clearRetainingCapacity();
                data.children[target_id].clearRetainingCapacity();

                // *******************************
                // Loop over every patch on l ****

                for (level.patch_offset..level.patch_offset + level.patch_total) |patch_id| {
                    // Reset arena for new "frame"
                    defer _ = arena.reset(.retain_capacity);

                    const patch = self.patches[patch_id];
                    const patch_space: IndexSpace = IndexSpace.fromBox(patch.bounds);

                    const source_tags = tags[patch.tile_offset .. patch.tile_offset + patch.tile_total];

                    // *****************************************************
                    // Find all new patches on l + 2 and use as clusters ***

                    // First, find the total number of grandchildren of this patch

                    const child_count: usize = blk: {
                        var result: usize = 0;
                        for (patch.children_offset..patch.children_offset + patch.children_total) |child_id| {
                            result += patch_map[child_id].total;
                        }
                        break :blk result;
                    };

                    // Now fill clusters and cluster to patch
                    const clusters = try scratch.alloc(IndexBox, child_count);
                    defer scratch.free(clusters);

                    const cluster_to_patch = try scratch.alloc(usize, child_count);
                    defer scratch.free(cluster_to_patch);

                    {
                        var cur: usize = 0;

                        for (patch.children_offset..patch.children_offset + patch.children_total) |child_id| {
                            const slice = patch_map[child_id];
                            for (slice.offset..slice.offset + slice.total) |grandchild_id| {
                                const bounds = data.blocks[target_id + 1].items[grandchild_id].bounds;

                                clusters[cur] = bounds.coarsened().coarsened().relativeTo(patch.bounds);
                                cluster_to_patch[cur] = grandchild_id;

                                cur += 1;
                            }
                        }
                    }

                    // ********************************
                    // Preprocess tags ****************

                    const patch_tags = try scratch.alloc(bool, patch.tile_total);
                    defer scratch.free(patch_tags);

                    @memcpy(patch_tags, source_tags);

                    preprocessTags(patch_tags, patch_space, clusters);

                    // ************************************************************
                    // Run partitioning algorithm to find new patches on l + 1 ****

                    const patch_partitions = try Partitions.init(scratch, patch_space.size, patch_tags, clusters, config.patch_max_tiles, config.patch_efficiency);
                    defer patch_partitions.deinit(allocator);

                    // Update patch map
                    patch_map[patch_id].offset = data.patches[target_id].items.len;
                    patch_map[patch_id].total = patch_partitions.len();

                    // *********************************************************
                    // Iterate new patches on l + 1 ****************************

                    try data.patches[target_id].ensureUnusedCapacity(allocator, patch_partitions.len());

                    for (0..patch_partitions.len()) |idx| {
                        const patch_children = patch_partitions.children[idx];
                        const patch_children_total = patch_children.end - patch_children.start;

                        // Partition bounds must be added to patch bounds to get origin in l global space
                        const partition_bounds = blk: {
                            var result: IndexBox = patch_partitions.bounds[idx];
                            result.origin = patch.bounds.globalFromLocal(patch_partitions.bounds[idx].origin);
                            break :blk result;
                        };

                        const partition_space = IndexSpace.fromBox(partition_bounds);

                        // *********************************************************
                        // Find blocks on l+2 that lie in new patch on l + 1 *******

                        const child_block_count = blk: {
                            var result: usize = 0;

                            const child_clusters = patch_partitions.buffer[patch_children.start..patch_children.end];

                            for (child_clusters) |cluster_id| {
                                const child_patch = data.patches[target_id + 1].items[cluster_to_patch[cluster_id]];

                                result += child_patch.block_total;
                            }

                            break :blk result;
                        };

                        const child_blocks = try scratch.alloc(IndexBox, child_block_count);
                        defer scratch.free(child_blocks);

                        {
                            var cur: usize = 0;

                            const child_clusters = patch_partitions.buffer[patch_children.start..patch_children.end];

                            for (child_clusters) |cluster_id| {
                                const child_patch = data.patches[target_id + 1].items[cluster_to_patch[cluster_id]];

                                for (child_patch.block_offset..child_patch.block_offset + child_patch.block_total) |block_id| {
                                    const bounds = data.blocks[target_id + 1].items[block_id].bounds;
                                    child_blocks[cur] = bounds.coarsened().coarsened().relativeTo(partition_bounds);
                                    cur += 1;
                                }
                            }
                        }

                        // *******************************************
                        // Use these blocks to preprocess tags *******

                        const partition_tags = try scratch.alloc(bool, partition_space.total());
                        defer scratch.free(partition_tags);

                        patch_space.copyWindow(partition_bounds, bool, partition_tags, source_tags);

                        preprocessTags(partition_tags, partition_space, child_blocks);

                        // *****************************************************
                        // Run partitioner to find blocks on new patch *********

                        const partitions = try Partitions.init(scratch, partition_space.size, partition_tags, &.{}, config.block_max_tiles, config.block_efficiency);
                        defer partitions.deinit(allocator);

                        const children_offset = data.children[target_id].items.len;
                        const block_offset = data.blocks[target_id].items.len;
                        const patch_offset = data.patches[target_id].items.len;

                        // ***********************************
                        // Add New patch *********************

                        const new_patch: Patch(N) = .{
                            .bounds = partition_bounds.refined(),
                            .level = target_id,
                            .parent = null,
                            .children_offset = children_offset,
                            .children_total = patch_children.end - patch_children.start,
                            .block_offset = block_offset,
                            .block_total = partitions.len(),
                        };

                        try data.children[target_id].ensureUnusedCapacity(allocator, patch_children_total);

                        for (patch_partitions.buffer[patch_children.start..patch_children.end]) |cluster_id| {
                            data.children[target_id].appendAssumeCapacity(cluster_to_patch[cluster_id]);
                        }

                        try data.blocks[target_id].ensureUnusedCapacity(allocator, partitions.len());

                        for (partitions.bounds) |block| {
                            const bounds = blk: {
                                var result: IndexBox = block;
                                result.origin = partition_bounds.globalFromLocal(result.origin);
                                break :blk result;
                            };

                            data.blocks[target_id].appendAssumeCapacity(.{
                                .bounds = bounds.refined(),
                                .patch = patch_offset,
                            });
                        }

                        data.patches[target_id].appendAssumeCapacity(new_patch);
                    }
                }
            }

            // *******************************
            // Flatten ***********************

            var blocks: BlockList = .{
                .capacity = self.block_capacity,
                .items = self.blocks,
            };

            var patches: PatchList = .{
                .capacity = self.patch_capacity,
                .items = self.patches,
            };

            var levels: LevelList = .{
                .capacity = self.level_capacity,
                .items = self.levels,
            };

            blocks.clearRetainingCapacity();
            patches.clearRetainingCapacity();
            levels.clearRetainingCapacity();

            try blocks.ensureUnusedCapacity(self.gpa, data.blockTotal() + 1);
            try patches.ensureUnusedCapacity(self.gpa, data.patchTotal() + 1);
            try levels.ensureUnusedCapacity(self.gpa, total_levels);

            // *****************************
            // Base patch ******************

            const base_child_offset: usize = if (total_levels > 1) 1 else 0;
            const base_child_total = if (total_levels > 1) data.patches[1].items.len else 0;

            const base_block: Block(N) = .{
                .bounds = .{
                    .origin = splat(0),
                    .size = self.index_size,
                },
                .patch = 0,
            };

            var base_patch: Patch(N) = .{
                .bounds = .{
                    .origin = splat(0),
                    .size = self.index_size,
                },
                .level = 0,
                .parent = null,
                .children_offset = base_child_offset,
                .children_total = base_child_total,
                .block_offset = 0,
                .block_total = 1,
            };

            const base_level: Level(N) = .{
                .index_size = self.index_size,
                .patch_offset = 0,
                .patch_total = 1,
                .block_offset = 0,
                .block_total = 1,
            };

            blocks.appendAssumeCapacity(base_block);
            patches.appendAssumeCapacity(base_patch);
            levels.appendAssumeCapacity(base_level);

            // Now fill higher levels
            base_patch.children_offset = 0;
            try data.patches[0].append(allocator, base_patch);
            try data.children[0].ensureUnusedCapacity(allocator, base_child_total);

            for (0..base_child_total) |id| {
                data.children[0].appendAssumeCapacity(id);
            }

            for (0..total_levels - 1) |level_id| {
                const target_id = level_id + 1;

                const global_patch_offset = patches.items.len;
                const global_block_offset = patches.items.len;

                const prev_patch_offset: usize = global_patch_offset - data.patches[level_id].items.len;
                const next_patch_offset: usize = global_patch_offset + data.patches[target_id].items.len;

                // Loop over patches in child order

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
                    .index_size = Index.scaled(levels.items[level_id].index_size, 2),
                    .patch_offset = global_patch_offset,
                    .patch_total = data.patches[target_id].items.len,
                    .block_offset = global_block_offset,
                    .block_total = data.blocks[target_id].items.len,
                });
            }

            self.block_capacity = blocks.capacity;
            self.patch_capacity = patches.capacity;
            self.level_capacity = levels.capacity;
            self.blocks = blocks.items;
            self.patches = patches.items;
            self.levels = levels.items;

            self.computeOffsets();

            // TODO Replace with array list unmanaged

            const block_map = try self.gpa.alloc(usize, self.tile_total);
            errdefer self.gpa.free(block_map);

            self.gpa.free(self.block_map);

            self.block_map = block_map;

            // Reset and free block map
            self.gpa.free(self.block_map);
            self.block_map = try self.gpa.alloc(usize, self.tile_total);

            self.buildBlockMap();
        }

        const MeshByLevel = struct {
            blocks: []BlockList,
            patches: []PatchList,
            children: []ArrayListUnmanaged(usize),

            fn init(allocator: Allocator, total_levels: usize) !@This() {
                const blocks: []BlockList = try allocator.alloc(BlockList, total_levels);
                @memset(blocks, .{});

                errdefer {
                    for (blocks) |*level_block| {
                        level_block.deinit(allocator);
                    }

                    allocator.free(blocks);
                }

                const patches: []PatchList = try allocator.alloc(PatchList, total_levels);
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

        fn preprocessTags(tags: []bool, space: IndexSpace, blocks: []const IndexBox) void {
            for (blocks) |block| {
                var cluster: IndexBox = block;

                for (0..N) |i| {
                    if (cluster.origin[i] > 0) {
                        cluster.origin[i] -= 1;
                        cluster.size[i] += 1;
                    }

                    if (cluster.origin[i] + cluster.size[i] < space.size[i]) {
                        cluster.size[i] += 1;
                    }
                }

                space.fillWindow(cluster, bool, tags, true);
            }
        }

        fn computeOffsets(self: *Self) void {
            var cell_offset: usize = 0;

            for (self.blocks) |*block| {
                const total = IndexSpace.fromSize(scaled(block.bounds.size, self.tile_width)).total();

                block.cell_offset = cell_offset;
                block.cell_total = total;
                cell_offset += total;
            }

            self.cell_total = cell_offset;

            var tile_offset: usize = 0;

            for (self.patches) |*patch| {
                const total = IndexSpace.fromBox(patch.bounds).total();

                patch.tile_offset = tile_offset;
                patch.tile_total = total;
                tile_offset += total;
            }

            self.tile_total = tile_offset;
        }

        /// Builds a block map, ie a map from each tile in the mesh to the id of the block that tile is in.
        fn buildBlockMap(self: *Self) void {
            assert(self.block_map.len == self.tile_total);

            @memset(self.block_map, maxInt(usize));

            for (self.patches) |patch| {
                const tile_to_block: []usize = self.block_map[patch.tile_offset .. patch.tile_offset + patch.tile_total];

                for (patch.block_offset..patch.block_total + patch.block_offset) |block_id| {
                    const block = self.blocks[block_id];
                    IndexSpace.fromBox(patch.bounds).fillWindow(block.bounds.relativeTo(patch.bounds), usize, tile_to_block, block_id);
                }
            }
        }
    };
}

// *****************************
// Transfer Map ****************
// *****************************

test "mesh regridding" {
    // const expect = std.testing.expect;
    // const expectEqualSlices = std.testing.expectEqualSlices;

    // const allocator = std.testing.allocator;

    // const config: Mesh(2).Config = .{
    //     .physical_bounds = .{
    //         .origin = [_]f64{ 0.0, 0.0 },
    //         .size = [_]f64{ 1.0, 1.0 },
    //     },
    //     .index_size = [_]usize{ 10, 10 },
    //     .tile_width = 16,
    // };

    // var mesh: Mesh(2) = try Mesh(2).init(allocator, config);
    // defer mesh.deinit();

    // var tags: []bool = try allocator.alloc(bool, mesh.tile_total);
    // defer allocator.free(tags);

    // // Tag all
    // @memset(tags, true);

    // try mesh.regrid(tags, 1, 0.1, 80, 0.7, 80);
}
