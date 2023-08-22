const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const ArenaAllocator = std.heap.ArenaAllocator;
const assert = std.debug.assert;
const exp2 = std.math.exp2;

const geometry = @import("../geometry/geometry.zig");
const Box = geometry.Box;
const Geometry = geometry.Geometry;
const IndexSpace = geometry.IndexSpace;
const UniformGeometry = geometry.UniformGeometry;
const Tiles = geometry.Tiles;
const Partitions = geometry.Partitions;

pub const TileSrc = enum(u2) {
    unchanged,
    /// If this bit is set, it indicates that the index points
    /// to a tile on level `l-1` and that the data must be
    /// interpolated to the new level.
    added,
    empty,
};

/// A mapping of from a tile on an old mesh
/// to a new one.
pub const TileMap = packed struct {
    src: TileSrc,
    /// The index of the source tile (if any)
    index: u62,
};

pub fn Mesh(comptime N: usize) type {
    return struct {
        gpa: Allocator,
        base: UniformGeometry(N),
        levels: ArrayListUnmanaged(Level),
        index_size: [N]usize,
        global_refinement: usize,
        total_tiles: usize,
        total_cells: usize,

        // Aliases
        const Self = @This();

        pub const Config = struct {
            physical_bounds: Box(N, f64),
            index_size: [N]usize,
            tile_width: usize,
            ghost_width: usize,
            global_refinement: usize,
        };

        pub const Level = struct {
            /// Geometry corresponding to this level.
            geometry: Geometry(N),
            /// Stores for each patch, the parent patch in level `l - 1`
            parents: ArrayListUnmanaged(usize),
            /// Stores for each patch, the children patches in level `l + 1`
            children: ArrayListUnmanaged([]usize),
            /// Index buffer (which children array points in to).
            index_buffer: ArrayListUnmanaged(usize),
            tile_offset: usize,
            cell_offset: usize,
        };

        pub fn init(allocator: Allocator, config: Config) Self {
            // Scale initial size by 2^global_refinement
            const scale: usize = exp2(config.global_refinement);

            var base_size: [N]usize = undefined;

            for (0..N) |axis| {
                base_size[axis] = scale * config.index_size[axis];
            }

            // Construct the base level
            const base = UniformGeometry(N){
                .physical_bounds = config.physical_bounds,
                .index_size = base_size,
                .tile_width = config.tile_width,
                .ghost_width = config.ghost_width,
            };

            return .{
                .gpa = allocator,
                .base = base,
                .levels = ArrayListUnmanaged(Level){},
                .index_size = config.index_size,
                .global_refinement = config.global_refinement,
                .total_tiles = base.tileTotal(),
                .total_cells = base.cellTotal(),
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.levels.items) |*level| {
                level.geometry.deinit();
                level.parents.deinit(self.gpa);
                level.children.deinit(self.gpa);
                level.index_buffer.deinit(self.gpa);
            }

            self.levels.deinit(self.gpa);
        }

        pub const RegridConfig = struct {
            max_level: usize,
            block_max_tiles: usize,
            block_efficiency: f64,
            patch_max_tiles: usize,
            patch_effciency: f64,
        };

        pub fn regrid(self: *Self, map: *ArrayList(TileMap), src: Self, tags: []const usize, config: RegridConfig) !void {
            // TODO: Ensure proper nesting by preprocessing tags.
            // TODO: Generate clusters correctly and update cluster heirarchy

            // 1. Copy base level, clear refined levels, and clear map.
            // ********************************************
            self.base = src.base;
            self.clearRetainingCapacity();
            map.clearRetainingCapacity();

            // 2. Set refined levels with appropriate settings.
            // ************************************************
            try self.ensureLength(config.max_level);
            self.propagateBaseSettings();

            // 3. Generate Levels on new mesh.
            // *******************************
            //     a. The tags that define level `l` are stored on
            //        the old level `l - 1`.
            //     b. Generate clusters by taking patches on `l+1`
            //        and scaling them to be on level `l-1`
            //     c. Run partitioning algorithm using these clusters
            //        and low target efficiency.
            //     d. Translate resulting patches to level `l` and mark
            //        tags appropriately.
            //     e. Run partitioning algorithm on these patches using
            //        higher efficiency.

            // Iterate each level in reverse order.
            for (0..config.max_level) |l| {
                // Index of the current level
                const level_index: usize = config.max_level - 1 - l;
                // Perform regridding.
                try self.regridLevel(src, tags, config, level_index);
            }

            // 4. Recompute level offsets and totals.
            // **************************************

            var tile_offset: usize = 0;
            var cell_offset: usize = 0;

            for (self.levels.items) |*level| {
                level.tile_offset = tile_offset;
                level.cell_offset = cell_offset;

                tile_offset += level.geometry.tileTotal();
                cell_offset += level.geometry.cellTotal();
            }

            self.total_cells = cell_offset;
            self.total_tiles = tile_offset;
        }
        fn regridLevel(
            self: *Self,
            src: Self,
            tags: []usize,
            config: RegridConfig,
            level_id: usize,
        ) !void {
            // Use arena allocator to cache memory allocations
            var arena: ArenaAllocator = ArenaAllocator.init(self.gpa);
            defer arena.deinit();

            var scratch = arena.allocator();

            // Get pointers to coarse level of old mesh and refined level of new mesh.
            const source_coarse: *const Level = &src.levels.items[level_id - 1];
            const source: *const Level = &src.levels.items[level_id - 1];
            _ = source;

            var target_coarse: *Level = &self.levels.items[level_id - 1];
            _ = target_coarse;
            var target: *Level = &self.levels.items[level_id];
            var target_refined: *Level = &self.levels.items[level_id + 1];

            // Offset into tags array.
            const level_tags = tags[source_coarse.tile_offset..];

            // Iterate the patches of the coarse level.
            for (source_coarse.geometry.patches.items(.bounds), source_coarse.geometry.patches.items(.offset), 0..) |bounds, offset, parent_id| {
                // Every "frame" reset arena.
                defer arena.reset(.retain_capacity);

                // Target coarse (which will be the target on the next iteration) should point to source_coarse.chil

                // Space corresponding to coarse level patch.
                const space = bounds.space();

                // Get a mutable slice to tags corresponding to this patch
                const coarse_tags = level_tags[offset..(offset + space.total())];

                // Run point clustering algorithm to determine new patches
                var partitions: Partitions(N) = .{};
                defer partitions.deinit(scratch);

                var clusters: ArrayListUnmanaged(Box(N, usize)) = .{};
                defer clusters.deinit(scratch);

                var cluster_indices: ArrayListUnmanaged(usize) = .{};
                defer cluster_indices.deinit(scratch);

                for (source_coarse.children.items[parent_id]) |child_id| {
                    for (target.children.items[child_id]) |refined_id| {
                        var cluster = target_refined.geometry.patches.items(.bounds)[refined_id];

                        cluster.coarsen();
                        cluster.coarsen();

                        try clusters.append(scratch, cluster);
                        try cluster_indices.append(scratch, refined_id);
                    }
                }

                try partitions.compute(
                    scratch,
                    .{
                        .size = bounds.size,
                        .tags = tags,
                        .clusters = clusters.items,
                    },
                    config.patch_max_tiles,
                    config.patch_effciency,
                );

                for (partitions.buffer.items) |index| {
                    try target.index_buffer.append(self.gpa, cluster_indices.items[index]);
                }

                for (partitions.blocks.items(.children)) |children| {
                    const child_offset: usize = (@intFromPtr(children.ptr) - @intFromPtr(partitions.buffer.items.ptr)) / @sizeOf(usize);
                    try target.children.append(self.gpa, target.index_buffer.items[child_offset..(child_offset + children.len)]);
                }

                try target.index_buffer.appendSlice(self.gpa, partitions.buffer.items);

                try target.children.appendSlice(self.gpa, partitions.blocks.items(.children));

                // Buffer for tags on new patches
                var patch_tags = ArrayListUnmanaged(usize){};
                defer patch_tags.deinit(scratch);
                // Buffer for blocks on new patches
                var patch_blocks = ArrayListUnmanaged(Box(N, usize)){};
                defer patch_blocks.deinit(scratch);
                // Cache object for building blocks on new patches.
                var patch_partitions: Partitions(N) = .{};
                defer patch_partitions.deinit(scratch);

                // Update hierarchy
                target.parents.resize(partitions.blocks.len);

                // Iterate newly computed patches
                for (partitions.blocks.items(.bounds), 0..) |patch, patch_id| {
                    // Represents patch space on coarse level
                    const patch_space = patch.space();
                    patch_tags.resize(scratch, patch_space.total());

                    space.window(patch, bool, patch_tags.items, coarse_tags);

                    // Perform point clustering.
                    try patch_partitions.compute(
                        scratch,
                        .{
                            .size = patch.widths,
                            .tags = patch_tags,
                            .clusters = &[_]usize{},
                        },
                        config.block_max_tiles,
                        config.block_efficiency,
                    );

                    // Translate from coarse to refined indices (scale by two).
                    patch_blocks.resize(scratch, patch_partitions.blocks.len);
                    @memcpy(patch_blocks.items, patch_partitions.blocks.items(.bounds));

                    for (patch_blocks.items) |*block| {
                        block.refine();
                    }

                    var patch_refined: Box(N, usize) = patch;
                    patch_refined.refine();

                    // Add this new patch to the refined geometry.
                    try target.geometry.addPatch(patch_refined, patch_blocks.items);
                    target.parents.appendAssumeCapacity(self.gpa, parent_id);

                    // Update parents of new patch to point to parent on coarse level
                    target.parents.items[patch_id] = parent_id;
                }
            }
        }

        fn repartitionCoarsePatch(
            partitions: *Partitions(N),
            allocator: Allocator,
            size: [N]usize,
            tags: []const bool,
            clusters: []const Box(N, usize),
            config: RegridConfig,
        ) !void {
            _ = config;
            _ = tags;
            _ = size;
            _ = partitions;
            var cclusters = allocator.alloc(Box(N, usize), clusters.len);
            defer allocator.free(cclusters);

            @memcpy(clusters, clusters);

            for (cclusters) |*cluster| {
                cluster.coarsen();
                cluster.coarsen();
            }
        }

        pub fn transfer(self: Self, map: ArrayList(TileMap), src: Self, new: *ArrayList(f64), old: ArrayList(f64)) !void {
            _ = old;
            _ = new;
            _ = src;
            _ = map;
            _ = self;
        }

        fn clearRetainingCapacity(self: *Self) void {
            for (self.levels.items) |*level| {
                level.geometry.clearRetainingCapacity();
                level.parents.clearRetainingCapacity();
                level.children.clearRetainingCapacity();
                level.index_buffer.clearRetainingCapacity();
            }
        }

        fn ensureLength(self: *Self, levels: usize) !void {
            if (levels > self.levels.items.len) {
                const n = levels - self.levels.items.len;
                const geom = Geometry(N).init(self.gpa, .{
                    .physical_bounds = .{
                        .origin = [1]f64{0.0} ** N,
                        .widths = [1]usize{0} ** N,
                    },
                    .index_size = [1]usize{0} ** N,
                    .tile_width = 0,
                    .ghost_width = 0,
                });

                try self.levels.appendNTimes(self.gpa, .{
                    .geometry = geom,
                    .parents = ArrayListUnmanaged(usize){},
                    .children = ArrayListUnmanaged(usize){},
                    .index_buffer = ArrayListUnmanaged(usize){},
                    .tile_offset = 0,
                    .cell_offset = 0,
                }, n);
            }
        }

        fn propagateBaseSettings(self: *Self) !void {
            // Now set physical bounds, index size, tile width and ghost width.
            for (self.levels.items, 1..) |*level, i| {
                level.geometry.physical_bounds = self.base.physical_bounds;
                level.geometry.index_size = self.base.index_size;
                level.geometry.tile_width = self.base.tile_width;
                level.geometry.ghost_width = self.base.ghost_width;

                const scale: usize = exp2(i);

                for (0..N) |axis| {
                    level.geometry.index_size[axis] *= scale;
                }
            }
        }
    };
}

pub fn Mesh2(comptime N: usize) type {
    return struct {
        gpa: Allocator,
        config: Config,
        tile_total: usize,
        cell_total: usize,
        levels: ArrayListUnmanaged(Level),
        active_levels: usize,

        // Aliases
        const Self = @This();
        const IndexBox = Box(N, usize);
        const RealBox = Box(N, f64);

        pub const Config = struct {
            physical_bounds: Box(N, f64),
            index_size: [N]usize,
            tile_width: usize,
            ghost_width: usize,
            global_refinement: usize,

            pub fn baseTileSize(self: Config) [N]usize {
                var size: [N]usize = undefined;

                const scale: usize = exp2(self.global_refinement);

                for (0..N) |axis| {
                    size[axis] = scale * self.index_size[axis];
                }

                return size;
            }

            pub fn baseCellSize(self: Config) [N]usize {
                var size: [N]usize = undefined;

                const scale: usize = exp2(self.global_refinement);

                for (0..N) |axis| {
                    size[axis] = scale * self.tile_width * self.index_size[axis] + 2 * self.ghost_width;
                }

                return size;
            }
        };

        pub const Block = struct {
            bounds: IndexBox,
            patch: usize,
            cell_offset: usize,
        };

        pub const Patch = struct {
            // Parent of this patch
            parent: usize,
            // Children of this patch
            children_offset: usize,
            children_count: usize,
            tile_offset: usize,
        };

        pub const Level = struct {
            blocks: MultiArrayList(Block),
            patches: MultiArrayList(Patch),
            /// Index buffer for children of patches
            children: ArrayListUnmanaged(usize),
            // Offset in tile array for this level
            tile_offset: usize,
            // Offset into cell array for this level
            cell_offset: usize,
            // Total number of tiles in this level
            tile_total: usize,
            // Total number of cells in this level.
            cell_total: usize,

            fn init() Level {
                return .{
                    .blocks = .{},
                    .patches = .{},
                    .children = .{},
                    .tile_offset = 0,
                    .cell_offset = 0,
                    .tile_total = 0,
                    .cell_total = 0,
                };
            }

            fn deinit(self: *Level, allocator: Allocator) void {
                self.blocks.deinit(allocator);
                self.patches.deinit(allocator);
                self.index_buffer.deinit(allocator);
            }
        };

        pub fn init(allocator: Allocator, config: Config) Self {
            // Scale initial size by 2^global_refinement
            const tile_space: IndexSpace(N) = .{ .size = config.baseTileSize() };
            const cell_space: IndexSpace(N) = .{ .size = config.baseCellSize() };

            return .{
                .gpa = allocator,
                .config = config,
                .tile_total = tile_space.total(),
                .cell_total = cell_space.total(),
                .levels = .{},
                .active_levels = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.levels.items) |*level| {
                level.deinit(self.gpa);
            }

            self.levels.deinit(self.gpa);
        }

        pub fn baseTileTotal(self: *const Self) usize {
            const space: IndexSpace(N) = .{ .size = self.config.baseTileSize() };
            return space.total();
        }

        pub fn baseCellTotal(self: *const Self) usize {
            const space: IndexSpace(N) = .{ .size = self.config.baseCellSize() };
            return space.total();
        }

        pub const RegridConfig = struct {
            max_level: usize,
            block_max_tiles: usize,
            block_efficiency: f64,
            patch_max_tiles: usize,
            patch_effciency: f64,
        };

        pub fn regrid(dest: *Self, map: *ArrayList(TileMap), src: *const Self, tags: []const usize, config: RegridConfig) !void {
            // TODO: Ensure proper nesting by preprocessing tags.
            // TODO: Generate clusters correctly and update cluster heirarchy

            // 1. Copy config, and clear map.
            // ********************************************
            dest.config = src.config;
            map.clearRetainingCapacity();

            // 2. Find total number of levels and preallocate dest.
            // **********************************************************
            const total_levels = count_levels: {
                assert(config.max_level >= src.active_levels);

                if (src.active_levels == config.max_level) {
                    break :count_levels config.max_level;
                }

                if (src.active_levels > 0) {
                    var tagged: bool = false;

                    for (tags[src.levels.getLast().tile_offset..]) |tag| {
                        tagged = tagged or tag;
                    }

                    break :count_levels tagged;
                } else {
                    var tagged: bool = false;

                    for (tags) |tag| {
                        tagged = tagged or tag;
                    }

                    break :count_levels tagged;
                }
            };

            while (total_levels > dest.levels.len) {
                dest.levels.append(dest.gpa, Level.init());
            }

            // 3. Generate Levels on new mesh.
            // *******************************
            //     a. The tags that define level `l` are stored on
            //        the old level `l - 1`.
            //     b. Generate clusters by taking patches on `l+1`
            //        and scaling them to be on level `l-1`
            //     c. Run partitioning algorithm using these clusters
            //        and low target efficiency.
            //     d. Translate resulting patches to level `l` and mark
            //        tags appropriately.
            //     e. Run partitioning algorithm on these patches using
            //        higher efficiency.

            // // Iterate each level in reverse order.
            // for (0..config.max_level) |l| {
            //     // Index of the current level
            //     const level_index: usize = config.max_level - 1 - l;
            //     // Perform regridding.
            //     try self.regridLevel(src, tags, config, level_index);
            // }

            // 4. Recompute level offsets and totals.
            // **************************************

            // var tile_offset: usize = 0;
            // var cell_offset: usize = 0;

            // for (self.levels.items) |*level| {
            //     level.tile_offset = tile_offset;
            //     level.cell_offset = cell_offset;

            //     tile_offset += level.geometry.tileTotal();
            //     cell_offset += level.geometry.cellTotal();
            // }

            // self.total_cells = cell_offset;
            // self.total_tiles = tile_offset;
        }
        fn regridLevel(
            self: *Self,
            src: Self,
            tags: []usize,
            config: RegridConfig,
            level_id: usize,
        ) !void {
            // Use arena allocator to cache memory allocations
            var arena: ArenaAllocator = ArenaAllocator.init(self.gpa);
            defer arena.deinit();

            var scratch = arena.allocator();

            // Get pointers to coarse level of old mesh and refined level of new mesh.
            const source_coarse: *const Level = &src.levels.items[level_id - 1];
            const source: *const Level = &src.levels.items[level_id - 1];
            _ = source;

            var target_coarse: *Level = &self.levels.items[level_id - 1];
            _ = target_coarse;
            var target: *Level = &self.levels.items[level_id];
            var target_refined: *Level = &self.levels.items[level_id + 1];

            // Offset into tags array.
            const level_tags = tags[source_coarse.tile_offset..];

            // Iterate the patches of the coarse level.
            for (source_coarse.geometry.patches.items(.bounds), source_coarse.geometry.patches.items(.offset), 0..) |bounds, offset, parent_id| {
                // Every "frame" reset arena.
                defer arena.reset(.retain_capacity);

                // Target coarse (which will be the target on the next iteration) should point to source_coarse.chil

                // Space corresponding to coarse level patch.
                const space = bounds.space();

                // Get a mutable slice to tags corresponding to this patch
                const coarse_tags = level_tags[offset..(offset + space.total())];

                // Run point clustering algorithm to determine new patches
                var partitions: Partitions(N) = .{};
                defer partitions.deinit(scratch);

                var clusters: ArrayListUnmanaged(Box(N, usize)) = .{};
                defer clusters.deinit(scratch);

                var cluster_indices: ArrayListUnmanaged(usize) = .{};
                defer cluster_indices.deinit(scratch);

                for (source_coarse.children.items[parent_id]) |child_id| {
                    for (target.children.items[child_id]) |refined_id| {
                        var cluster = target_refined.geometry.patches.items(.bounds)[refined_id];

                        cluster.coarsen();
                        cluster.coarsen();

                        try clusters.append(scratch, cluster);
                        try cluster_indices.append(scratch, refined_id);
                    }
                }

                try partitions.compute(
                    scratch,
                    .{
                        .size = bounds.size,
                        .tags = tags,
                        .clusters = clusters.items,
                    },
                    config.patch_max_tiles,
                    config.patch_effciency,
                );

                for (partitions.buffer.items) |index| {
                    try target.index_buffer.append(self.gpa, cluster_indices.items[index]);
                }

                for (partitions.blocks.items(.children)) |children| {
                    const child_offset: usize = (@intFromPtr(children.ptr) - @intFromPtr(partitions.buffer.items.ptr)) / @sizeOf(usize);
                    try target.children.append(self.gpa, target.index_buffer.items[child_offset..(child_offset + children.len)]);
                }

                try target.index_buffer.appendSlice(self.gpa, partitions.buffer.items);

                try target.children.appendSlice(self.gpa, partitions.blocks.items(.children));

                // Buffer for tags on new patches
                var patch_tags = ArrayListUnmanaged(usize){};
                defer patch_tags.deinit(scratch);
                // Buffer for blocks on new patches
                var patch_blocks = ArrayListUnmanaged(Box(N, usize)){};
                defer patch_blocks.deinit(scratch);
                // Cache object for building blocks on new patches.
                var patch_partitions: Partitions(N) = .{};
                defer patch_partitions.deinit(scratch);

                // Update hierarchy
                target.parents.resize(partitions.blocks.len);

                // Iterate newly computed patches
                for (partitions.blocks.items(.bounds), 0..) |patch, patch_id| {
                    // Represents patch space on coarse level
                    const patch_space = patch.space();
                    patch_tags.resize(scratch, patch_space.total());

                    space.window(patch, bool, patch_tags.items, coarse_tags);

                    // Perform point clustering.
                    try patch_partitions.compute(
                        scratch,
                        .{
                            .size = patch.widths,
                            .tags = patch_tags,
                            .clusters = &[_]usize{},
                        },
                        config.block_max_tiles,
                        config.block_efficiency,
                    );

                    // Translate from coarse to refined indices (scale by two).
                    patch_blocks.resize(scratch, patch_partitions.blocks.len);
                    @memcpy(patch_blocks.items, patch_partitions.blocks.items(.bounds));

                    for (patch_blocks.items) |*block| {
                        block.refine();
                    }

                    var patch_refined: Box(N, usize) = patch;
                    patch_refined.refine();

                    // Add this new patch to the refined geometry.
                    try target.geometry.addPatch(patch_refined, patch_blocks.items);
                    target.parents.appendAssumeCapacity(self.gpa, parent_id);

                    // Update parents of new patch to point to parent on coarse level
                    target.parents.items[patch_id] = parent_id;
                }
            }
        }

        fn repartitionCoarsePatch(
            partitions: *Partitions(N),
            allocator: Allocator,
            size: [N]usize,
            tags: []const bool,
            clusters: []const Box(N, usize),
            config: RegridConfig,
        ) !void {
            _ = config;
            _ = tags;
            _ = size;
            _ = partitions;
            var cclusters = allocator.alloc(Box(N, usize), clusters.len);
            defer allocator.free(cclusters);

            @memcpy(clusters, clusters);

            for (cclusters) |*cluster| {
                cluster.coarsen();
                cluster.coarsen();
            }
        }

        pub fn transfer(self: Self, map: ArrayList(TileMap), src: Self, new: *ArrayList(f64), old: ArrayList(f64)) !void {
            _ = old;
            _ = new;
            _ = src;
            _ = map;
            _ = self;
        }

        fn clearRetainingCapacity(self: *Self) void {
            for (self.levels.items) |*level| {
                level.geometry.clearRetainingCapacity();
                level.parents.clearRetainingCapacity();
                level.children.clearRetainingCapacity();
                level.index_buffer.clearRetainingCapacity();
            }
        }

        fn ensureLength(self: *Self, levels: usize) !void {
            if (levels > self.levels.items.len) {
                const n = levels - self.levels.items.len;
                const geom = Geometry(N).init(self.gpa, .{
                    .physical_bounds = .{
                        .origin = [1]f64{0.0} ** N,
                        .widths = [1]usize{0} ** N,
                    },
                    .index_size = [1]usize{0} ** N,
                    .tile_width = 0,
                    .ghost_width = 0,
                });

                try self.levels.appendNTimes(self.gpa, .{
                    .geometry = geom,
                    .parents = ArrayListUnmanaged(usize){},
                    .children = ArrayListUnmanaged(usize){},
                    .index_buffer = ArrayListUnmanaged(usize){},
                    .tile_offset = 0,
                    .cell_offset = 0,
                }, n);
            }
        }

        fn propagateBaseSettings(self: *Self) !void {
            // Now set physical bounds, index size, tile width and ghost width.
            for (self.levels.items, 1..) |*level, i| {
                level.geometry.physical_bounds = self.base.physical_bounds;
                level.geometry.index_size = self.base.index_size;
                level.geometry.tile_width = self.base.tile_width;
                level.geometry.ghost_width = self.base.ghost_width;

                const scale: usize = exp2(i);

                for (0..N) |axis| {
                    level.geometry.index_size[axis] *= scale;
                }
            }
        }
    };
}
