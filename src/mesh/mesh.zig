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
            bounds: IndexBox,
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
            // Total number of tiles in this level
            tile_total: usize,
            // Total number of cells in this level.
            cell_total: usize,
            // Offset in tile array for this level
            tile_offset: usize,
            // Offset into cell array for this level
            cell_offset: usize,

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

            fn childrenSlice(self: *Level, patch: usize) []usize {
                const offset = self.patches.items(.children_offset)[patch];
                const count = self.patches.items(.children_count)[patch];
                return self.children[offset..(offset + count)];
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

            // 1. Copy config, and clear map.
            // ********************************************
            dest.config = src.config;
            map.clearRetainingCapacity();

            // 2. Find total number of levels and preallocate dest.
            // **********************************************************
            const total_levels = count_levels: {
                // The max number of levels can not truncate the grid.
                assert(config.max_level >= src.active_levels);

                // Clamp to max levels
                if (src.active_levels == config.max_level) {
                    break :count_levels config.max_level;
                }

                // Check if any on the highest level is tagged
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

            // While necessary allocate additional levels, until dest reaches total levels.
            while (total_levels > dest.levels.len) {
                dest.levels.append(dest.gpa, Level.init());
            }

            // 3. Recursively generate levels on new mesh.
            // *******************************************

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(dest.gpa);
            defer arena.deinit(dest.gpa);

            var scratch: Allocator = arena.allocator();

            // Build a cache for children indices of underlying level.
            var children_cache: ArrayListUnmanaged(usize) = .{};
            defer children_cache.deinit(dest.gpa);

            // Bounds for base level.
            const bbounds: IndexBox = .{
                .origin = [1]usize ** N,
                .size = dest.config.baseTileSize(),
            };

            // Slices at top of scope ensures we don't reference a temporary.
            const bbounds_slice: []const IndexBox = &[_]IndexBox{bbounds};
            const boffsets_slice: []const usize = &[_]usize{0};

            // Loop through levels from highest to lowest
            for (0..total_levels) |reverse_level_id| {
                const level_id: usize = total_levels - 1 - reverse_level_id;

                // Get a mutable reference to the target level.
                var target: *Level = dest.levels.items[level_id];

                // Now we can clear the target arrays
                target.blocks.shrinkRetainingCapacity(0);
                target.patches.shrinkRetainingCapacity(0);
                target.children.shrinkRetainingCapacity(0);

                // Check if there exists a level higher than the current one.
                const refined_exists: usize = level_id < total_levels - 1;
                // Check if we are over base level.
                const coarse_exists: usize = level_id > 0;

                // Variables that depend on coarse existing
                var utags: []const usize = undefined;
                var ulen: usize = undefined;
                var ubounds: []const IndexBox = undefined;
                var uoffsets: []const usize = undefined;

                // 3.1 Build clusters on refined mesh to consider when repartitioning target.
                // **************************************************************************

                // Stores offsets per underlying patch into clusters array
                var cluster_offsets: []usize = try dest.gpa.alloc(usize, ulen + 1);
                defer dest.gpa.free(cluster_offsets);

                // Stores bounds of clusters
                var clusters: ArrayListUnmanaged(IndexBox) = .{};
                defer clusters.deinit(dest.gpa);

                // Stores map from cluster bounds to global index in l+1
                var cluster_index_map: ArrayListUnmanaged(usize) = .{};
                defer cluster_index_map.deinit(dest.gpa);

                if (coarse_exists) {
                    // Get underlying data
                    const underlying: *const Level = src.levels.items[level_id - 1];

                    utags = tags[underlying.tile_offset..(underlying.tile_offset + underlying.tile_total)];
                    ulen = underlying.patches.len;
                    ubounds = underlying.patches.items(.bounds);
                    uoffsets = underlying.patches.items(.tile_offsets);

                    //
                    for (0..ulen) |underlying_id| {
                        if (refined_exists) {
                            const target_refined: *const Level = dest.levels.items[level_id + 1];

                            const offset: usize = underlying.patches.items(.children_offset)[underlying_id];
                            const count: usize = underlying.patches.items(.children_count)[underlying_id];

                            for (underlying.children[offset..(offset + count)]) |child| {
                                for (children_cache.items[child]..children_cache.items[child + 1]) |refined| {
                                    var cluster: IndexBox = target_refined.patches.items(.bounds)[refined];

                                    cluster.coarsen();
                                    cluster.coarsen();

                                    try clusters.append(cluster);
                                    try cluster_index_map.append(refined);
                                }
                            }
                        }

                        cluster_offsets[underlying_id + 1] = clusters.len;
                    }
                } else {
                    // Otherwise use base data
                    utags = tags[0..dest.baseTileTotal()];
                    ulen = 1;
                    ubounds = bbounds_slice;
                    uoffsets = boffsets_slice;
                    // All registered patches are children of base
                    for (0..children_cache.len, 1..) |i, j| {
                        if (refined_exists) {
                            const target_refined: *const Level = dest.levels.items[level_id + 1];

                            for (children_cache.items[i]..children_cache.items[j]) |refined| {
                                var cluster: IndexBox = target_refined.patches.items(.bounds)[refined];

                                cluster.coarsen();
                                cluster.coarsen();

                                try clusters.append(cluster);
                                try cluster_index_map.append(refined);
                            }
                        }
                    }

                    cluster_offsets[1] = clusters.len;
                }

                // Clear cache for next cycle
                children_cache.clearRetainingCapacity();
                children_cache.ensureTotalCapacity(dest.gpa, ulen + 1);

                // Append first offset
                try children_cache.append(dest.gpa, 0);

                // 3.2 Generate new patches.
                // *************************

                // Per patch tile offset
                var tile_offset: usize = 0;

                for (0..ulen) |upid| {
                    // Reset arena for new "frame"
                    _ = arena.reset(.retain_capacity);

                    // Make aliases for various variables
                    const upbounds: IndexBox = ubounds[upid];
                    const upoffset: usize = uoffsets[upid];
                    const upspace: IndexSpace(N) = upbounds.space();
                    const uptags: []const usize = utags[upoffset..(upoffset + upspace.total())];

                    // Scratch space for partitioning algorithm
                    var partitions: Partitions(N) = .{};
                    defer partitions.deinit(scratch);

                    // Run partitioning algorithm
                    try partitions.compute(
                        scratch,
                        .{
                            .size = upbounds.size,
                            .tags = uptags,
                            .clusters = clusters.items[cluster_offsets[upid]..cluster_offsets[upid + 1]],
                        },
                        config.patch_max_tiles,
                        config.patch_effciency,
                    );

                    const patch_children_offset: usize = target.children.len;

                    // Append mapped children indices to buffer.
                    for (partitions.children.items) |index| {
                        try target.children.append(dest.gpa, cluster_index_map.items[index]);
                    }

                    // Loop through new patches
                    for (partitions.blocks.items(.bounds), partitions.blocks.items(.children_offset), partitions.blocks.items(.children_count)) |bounds, offset, count| {
                        // Compute refined bounds in global space.
                        var rbounds: IndexBox = undefined;

                        for (0..N) |axis| {
                            rbounds.origin[axis] = upbounds.origin[axis] + bounds.origin[axis];
                            rbounds.size[axis] = bounds.size[axis];
                        }

                        rbounds.refine();

                        // Append to patch list
                        try target.patches.append(dest.gpa, Patch{
                            .bounds = rbounds,
                            .parent = upid,
                            .children_offset = patch_children_offset + offset,
                            .children_count = count,
                            .tile_offset = tile_offset,
                        });

                        // Increment tile offset
                        tile_offset += rbounds.space().total();
                    }

                    try children_cache.append(dest.gpa, target.patches.len);
                }

                target.tile_total = tile_offset;

                // If refined exists, update parents of refined patches
                if (refined_exists) {
                    var target_refined: *Level = dest.levels.items[level_id + 1];

                    // Loop through
                    for (target.patches.items(.children_offset), target.patches.items(.children_count), 0..) |offset, count, i| {
                        for (target.children[offset..(offset + count)]) |child| {
                            target_refined.patches.items(.parent)[child] = i;
                        }
                    }
                }

                // 3.3 Generate new blocks by partitioning new patches.
                // ****************************************************

                // Level wide offset into cells array.
                var cell_offset: usize = 0;

                // Loop through newly created patches
                for (target.patches.items(.bounds), target.patches.items(.parent), 0..) |bounds, parent, patch| {
                    // Reset arena for new frame.
                    _ = arena.reset(.retain_capacity);

                    // Get coarse bounds and space.
                    var cbounds: IndexBox = bounds;
                    cbounds.coarsen();

                    const cspace: IndexSpace(N) = cbounds.space();

                    // Aliases for underlying variables
                    const upbounds: IndexBox = ubounds[parent];
                    const upoffset: usize = uoffsets[parent];
                    const upspace: IndexSpace(N) = upbounds.space();
                    const uptags: []const usize = utags[upoffset..(upoffset + upspace.total())];

                    // Build patch tags
                    var ptags: []const usize = scratch.alloc(bool, cspace.total());

                    // Set ptags using window of uptags
                    var cindices = cspace.cartesianIndices();

                    var i: usize = 0;

                    while (cindices.next()) |clocal| {
                        var global: [N]usize = cbounds.globalFromLocal(clocal);

                        for (0..N) |axis| {
                            global[axis] -= upbounds.origin[axis];
                        }

                        ptags[i] = uptags[upspace.linearFromCartesian(global)];

                        i += 1;
                    }

                    // Scratch space for partitioning algorithm
                    var partitions: Partitions(N) = .{};
                    defer partitions.deinit(scratch);

                    // Run partitioning algorithm
                    try partitions.compute(
                        scratch,
                        .{
                            .size = cbounds.size,
                            .tags = ptags,
                            .clusters = &[_]IndexBox{},
                        },
                        config.block_max_tiles,
                        config.block_effciency,
                    );

                    // For each resulting block
                    for (partitions.blocks.items(.bounds)) |pbounds| {
                        // Compute refined global bounds
                        var rbounds: IndexBox = undefined;

                        for (0..N) |axis| {
                            rbounds.origin[axis] = cbounds.origin[axis] + pbounds.origin[axis];
                            rbounds.size[axis] = pbounds.size[axis];
                        }

                        rbounds.refine();

                        // Add to list of blocks with appropriate patch id.
                        try target.blocks.append(dest.gpa, Block{
                            .bounds = rbounds,
                            .patch = patch,
                            .cell_offset = cell_offset,
                        });

                        // Inc cell offset with total ghost nodes.
                        cell_offset += rbounds.space().totalWithGhost(dest.config.tile_width, dest.config.ghost_width);
                    }
                }

                // Set level total cells
                target.cell_total = cell_offset;
            }

            // 4. Recompute level offsets and totals.
            // **************************************

            var tile_offset: usize = 0;
            var cell_offset: usize = 0;

            for (dest.levels.items) |*level| {
                level.tile_offset = tile_offset;
                level.cell_offset = cell_offset;

                tile_offset += level.tile_total;
                cell_offset += level.cell_total;
            }

            dest.total_cells = cell_offset;
            dest.total_tiles = tile_offset;
        }

        pub fn transfer(self: Self, map: ArrayList(TileMap), src: Self, new: *ArrayList(f64), old: ArrayList(f64)) !void {
            _ = old;
            _ = new;
            _ = src;
            _ = map;
            _ = self;
        }
    };
}
