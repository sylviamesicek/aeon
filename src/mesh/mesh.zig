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
        /// Allocator used for various arraylists stored in this struct.
        gpa: Allocator,
        /// The physical bounds of the mesh
        physical_bounds: RealBox,
        /// The index space of the mesh before performing `global_refinement`
        index_space: IndexSpace(N),
        global_refinement: usize,
        /// Number of cells per tile edge
        tile_width: usize,
        /// Number of ghost cells along boundary of domain
        ghost_width: usize,
        /// Total number of tiles in mesh
        tile_total: usize,
        /// Total number of cells in mesh
        cell_total: usize,
        /// Base level (after performing global_refinement).
        base: Base,
        /// Refined levels
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

            pub fn baseTileSpace(self: Config) IndexSpace(N) {
                var scale: usize = 1;

                for (0..self.global_refinement) |_| {
                    scale *= 2;
                }

                return IndexSpace(N).fromSize(self.index_size).scale(scale);
            }

            pub fn baseCellSpace(self: Config) IndexSpace(N) {
                return self.baseTileSpace().scale(self.tile_width);
            }

            pub fn check(self: Config) void {
                assert(self.tile_width >= 1);
                assert(self.tile_width >= self.ghost_width);
                for (0..N) |i| {
                    assert(self.index_size[i] > 0);
                    assert(self.physical_bounds.size[i] > 0.0);
                }
            }
        };

        pub const Base = struct {
            index_space: IndexSpace(N),
            tile_total: usize,
            cell_total: usize,
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
            index_space: IndexSpace(N),
            /// The blocks belonging to this level.
            blocks: MultiArrayList(Block),
            /// The patches belonging to this level.
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
                    .index_space = .{
                        .size = [1]usize{1} ** N,
                    },
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
                self.children.deinit(allocator);
            }

            fn childrenSlice(self: *Level, patch: usize) []usize {
                const offset = self.patches.items(.children_offset)[patch];
                const count = self.patches.items(.children_count)[patch];
                return self.children[offset..(offset + count)];
            }
        };

        pub fn init(allocator: Allocator, config: Config) Self {
            // Check config
            config.check();
            // Scale initial size by 2^global_refinement
            const tile_space: IndexSpace(N) = config.baseTileSpace();
            const cell_space: IndexSpace(N) = config.baseCellSpace();

            const base: Base = .{
                .index_space = tile_space,
                .tile_total = tile_space.total(),
                .cell_total = cell_space.total(),
            };

            return .{
                .gpa = allocator,
                .physical_bounds = config.physical_bounds,
                .index_space = IndexSpace(N).fromSize(config.index_size),
                .global_refinement = config.global_refinement,
                .tile_width = config.tile_width,
                .ghost_width = config.ghost_width,
                .tile_total = base.tile_total,
                .cell_total = base.cell_total,
                .base = base,
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
            return self.base.tile_total;
        }

        pub fn baseCellTotal(self: *const Self) usize {
            return self.base.cell_total;
        }

        pub const RegridConfig = struct {
            max_levels: usize,
            block_max_tiles: usize,
            block_efficiency: f64,
            patch_max_tiles: usize,
            patch_efficiency: f64,
        };

        pub fn regrid(dest: *Self, src: *const Self, tags: []bool, config: RegridConfig) !void {
            // 1. Copy config, and clear map.
            // ********************************************
            dest.physical_bounds = src.physical_bounds;
            dest.index_space = src.index_space;
            dest.tile_total = src.tile_total;
            dest.cell_total = src.cell_total;
            dest.global_refinement = src.global_refinement;
            dest.base = src.base;

            // 2. Find total number of levels and preallocate dest.
            // **********************************************************
            const total_levels = count_levels: {
                // The max number of levels can not truncate the grid.
                assert(config.max_levels >= src.active_levels);

                // Clamp to max levels
                if (src.active_levels == config.max_levels) {
                    break :count_levels config.max_levels;
                }

                // Check if any on the highest level is tagged
                if (src.active_levels > 0) {
                    for (tags[src.levels.getLast().tile_offset..]) |tag| {
                        if (tag) {
                            break :count_levels src.active_levels + 1;
                        }
                    }

                    break :count_levels src.active_levels;
                } else {
                    for (tags) |tag| {
                        if (tag) {
                            break :count_levels src.active_levels + 1;
                        }
                    }

                    break :count_levels src.active_levels;
                }
            };

            // While necessary allocate additional levels, until dest reaches total levels.
            while (total_levels > dest.levels.items.len) {
                try dest.levels.append(dest.gpa, Level.init());
            }

            if (total_levels > 0) {
                dest.levels.items[0].index_space = dest.base.index_space.scale(2);
            }

            // Update level index spaces
            for (1..total_levels) |i| {
                dest.levels.items[i].index_space = dest.levels.items[i - 1].index_space.scale(2);
            }

            // 3. Recursively generate levels on new mesh.
            // *******************************************

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(dest.gpa);
            defer arena.deinit();

            var scratch: Allocator = arena.allocator();

            // Build a cache for children indices of underlying level.
            var children_cache: ArrayListUnmanaged(usize) = .{};
            defer children_cache.deinit(dest.gpa);

            // Bounds for base level.
            const bbounds: IndexBox = .{
                .origin = [1]usize{0} ** N,
                .size = dest.base.index_space.size,
            };

            // Slices at top of scope ensures we don't reference a temporary.
            const bbounds_slice: []const IndexBox = &[_]IndexBox{bbounds};
            const boffsets_slice: []const usize = &[_]usize{0};

            // Loop through levels from highest to lowest
            for (0..total_levels) |reverse_level_id| {
                const level_id: usize = total_levels - 1 - reverse_level_id;

                // Get a mutable reference to the target level.
                var target: *Level = &dest.levels.items[level_id];

                // Now we can clear the target arrays
                target.blocks.shrinkRetainingCapacity(0);
                target.patches.shrinkRetainingCapacity(0);
                target.children.shrinkRetainingCapacity(0);

                // Check if there exists a level higher than the current one.
                const refined_exists: bool = level_id < total_levels - 1;
                // Check if we are over base level.
                const coarse_exists: bool = level_id > 0;

                // Variables that depend on coarse existing
                var utags: []bool = undefined;
                var ulen: usize = undefined;
                var ubounds: []const IndexBox = undefined;
                var uoffsets: []const usize = undefined;

                // 3.1 Build clusters on refined mesh to consider when repartitioning target.
                // **************************************************************************

                // Stores offsets per underlying patch into clusters array
                var cluster_offsets: []usize = &[_]usize{};
                defer dest.gpa.free(cluster_offsets);

                // Stores bounds of clusters
                var clusters: ArrayListUnmanaged(IndexBox) = .{};
                defer clusters.deinit(dest.gpa);

                // Stores map from cluster bounds to global index in l+1
                var cluster_index_map: ArrayListUnmanaged(usize) = .{};
                defer cluster_index_map.deinit(dest.gpa);

                if (coarse_exists) {
                    // Get underlying data
                    const underlying: *const Level = &src.levels.items[level_id - 1];

                    utags = tags[underlying.tile_offset..(underlying.tile_offset + underlying.tile_total)];
                    ulen = underlying.patches.len;
                    ubounds = underlying.patches.items(.bounds);
                    uoffsets = underlying.patches.items(.tile_offset);

                    cluster_offsets = try dest.gpa.alloc(usize, ulen + 1);
                    cluster_offsets[0] = 0;

                    // Update clusters
                    for (0..ulen) |underlying_id| {
                        if (refined_exists) {
                            const target_refined: *const Level = &dest.levels.items[level_id + 1];

                            const offset: usize = underlying.patches.items(.children_offset)[underlying_id];
                            const count: usize = underlying.patches.items(.children_count)[underlying_id];

                            for (underlying.children.items[offset..(offset + count)]) |child| {
                                for (children_cache.items[child]..children_cache.items[child + 1]) |refined| {
                                    var cluster: IndexBox = target_refined.patches.items(.bounds)[refined];

                                    try cluster.coarsen();
                                    try cluster.coarsen();

                                    try clusters.append(dest.gpa, cluster);
                                    try cluster_index_map.append(dest.gpa, refined);
                                }
                            }
                        }

                        cluster_offsets[underlying_id + 1] = clusters.items.len;
                    }
                } else {
                    // Otherwise use base data
                    utags = tags[0..dest.baseTileTotal()];
                    ulen = 1;
                    ubounds = bbounds_slice;
                    uoffsets = boffsets_slice;

                    cluster_offsets = try dest.gpa.alloc(usize, ulen + 1);
                    cluster_offsets[0] = 0;

                    // All registered patches are children of base
                    for (0..children_cache.items.len, 1..) |i, j| {
                        if (refined_exists) {
                            const target_refined: *const Level = &dest.levels.items[level_id + 1];

                            for (children_cache.items[i]..children_cache.items[j]) |refined| {
                                var cluster: IndexBox = target_refined.patches.items(.bounds)[refined];

                                try cluster.coarsen();
                                try cluster.coarsen();

                                try clusters.append(dest.gpa, cluster);
                                try cluster_index_map.append(dest.gpa, refined);
                            }
                        }
                    }

                    cluster_offsets[1] = clusters.items.len;
                }

                // Clear cache for next cycle
                children_cache.clearRetainingCapacity();
                try children_cache.ensureTotalCapacity(dest.gpa, ulen + 1);

                // Append first offset
                try children_cache.append(dest.gpa, 0);

                // 3.3 Generate new patches.
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
                    const uptags: []bool = utags[upoffset..(upoffset + upspace.total())];
                    const upclusters: []const IndexBox = clusters.items[cluster_offsets[upid]..cluster_offsets[upid + 1]];
                    const upcluster_index_map: []const usize = cluster_index_map.items[cluster_offsets[upid]..cluster_offsets[upid + 1]];

                    // Preprocess tags to include all elements from clusters
                    for (upclusters) |cluster| {
                        upbounds.space().fillSubspace(cluster, bool, uptags, true);
                    }

                    // Scratch space for partitioning algorithm
                    var partitions: Partitions(N) = .{};
                    defer partitions.deinit(scratch);

                    // Run partitioning algorithm
                    try partitions.compute(
                        scratch,
                        .{
                            .size = upbounds.size,
                            .tags = uptags,
                            .clusters = upclusters,
                        },
                        config.patch_max_tiles,
                        config.patch_efficiency,
                    );

                    const patch_children_offset: usize = target.children.items.len;

                    // Append mapped children indices to buffer.
                    for (partitions.children.items) |index| {
                        try target.children.append(dest.gpa, upcluster_index_map[index]);
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
                    var target_refined: *Level = &dest.levels.items[level_id + 1];

                    // Loop through
                    for (target.patches.items(.children_offset), target.patches.items(.children_count), 0..) |offset, count, i| {
                        for (target.children.items[offset..(offset + count)]) |child| {
                            target_refined.patches.items(.parent)[child] = i;
                        }
                    }
                }

                // 3.4 Generate new blocks by partitioning new patches.
                // ****************************************************

                // Level wide offset into cells array.
                var cell_offset: usize = 0;

                // Loop through newly created patches
                for (target.patches.items(.bounds), target.patches.items(.parent), 0..) |bounds, parent, patch| {
                    // Reset arena for new frame.
                    _ = arena.reset(.retain_capacity);

                    // Get coarse bounds and space.
                    var cbounds: IndexBox = bounds;
                    try cbounds.coarsen();

                    const cspace: IndexSpace(N) = cbounds.space();

                    // Aliases for underlying variables
                    const upbounds: IndexBox = ubounds[parent];
                    const upoffset: usize = uoffsets[parent];
                    const upspace: IndexSpace(N) = upbounds.space();
                    const uptags: []const bool = utags[upoffset..(upoffset + upspace.total())];

                    // Build patch tags
                    var ptags: []bool = try scratch.alloc(bool, cspace.total());
                    defer scratch.free(ptags);
                    // Set ptags using window of uptags
                    upspace.fillWindow(cbounds, bool, ptags, uptags);

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
                        config.block_efficiency,
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
                        cell_offset += rbounds.space().scale(dest.tile_width).total();
                    }
                }

                // Set level total cells
                target.cell_total = cell_offset;
            }

            // 4. Recompute level offsets and totals.
            // **************************************

            var tile_offset: usize = dest.base.tile_total;
            var cell_offset: usize = dest.base.cell_total;

            for (dest.levels.items) |*level| {
                level.tile_offset = tile_offset;
                level.cell_offset = cell_offset;

                tile_offset += level.tile_total;
                cell_offset += level.cell_total;
            }

            dest.cell_total = cell_offset;
            dest.tile_total = tile_offset;
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

test "mesh regridding" {
    const expect = std.testing.expect;
    _ = expect;
    const expectEqualSlices = std.testing.expectEqualSlices;
    _ = expectEqualSlices;

    const allocator = std.testing.allocator;

    const Mesh2 = Mesh(2);

    const config: Mesh2.Config = .{
        .physical_bounds = .{
            .origin = [_]f64{ 0.0, 0.0 },
            .size = [_]f64{ 1.0, 1.0 },
        },
        .index_size = [_]usize{ 10, 10 },
        .ghost_width = 0,
        .tile_width = 16,
        .global_refinement = 2,
    };

    var mesh: Mesh2 = Mesh2.init(allocator, config);
    defer mesh.deinit();

    var tags: []bool = try allocator.alloc(bool, mesh.tile_total);
    defer allocator.free(tags);

    // Tag all
    @memset(tags, true);

    var dest: Mesh2 = Mesh2.init(allocator, config);
    defer dest.deinit();

    try dest.regrid(&mesh, tags, .{
        .max_levels = 1,
        .block_max_tiles = 10,
        .block_efficiency = 0.7,
        .patch_max_tiles = 10,
        .patch_efficiency = 0.1,
    });
}
