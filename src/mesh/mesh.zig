const std = @import("std");
const meta = std.meta;

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const ArenaAllocator = std.heap.ArenaAllocator;

const assert = std.debug.assert;
const exp2 = std.math.exp2;

const geometry = @import("../geometry/geometry.zig");

const Box = geometry.Box;
const IndexSpace = geometry.IndexSpace;
const PartitionSpace = geometry.PartitionSpace;

const transfer = @import("transfer.zig");

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
        /// The index size of the mesh before performing `global_refinement`
        index_size: [N]usize,
        /// The number of levels of global refinement to apply to get base level.
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
        /// Number of levels which are active.
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
                return self.baseTileSpace().scale(self.tile_width).extendUniform(2 * self.ghost_width);
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
            index_size: [N]usize,
            tile_total: usize,
            cell_total: usize,
        };

        pub const Block = struct {
            bounds: IndexBox,
            patch: usize,
            cell_total: usize = 0,
            cell_offset: usize = 0,
            boundary: [2 * N]bool = [1]bool{false} ** 2 ** N,
        };

        pub const Patch = struct {
            /// Bounds of this patch in level index space.
            bounds: IndexBox,
            /// Children of this patch
            children_offset: usize,
            children_total: usize,
            /// The total number of tiles on this patch.
            tile_total: usize = 0,
            /// The offset into a level-wide array of tiles.
            tile_offset: usize = 0,
        };

        pub const Level = struct {
            index_size: IndexSpace(N),
            /// The blocks belonging to this level.
            blocks: MultiArrayList(Block),
            /// The patches belonging to this level.
            patches: MultiArrayList(Patch),
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
            }

            /// Gets a slice of children indices for each patch
            fn childrenSlice(self: *Level, patch: usize) []usize {
                const offset = self.patches.items(.children_offset)[patch];
                const count = self.patches.items(.children_count)[patch];
                return self.children[offset..(offset + count)];
            }

            /// Computes patch and block offsets and level totals for tiles and cells.
            fn computeOffsets(self: *Level, tile_width: usize, ghost_width: usize) void {
                var tile_offset: usize = 0;

                for (self.patches.items(.bounds), self.patches.items(.tile_total), self.patches.items(.tile_offset)) |bounds, *total, *offset| {
                    offset.* = tile_offset;
                    total.* = bounds.space().total();
                    tile_offset += total.*;
                }

                var cell_offset: usize = 0;

                for (self.blocks.items(.bounds), self.blocks.items(.cell_total), self.blocks.items(.cell_offset)) |bounds, *total, *offset| {
                    offset.* = cell_offset;
                    total.* = bounds.space().scale(tile_width).extendUniform(2 * ghost_width).total();
                    cell_offset += total.*;
                }

                self.tile_total = tile_offset;
                self.cell_total = cell_offset;
            }

            /// Refines every patch and block on this level.
            fn refine(self: *Level) void {
                for (self.patches.items(.bounds)) |*bounds| {
                    bounds.refine();
                }

                for (self.blocks.items(.bounds)) |*bounds| {
                    bounds.refine();
                }
            }
        };

        pub fn init(allocator: Allocator, config: Config) Self {
            // Check config
            config.check();
            // Scale initial size by 2^global_refinement
            const tile_space: IndexSpace(N) = config.baseTileSpace();
            const cell_space: IndexSpace(N) = config.baseCellSpace();

            const base: Base = .{
                .index_size = tile_space.size,
                .tile_total = tile_space.total(),
                .cell_total = cell_space.total(),
            };

            return .{
                .gpa = allocator,
                .physical_bounds = config.physical_bounds,
                .index_size = config.index_size,
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

        // **********************************
        // Helpers for querying totals ******
        // **********************************

        pub fn tileTotal(self: *const Self) usize {
            return self.tile_total;
        }

        pub fn cellTotal(self: *const Self) usize {
            return self.cell_total;
        }

        pub fn baseTileTotal(self: *const Self) usize {
            return self.base.tile_total;
        }

        pub fn baseCellTotal(self: *const Self) usize {
            return self.base.cell_total;
        }

        pub fn levelTileTotal(self: *const Self, level: usize) usize {
            return self.levels.items[level].tile_total;
        }

        pub fn levelCellTotal(self: *const Self, level: usize) usize {
            return self.levels.items[level].tile_total;
        }

        pub fn baseTileSlice(self: *const Self, comptime T: type, slice: []T) []T {
            return slice[0..self.baseTileTotal()];
        }

        pub fn baseCellSlice(self: *const Self, comptime T: type, slice: []T) []T {
            return slice[0..self.baseCellTotal()];
        }

        pub fn levelTileSlice(self: *const Self, level: usize, comptime T: type, slice: []T) []T {
            const l: *const Level = self.levels.items[level];
            return slice[l.tile_offset..(l.tile_offset + l.tile_total)];
        }

        pub fn levelCellSlice(self: *const Self, level: usize, comptime T: type, slice: []T) []T {
            const l: *const Level = self.levels.items[level];
            return slice[l.tile_offset..(l.cell_offset + l.cell_total)];
        }

        // *************************
        // Block Map ***************
        // *************************

        pub fn buildBlockMap(self: *const Self, map: []usize) !void {
            assert(map.len == self.tileTotal());

            @memset(map, std.math.maxInt(usize));
            @memset(self.baseTileSlice(usize, map), 0);

            for (self.levels.items, 0..) |*level, l| {
                const level_map: []usize = self.levelTileSlice(l, usize, map);

                for (level.blocks.items(.bounds), level.blocks.items(.patch), 0..) |bounds, parent, id| {
                    const pbounds: IndexBox = level.patches.items(.bounds)[parent];
                    const tile_offset: usize = level.patches.items(.tile_offset)[parent];
                    const tile_total: usize = level.patches.items(.tile_total)[parent];
                    const tile_to_block: []usize = level_map[tile_offset..(tile_offset + tile_total)];

                    pbounds.space().fillSubspace(bounds.relativeTo(pbounds), usize, tile_to_block, id);
                }
            }
        }

        // *************************
        // Regridding **************
        // *************************

        pub const RegridConfig = struct {
            max_levels: usize,
            block_max_tiles: usize,
            block_efficiency: f64,
            patch_max_tiles: usize,
            patch_efficiency: f64,
        };

        pub fn regrid(self: *Self, tags: []bool, config: RegridConfig) !void {
            assert(config.max_levels >= self.active_levels);

            // 1. Find total number of levels and preallocate dest.
            // **********************************************************
            const total_levels = self.computeTotalLevels(tags, config);
            self.active_levels = total_levels;
            self.cacheLevels(total_levels);

            // 2. Recursively generate levels on new mesh.
            // *******************************************

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(self.gpa);
            defer arena.deinit();

            var scratch: Allocator = arena.allocator();

            // Build a cache for children indices of underlying level.
            var children_cache: ArrayListUnmanaged(usize) = .{};
            defer children_cache.deinit(self.gpa);

            // Bounds for base level.
            const bbounds: IndexBox = .{
                .origin = [1]usize{0} ** N,
                .size = self.base.index_size,
            };

            // Slices at top of scope ensures we don't reference a temporary.
            const bbounds_slice: []const IndexBox = &[_]IndexBox{bbounds};
            const boffsets_slice: []const usize = &[_]usize{0};

            var clusters: ArrayListUnmanaged(IndexBox) = .{};
            defer clusters.deinit(self.gpa);

            var cluster_index_map: ArrayListUnmanaged(usize) = .{};
            defer cluster_index_map.deinit(self.gpa);

            var cluster_offsets: ArrayListUnmanaged(usize) = .{};
            defer cluster_offsets.deinit(self.gpa);

            var coarse_offsets: ArrayListUnmanaged(usize) = .{};
            defer coarse_offsets.deinit(self.gpa);

            // Loop through levels from highest to lowest
            for (0..total_levels) |reverse_level_id| {
                const level_id: usize = total_levels - 1 - reverse_level_id;

                // Get a mutable reference to the target level.
                const target: *Level = &self.levels.items[level_id];

                // Check if there exists a level higher than the current one.
                const refined_exists: bool = level_id < total_levels - 1;
                // Check if we are over base level.
                const coarse_exists: bool = level_id > 0;

                // At this moment in time
                // - coarse is old
                // - target is old (but children and parents have been updated)
                // - refined has been fully updated

                // To assemble clusters per patch we iterate children of coarse, then children of target

                clusters.shrinkRetainingCapacity(0);
                cluster_index_map.shrinkRetainingCapacity(0);
                cluster_offsets.shrinkRetainingCapacity(0);

                try cluster_offsets.append(self.gpa, 0);

                if (refined_exists and coarse_exists) {
                    const refined: *const Level = &self.levels.items[level_id + 1];
                    const coarse: *const Level = &self.levels.items[level_id - 1];

                    for (0..coarse.patches.len) |cpid| {
                        for (coarse.childrenSlice(cpid)) |tpid| {
                            for (target.childrenSlice(tpid)) |child| {
                                var patch: IndexBox = refined.patches.items(.bounds)[child];

                                patch.coarsen();
                                patch.coarsen();

                                try clusters.append(self.gpa, patch);
                                try cluster_index_map.append(self.gpa, child);
                            }
                        }

                        try cluster_offsets.append(self.gpa, clusters.len);
                    }
                } else if (refined_exists) {
                    const refined: *const Level = &self.levels.items[level_id + 1];

                    for (0..target.patches.len) |tpid| {
                        for (target.childrenSlice(tpid)) |child| {
                            var patch: IndexBox = refined.patches.items(.bounds)[child];

                            patch.coarsen();
                            patch.coarsen();

                            try clusters.append(self.gpa, patch);
                            try cluster_index_map.append(self.gpa, child);
                        }
                    }

                    try cluster_offsets.append(self.gpa, clusters.len);
                }

                // Now we can clear the target arrays
                target.patches.shrinkRetainingCapacity(0);
                target.blocks.shrinkRetainingCapacity(0);
                target.children.shrinkRetainingCapacity(0);
                target.parents.shrinkRetainingCapacity(0);

                // At this moment in time
                // - coarse is old
                // - target is cleared
                // - refined has been fully updated

                // Variables that depend on coarse existing
                var utags: []bool = undefined;
                var ulen: usize = undefined;
                var ubounds: []const IndexBox = undefined;
                var uoffsets: []const usize = undefined;

                if (coarse_exists) {
                    // Get underlying data
                    const underlying: *const Level = &self.levels.items[level_id - 1];

                    utags = self.levelTileSlice(level_id - 1, bool, tags);
                    ulen = underlying.patches.len;
                    ubounds = underlying.patches.items(.bounds);
                    uoffsets = underlying.patches.items(.tile_offset);
                } else {
                    // Otherwise use base data
                    utags = self.baseTileSlice(bool, tags);
                    ulen = 1;
                    ubounds = bbounds_slice;
                    uoffsets = boffsets_slice;
                }

                // 3.3 Generate new patches.
                // *************************

                coarse_offsets.shrinkRetainingCapacity(0);

                try coarse_offsets.append(self.gpa, 0);

                for (0..ulen) |upid| {
                    // Reset arena for new "frame"
                    _ = arena.reset(.retain_capacity);

                    // Make aliases for patch variables
                    const upbounds: IndexBox = ubounds[upid];
                    const upoffset: usize = uoffsets[upid];
                    const upspace: IndexSpace(N) = upbounds.space();
                    const uptags: []bool = utags[upoffset..(upoffset + upspace.total())];

                    // As well as clusters in this patch
                    const upclusters: []const IndexBox = clusters.items[cluster_offsets[upid]..cluster_offsets[upid + 1]];
                    const upcluster_index_map: []const usize = cluster_index_map.items[cluster_offsets[upid]..cluster_offsets[upid + 1]];

                    // Preprocess tags to include all elements from clusters (and one tile buffer region around cluster)
                    preprocessTagsOnPatch(uptags, upbounds, upclusters);

                    // Run partitioning algorithm
                    var partition_space = PartitionSpace(N).init(scratch, upbounds.size, upclusters);
                    defer partition_space.deinit();

                    try partition_space.build(uptags, config.patch_max_tiles, config.patch_efficiency);

                    // Add patches to target
                    const patch_children_offset: usize = target.children.items.len;

                    for (partition_space.parts.items(.bounds), partition_space.parts.items(.children_offset), partition_space.parts.items(.children_total)) |bounds, offset, total| {
                        // Compute refined bounds in global space.
                        var rbounds: IndexBox = undefined;

                        for (0..N) |axis| {
                            rbounds.origin[axis] = upbounds.origin[axis] + bounds.origin[axis];
                            rbounds.size[axis] = bounds.size[axis];
                        }

                        // Append to patch list
                        try target.patches.append(self.gpa, Patch{
                            .bounds = rbounds,
                            .children_offset = patch_children_offset + offset,
                            .children_total = total,
                        });
                    }

                    // Append mapped children indices to buffer.
                    for (partition_space.children.items) |index| {
                        try target.children.append(self.gpa, upcluster_index_map[index]);
                    }

                    // Set parents to corret values
                    for (partition_space.parts.items(.children_offset), partition_space.parts.items(.children_total), 0..) |offset, total, i| {
                        for (target.children.items[patch_children_offset..][offset..(offset + total)]) |child| {
                            target.parents.items[child] = i;
                        }
                    }

                    try coarse_offsets.append(self.gpa, target.patches.len);
                }

                if (coarse_exists) {
                    const coarse: *const Level = &self.levels.items[level_id - 1];

                    try coarse.children.resize(self.gpa, target.patches.len);
                    try coarse.parents.resize(self.gpa, target.patches.len);

                    for (0..target.patches.len) |i| {
                        coarse.children[i] = i;
                    }

                    for (0..ulen) |cpid| {
                        const start = coarse_offsets[cpid];
                        const end = coarse_offsets[cpid + 1];

                        coarse.patches.items(.children_offset)[cpid] = start;
                        coarse.patches.items(.children_total)[cpid] = end - start;

                        @memset(coarse.parents.items[start..end], cpid);
                    }
                }

                // At this moment in time
                // - coarse is old (but heirarchy is updated)
                // - target is updated (but blocks are still empty and patches are still coarse)
                // - refined has been fully updated

                // 3.4 Generate new blocks by partitioning new patches.
                // ****************************************************

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
                        });
                    }
                }
            }

            // 4. Recompute level offsets and totals.
            // **************************************

            dest.computeOffsets();
            try dest.computeBoundaries();
        }

        pub fn transfer(self: Self, map: ArrayList(TileMap), src: Self, new: *ArrayList(f64), old: ArrayList(f64)) !void {
            _ = old;
            _ = new;
            _ = src;
            _ = map;
            _ = self;
        }

        fn preprocessTagsOnPatch(tags: []bool, bounds: IndexBox, clusters: []const IndexBox) void {
            for (clusters) |upcluster| {
                var cluster: Box(N, usize) = upcluster;

                for (0..N) |i| {
                    if (cluster.origin[i] > 0) {
                        cluster.origin[i] -= 1;
                        cluster.size[i] += 1;
                    }

                    if (cluster.origin[i] + cluster.size[i] < bounds.size[i]) {
                        cluster.size[i] += 1;
                    }
                }

                bounds.space().fillSubspace(cluster, bool, tags, true);
            }
        }

        fn computeOffsets(self: *Self) void {
            var tile_offset: usize = self.base.tile_total;
            var cell_offset: usize = self.base.cell_total;

            for (self.levels.items) |*level| {
                level.computeOffsets(self.tile_width, self.ghost_width);

                level.tile_offset = tile_offset;
                level.cell_offset = cell_offset;

                tile_offset += level.tile_total;
                cell_offset += level.cell_total;
            }

            self.cell_total = cell_offset;
            self.tile_total = tile_offset;
        }

        fn computeTotalLevels(self: *const Self, tags: []const usize, config: RegridConfig) usize {
            // Clamp to max levels
            if (self.active_levels == config.max_levels) {
                return config.max_levels;
            }

            // Check if any on the highest level is tagged
            if (self.active_levels > 0) {
                for (tags[self.levels.getLast().tile_offset..]) |tag| {
                    if (tag) {
                        return self.active_levels + 1;
                    }
                }

                return self.active_levels;
            } else {
                for (tags) |tag| {
                    if (tag) {
                        return self.active_levels + 1;
                    }
                }

                return self.active_levels;
            }
        }

        fn cacheLevels(self: *Self, total: usize) !void {
            while (total > self.levels.items.len) {
                if (self.levels.items.len == 0) {
                    const size: [N]usize = IndexSpace(N).fromSize(self.base.index_size).scale(2).size;

                    try self.levels.append(self.gpa, Level.init(size));
                } else {
                    const size: [N]usize = IndexSpace(N).fromSize(self.levels.getLast().index_size).scale(2).size;

                    try self.levels.append(self.gpa, Level.init(size));
                }
            }
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
