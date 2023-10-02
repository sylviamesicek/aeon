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

const levels = @import("levels.zig");

// ************************
// Mesh *******************
// ************************

/// Represents a mesh, ie a numerical discretisation of a physical domain into a number of levels,
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
        /// Refined levels
        levels: ArrayListUnmanaged(Level),
        /// Number of levels which are currently active.
        active_levels: usize,
        /// Total number of tiles in mesh
        tile_total: usize,
        /// Total number of cells in mesh
        cell_total: usize,

        // Public types
        pub const Level = levels.Level(N);
        pub const Block = levels.Block(N);
        pub const Patch = levels.Patch(N);

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
        const PartitionSpace = geometry.PartitionSpace(N);
        const Region = geometry.Region(N);

        // Mixins
        const Index = @import("../index.zig").Index(N);

        const add = Index.add;
        const sub = Index.sub;
        const scaled = Index.scaled;
        const splat = Index.splat;
        const toSigned = Index.toSigned;

        /// Initialises a new mesh with an general purpose allocator, subject to the given configuration.
        pub fn init(allocator: Allocator, config: Config) !Self {
            // Check config
            config.check();

            var level: Level = Level.init(config.index_size);
            errdefer level.deinit(allocator);

            try level.buildBase(allocator, config.index_size, 0);
            level.computeOffsets(config.tile_width);

            var levs: ArrayListUnmanaged(Level) = .{};
            errdefer levs.deinit(allocator);

            try levs.append(allocator, level);

            var self: Self = .{
                .gpa = allocator,
                .index_size = config.index_size,
                .physical_bounds = config.physical_bounds,
                .tile_width = config.tile_width,
                .levels = levs,
                .active_levels = 1,
                .tile_total = 0,
                .cell_total = 0,
            };

            self.computeOffsets();

            return self;
        }

        /// Deinitalises a mesh.
        pub fn deinit(self: *Self) void {
            for (self.levels.items) |*level| {
                level.deinit(self.gpa);
            }

            self.levels.deinit(self.gpa);
        }

        // **********************************
        // Helpers **************************
        // **********************************

        /// Get an level of this mesh.
        pub fn getLevel(self: *const Self, level: usize) *const Level {
            assert(level < self.active_levels);
            return &self.levels.items[level];
        }

        pub fn patchTileOffset(self: *const Self, level: usize, patch: usize) usize {
            const target = self.getLevel(level);
            const patch_offset = target.patches.items(.tile_offset)[patch];

            return target.tile_offset + patch_offset;
        }

        pub fn patchTileTotal(self: *const Self, level: usize, patch: usize) usize {
            const target = self.getLevel(level);
            return target.patches.items(.tile_total)[patch];
        }

        pub fn blockCellOffset(self: *const Self, level: usize, block: usize) usize {
            const target = self.getLevel(level);
            const block_offset = target.blocks.items(.cell_offset)[block];

            return target.cell_offset + block_offset;
        }

        pub fn blockCellTotal(self: *const Self, level: usize, block: usize) usize {
            const target = self.getLevel(level);
            return target.blocks.items(.cell_total)[block];
        }

        pub fn patchTileSlice(self: *const Self, level: usize, patch: usize, mesh_slice: anytype) @TypeOf(mesh_slice) {
            const target = self.getLevel(level);
            const patch_offset = target.patches.items(.tile_offset)[patch];
            const patch_total = target.patches.items(.tile_total)[patch];
            const offset = target.tile_offset + patch_offset;

            return mesh_slice[offset..patch_total];
        }

        pub fn blockCellSlice(self: *const Self, level: usize, block: usize, mesh_slice: anytype) @TypeOf(mesh_slice) {
            const target = self.getLevel(level);
            const block_offset = target.blocks.items(.cell_offset)[block];
            const block_total = target.blocks.items(.cell_total)[block];
            const offset = target.cell_offset + block_offset;

            return mesh_slice[offset..block_total];
        }

        // *************************
        // Block Map ***************
        // *************************

        /// Builds a block map, ie a map from each tile in the mesh to the id of the block that tile is in.
        pub fn buildBlockMap(self: *const Self, map: []usize) void {
            assert(map.len == self.tile_total);

            @memset(map, maxInt(usize));

            for (self.levels.items[0..self.active_levels], 0..) |level, l| {
                for (level.blocks.items(.bounds), level.blocks.items(.patch), 0..) |bounds, patch, id| {
                    const pbounds: IndexBox = level.patches.items(.bounds)[patch];
                    const tile_to_block: []usize = self.patchTileSlice(l, patch, map);
                    IndexSpace.fromBox(pbounds).fillWindow(bounds.relativeTo(pbounds), usize, tile_to_block, id);
                }
            }
        }

        /// Builds a block child map. ie a map from each tile in the mesh to the id of the block that covers it
        /// on the next level, if any.
        pub fn buildBlockChildMap(self: *const Self, map: []usize) void {
            assert(map.len == self.tile_total);

            @memset(map, maxInt(usize));

            for (0..self.active_levels - 1) |l| {
                const level = self.getLevel(l);
                const refined = self.getLevel(l + 1);

                for (0..level.patchTotal()) |patch| {
                    const patch_bounds: IndexBox = level.patches.items(.bounds)[patch];
                    const patch_map: []usize = self.patchTileSlice(l, patch, map);
                    for (level.childrenSlice(patch)) |child_patch| {
                        const offset = refined.patches.items(.block_offset)[child_patch];
                        const total = refined.patches.items(.block_total)[child_patch];
                        for (offset..offset + total) |child_block| {
                            var bounds = refined.blocks.items(.bounds)[child_block];
                            bounds.coarsen();

                            IndexSpace.fromBox(patch_bounds).fillWindow(bounds.relativeTo(patch), usize, patch_map, child_block);
                        }
                    }
                }

                for (level.blocks.items(.bounds), level.blocks.items(.patch), 0..) |bounds, patch, id| {
                    const pbounds: IndexBox = level.patches.items(.bounds)[patch];
                    const tile_to_block: []usize = self.patchTileSlice(l, patch, map);
                    IndexSpace.fromBox(pbounds).fillWindow(bounds.relativeTo(pbounds), usize, tile_to_block, id);
                }
            }
        }

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
        pub fn blockPhysicalBounds(self: *const Self, level: usize, block: usize) RealBox {
            // Get bounds of this block
            const bounds = self.levels.items[level].blocks.items(.bounds)[block];

            const index_size: [N]usize = self.levels.items[level].index_size;

            var physical_bounds: RealBox = undefined;

            for (0..N) |i| {
                const sratio: f64 = @as(f64, @floatFromInt(bounds.size[i])) / @as(f64, @floatFromInt(index_size[i]));
                const oratio: f64 = @as(f64, @floatFromInt(bounds.origin[i])) / @as(f64, @floatFromInt(index_size[i] - 1));

                physical_bounds.size[i] = self.physical_bounds.size[i] * sratio;
                physical_bounds.origin[i] = self.physical_bounds.origin[i] + self.physical_bounds.size[i] * oratio;
            }

            return physical_bounds;
        }

        // *************************
        // Regridding **************
        // *************************

        /// Regenerates the current mesh using the set of tags to determine which tiles on each patch
        /// should be included in the new mesh.
        pub fn regrid(
            self: *Self,
            tags: []bool,
            max_levels: usize,
            patch_efficiency: f64,
            patch_max_tiles: usize,
            block_efficiency: f64,
            block_max_tiles: usize,
        ) !void {
            // Check regridding parameter
            assert(max_levels >= self.active_levels);
            assert(block_max_tiles > 0 and patch_max_tiles > 0 and patch_max_tiles >= block_max_tiles);
            assert(block_efficiency >= patch_efficiency);

            // 1. Find total number of levels and preallocate dest.
            // **********************************************************
            const total_levels = self.computeTotalLevels(tags, max_levels);
            try self.resizeActiveLevels(total_levels);

            // 2. Recursively generate levels on new mesh.
            // *******************************************

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(self.gpa);
            defer arena.deinit();

            var scratch: Allocator = arena.allocator();

            // Stores a set of clusters to consider when repartitioning l. This consists of all patches on l+1.
            var clusters: ArrayListUnmanaged(IndexBox) = .{};
            defer clusters.deinit(self.gpa);

            // A map from index in clusters to index of patch on l+1.
            var cluster_index_map: ArrayListUnmanaged(usize) = .{};
            defer cluster_index_map.deinit(self.gpa);

            // Allows mapping from coarse patch to clusters.
            var cluster_offsets: ArrayListUnmanaged(usize) = .{};
            defer cluster_offsets.deinit(self.gpa);

            // At the end of iterating level l, this contains offsets from coarse patches on l-1 to new patches on l.
            var coarse_children: ArrayListUnmanaged(usize) = .{};
            defer coarse_children.deinit(self.gpa);

            // Loop through levels from highest to lowest
            for (0..(total_levels - 1)) |reverse_level_id| {
                const level_id: usize = total_levels - 1 - reverse_level_id;
                // Get a mutable reference to the target level.
                const target: *Level = &self.levels.items[level_id];

                // Check if there exists a level higher than the current one.
                const refined_exists: bool = level_id < total_levels - 1;

                // At this moment in time
                // - coarse is old
                // - target is old
                // - refined has been fully updated

                // To assemble clusters per patch we iterate children of coarse, then children of target

                clusters.shrinkRetainingCapacity(0);
                cluster_index_map.shrinkRetainingCapacity(0);
                cluster_offsets.shrinkRetainingCapacity(0);

                try cluster_offsets.append(self.gpa, 0);

                const coarse: *const Level = &self.levels.items[level_id - 1];

                if (refined_exists) {
                    const refined: *const Level = &self.levels.items[level_id + 1];

                    for (coarse.patches.items(.bounds), 0..) |cpbounds, cpid| {
                        for (coarse.childrenSlice(cpid)) |tpid| {
                            const start = coarse_children.items[tpid];
                            const end = coarse_children.items[tpid + 1];

                            for (start..end) |child| {
                                var patch: IndexBox = refined.patches.items(.bounds)[child];

                                try patch.coarsen();
                                try patch.coarsen();

                                try clusters.append(self.gpa, patch.relativeTo(cpbounds));
                                try cluster_index_map.append(self.gpa, child);
                            }
                        }

                        try cluster_offsets.append(self.gpa, clusters.items.len);
                    }
                } else {
                    try cluster_offsets.append(self.gpa, clusters.items.len);
                }

                try target.setTotalChildren(self.gpa, clusters.items.len);
                target.clearRetainingCapacity();

                // At this moment in time
                // - coarse is old
                // - target is cleared
                // - refined has been fully updated

                const ctags = tags[coarse.tile_offset .. coarse.tile_offset + coarse.tile_total];

                // 3.3 Generate new patches.
                // *************************

                // Start filling coarse children
                coarse_children.shrinkRetainingCapacity(0);

                try coarse_children.append(self.gpa, 0);

                for (coarse.patches.items(.bounds), coarse.patches.items(.tile_offset), 0..) |cpbounds, cpoffset, cpid| {
                    // Reset arena for new "frame"
                    defer _ = arena.reset(.retain_capacity);

                    // Make aliases for patch variables
                    const cpspace: IndexSpace = IndexSpace.fromBox(cpbounds);
                    const cptags: []bool = ctags[cpoffset..(cpoffset + cpspace.total())];

                    // As well as clusters in this patch
                    const upclusters: []const IndexBox = clusters.items[cluster_offsets.items[cpid]..cluster_offsets.items[cpid + 1]];
                    const upcluster_index_map: []const usize = cluster_index_map.items[cluster_offsets.items[cpid]..cluster_offsets.items[cpid + 1]];

                    // Preprocess tags to include all elements from clusters (and one tile buffer region around cluster)
                    preprocessTagsOnPatch(cptags, cpspace, upclusters);

                    // Run partitioning algorithm on coarse patch to determine blocks.
                    var cppartitioner = try PartitionSpace.init(scratch, cpbounds.size, upclusters);
                    defer cppartitioner.deinit();

                    try cppartitioner.build(cptags, patch_max_tiles, patch_efficiency);

                    // Iterate computed patches
                    for (cppartitioner.partitions()) |patch| {
                        // Build a space from the patch size.
                        const pspace: IndexSpace = IndexSpace.fromBox(patch.bounds);

                        // Allocate sufficient space to hold children of this patch
                        var pchildren: []usize = try scratch.alloc(usize, patch.children_total);
                        defer scratch.free(pchildren);

                        // Fill with computed children of patch (using index map to find global index into refined children)
                        for (patch.children_offset..(patch.children_offset + patch.children_total), pchildren) |child_id, *child| {
                            child.* = upcluster_index_map[child_id];
                        }

                        // Build window into tags for this patch
                        var ptags: []bool = try scratch.alloc(bool, pspace.total());
                        defer scratch.free(ptags);
                        // Set ptags using window of uptags
                        cpspace.copyWindow(patch.bounds, bool, ptags, cptags);

                        // Run patitioning algorithm on patch to determine blocks
                        var ppartitioner = try PartitionSpace.init(scratch, pspace.size, &[_]IndexBox{});
                        defer ppartitioner.deinit();

                        try ppartitioner.build(ptags, block_max_tiles, block_efficiency);

                        // Offset blocks to be in global space
                        var pblocks: []IndexBox = try scratch.alloc(IndexBox, ppartitioner.partitions().len);
                        scratch.free(pblocks);

                        // Compute global bounds of the patch
                        const pbounds: IndexBox = .{
                            .origin = add(patch.bounds.origin, cpbounds.origin),
                            .size = patch.bounds.size,
                        };

                        // Iterate computed blocks and offset to find global bounds of each block
                        for (ppartitioner.partitions(), pblocks) |block, *pblock| {
                            pblock.* = block.bounds;

                            for (0..N) |axis| {
                                pblock.origin[axis] += pbounds.origin[axis];
                            }
                        }

                        // Add patch to level
                        try target.addPatch(self.gpa, pbounds, pblocks, pchildren);
                    }

                    try coarse_children.append(self.gpa, target.patchTotal());

                    // target.transfer_patches.items(.patch_total)[cpid] += partition_space.parts.len;
                }

                target.refine();

                // At this moment in time
                // - coarse is old
                // - target has been fully updated
                // - refined has been fully updated
            }

            // Update base
            const base: *Level = &self.levels.items[0];

            if (self.active_levels > 1) {
                try base.buildBase(self.gpa, self.index_size, self.getLevel(1).patchTotal());
            } else {
                try base.buildBase(self.gpa, self.index_size, 0);
            }

            // 4. Recompute level offsets and totals.
            // **************************************

            self.computeOffsets();
        }

        fn preprocessTagsOnPatch(tags: []bool, space: IndexSpace, clusters: []const IndexBox) void {
            for (clusters) |upcluster| {
                var cluster: IndexBox = upcluster;

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
            var tile_offset: usize = 0;
            var cell_offset: usize = 0;

            for (self.levels.items) |*level| {
                level.computeOffsets(self.tile_width);

                level.tile_offset = tile_offset;
                level.cell_offset = cell_offset;

                tile_offset += level.tile_total;
                cell_offset += level.cell_total;
            }

            self.cell_total = cell_offset;
            self.tile_total = tile_offset;
        }

        fn computeTotalLevels(self: *const Self, tags: []const bool, max_levels: usize) usize {
            // Clamp to max levels
            if (self.active_levels == max_levels) {
                return max_levels;
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

        fn resizeActiveLevels(self: *Self, total: usize) !void {
            assert(self.levels.items.len > 0);
            while (total > self.levels.items.len) {
                const size: [N]usize = scaled(self.levels.getLast().index_size, 2);

                try self.levels.append(self.gpa, Level.init(size));
            }

            self.active_levels = total;
        }
    };
}

test "mesh regridding" {
    // const expect = std.testing.expect;
    // const expectEqualSlices = std.testing.expectEqualSlices;

    const allocator = std.testing.allocator;

    const config: Mesh(2).Config = .{
        .physical_bounds = .{
            .origin = [_]f64{ 0.0, 0.0 },
            .size = [_]f64{ 1.0, 1.0 },
        },
        .index_size = [_]usize{ 10, 10 },
        .tile_width = 16,
    };

    var mesh: Mesh(2) = try Mesh(2).init(allocator, config);
    defer mesh.deinit();

    var tags: []bool = try allocator.alloc(bool, mesh.tile_total);
    defer allocator.free(tags);

    // Tag all
    @memset(tags, true);

    try mesh.regrid(tags, 1, 0.1, 80, 0.7, 80);
}

test {}
