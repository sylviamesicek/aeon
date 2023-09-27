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
        /// The number of cells along each time edge.
        tile_width: usize,
        /// The maximum number of tiles on one edge of a patch.
        patch_max_tiles: usize,
        /// The maximum number of tiles on one edge of a block
        block_max_tiles: usize,
        /// Base level (after performing global_refinement).
        base: Base,
        /// Refined levels
        levels: ArrayListUnmanaged(Level),
        /// Number of levels which are currently active.
        active_levels: usize,
        /// Total number of tiles in mesh
        tile_total: usize,
        /// Total number of cells in mesh
        cell_total: usize,

        // Public types
        pub const Base = levels.Base(N);
        pub const Level = levels.Level(N);
        pub const Block = levels.Block(N);
        pub const Patch = levels.Patch(N);

        /// Configuration for a mesh.
        pub const Config = struct {
            physical_bounds: RealBox,
            index_size: [N]usize,
            tile_width: usize,
            patch_max_tiles: usize,
            block_max_tiles: usize,

            /// Checks that the given mesh config is valid.
            pub fn check(self: Config) void {
                assert(self.tile_width >= 1);
                assert(self.patch_max_tiles >= self.block_max_tiles);
                assert(self.block_max_tiles > 0);
                for (0..N) |i| {
                    assert(self.index_size[i] > 0);
                    assert(self.physical_bounds.size[i] > 0.0);
                    assert(self.block_max_tiles >= self.index_size[i]);
                }
            }

            fn baseTileSpace(self: Config) IndexSpace {
                return IndexSpace.fromSize(self.index_size);
            }

            fn baseCellSpace(self: Config) IndexSpace {
                return IndexSpace.fromSize(scaled(self.index_size, self.tile_width));
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
        pub fn init(allocator: Allocator, config: Config) Self {
            // Check config
            config.check();
            // Scale initial size by 2^global_refinement
            const tile_space: IndexSpace = config.baseTileSpace();
            const cell_space: IndexSpace = config.baseCellSpace();

            const b: Base = .{
                .index_size = tile_space.size,
                .tile_total = tile_space.total(),
                .cell_total = cell_space.total(),
            };

            return .{
                .gpa = allocator,
                .physical_bounds = config.physical_bounds,
                .tile_width = config.tile_width,
                .patch_max_tiles = config.patch_max_tiles,
                .block_max_tiles = config.block_max_tiles,
                .base = b,
                .levels = .{},
                .active_levels = 0,
                .tile_total = b.tile_total,
                .cell_total = b.cell_total,
            };
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

        /// Get the base level of this mesh.
        pub fn getBase(self: *const Self) *const Base {
            return &self.base;
        }

        /// Get a higher active level of this mesh.
        pub fn getLevel(self: *const Self, level: usize) *const Level {
            assert(level < self.active_levels);
            return &self.levels.items[level];
        }

        pub fn baseTileSlice(self: *const Self, mesh_slice: anytype) @TypeOf(mesh_slice) {
            return mesh_slice[0..self.base.tile_total];
        }

        pub fn baseCellSlice(self: *const Self, mesh_slice: anytype) @TypeOf(mesh_slice) {
            return mesh_slice[0..self.base.cell_total];
        }

        pub fn levelTileSlice(self: *const Self, level: usize, patch: usize, mesh_slice: anytype) @TypeOf(mesh_slice) {
            const target = self.getLevel(level);
            const patch_offset = target.patches.items(.tile_offset)[patch];
            const patch_total = target.patches.items(.tile_total)[patch];
            const offset = target.tile_offset + patch_offset;

            return mesh_slice[offset..patch_total];
        }

        pub fn levelCellSlice(self: *const Self, level: usize, block: usize, mesh_slice: anytype) @TypeOf(mesh_slice) {
            const target = self.getLevel(level);
            const patch_offset = target.blocks.items(.cell_offset)[block];
            const patch_total = target.blocks.items(.cell_total)[block];
            const offset = target.cell_offset + patch_offset;

            return mesh_slice[offset..patch_total];
        }

        // *************************
        // Block Map ***************
        // *************************

        /// Builds a block map, ie a map from each tile in the mesh to the id of the block that tile is in.
        pub fn buildBlockMap(self: *const Self, map: []usize) !void {
            assert(map.len == self.tileTotal());

            @memset(map, maxInt(usize));
            @memset(self.baseTileSlice(usize, map), 0);

            for (self.levels.items, 0..) |level, l| {
                for (level.blocks.items(.bounds), level.blocks.items(.patch), 0..) |bounds, parent, id| {
                    const pbounds: IndexBox = level.patches.items(.bounds)[parent];
                    const tile_to_block: []usize = self.levelTileSlice(l, parent, map);

                    pbounds.space().fillSubspace(bounds.relativeTo(pbounds), usize, tile_to_block, id);
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

        /// Gets the physical bounds of the base.
        pub fn basePhysicalBounds(self: *const Self) RealBox {
            return self.physical_bounds;
        }

        /// Gets the physical bounds of a higher level
        pub fn levelPhysicalBounds(self: *const Self, level: usize, block: usize) RealBox {
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
            block_efficiency: f64,
            patch_efficiency: f64,
        ) !void {
            assert(max_levels >= self.active_levels);

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

            // Bounds for base level.
            const bbounds: IndexBox = .{
                .origin = [1]usize{0} ** N,
                .size = self.base.index_size,
            };

            // Slices at top of scope ensures we don't reference a temporary.
            const bbounds_slice: []const IndexBox = &[_]IndexBox{bbounds};
            const boffsets_slice: []const usize = &[_]usize{0};

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
                // - target is old
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
                        const cpbounds: IndexBox = coarse.patches.items(.bounds)[cpid];

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
                } else if (refined_exists) {
                    const refined: *const Level = &self.levels.items[level_id + 1];

                    // Every patch in target is a child of base block.
                    for (0..target.patches.len) |tpid| {
                        for (target.childrenSlice(tpid)) |child| {
                            var patch: IndexBox = refined.patches.items(.bounds)[child];

                            try patch.coarsen();
                            try patch.coarsen();

                            try clusters.append(self.gpa, patch);
                            try cluster_index_map.append(self.gpa, child);
                        }
                    }

                    try cluster_offsets.append(self.gpa, clusters.items.len);
                } else {
                    try cluster_offsets.append(self.gpa, clusters.items.len);
                }

                // Now shrinkRetainingCapacitywe can clear the target arrays
                // target.transfer_blocks.shrinkRetainingCapacity(0);
                // target.transfer_patches.shrinkRetainingCapacity(0);

                // if (coarse_exists) {
                //     const coarse: *const Level = &self.levels.items[level_id - 1];

                //     for (0..coarse.patches.len) |cpid| {
                //         const upbounds: IndexBox = coarse.patches.items(.bounds)[cpid];

                //         try target.transfer_patches.append(self.gpa, TransferPatch{
                //             .bounds = upbounds,
                //             .block_offset = 0,
                //             .block_total = 0,
                //             .patch_offset = 0,
                //             .patch_total = 0,
                //         });
                //     }

                //     for (target.blocks.items(.bounds), target.blocks.items(.patch)) |bounds, patch| {
                //         const cpid: usize = coarse.parents.items[patch];
                //         target.transfer_patches.items(.block_total)[cpid] += 1;

                //         try target.transfer_blocks.append(self.gpa, TransferBlock{
                //             .bounds = bounds,
                //             .patch = cpid,
                //         });
                //     }
                // } else {
                //     try target.transfer_patches.append(self.gpa, TransferPatch{
                //         .bounds = bbounds,
                //         .block_offset = 0,
                //         .block_total = 0,
                //         .patch_offset = 0,
                //         .patch_total = 0,
                //     });

                //     for (target.blocks.items(.bounds)) |bounds| {
                //         target.transfer_patches.items(.block_total)[0] += 1;

                //         try target.transfer_blocks.append(self.gpa, TransferBlock{
                //             .bounds = bounds,
                //             .patch = 0,
                //         });
                //     }
                // }

                try target.setTotalChildren(self.gpa, clusters.items.len);
                target.clearRetainingCapacity();

                // At this moment in time
                // - coarse is old
                // - target is cleared
                // - refined has been fully updated

                // Variables that depend on coarse existing
                var ctags: []bool = undefined;
                var clen: usize = undefined;
                var cbounds: []const IndexBox = undefined;
                var coffsets: []const usize = undefined;

                if (coarse_exists) {
                    // Get underlying data
                    const coarse: *const Level = &self.levels.items[level_id - 1];

                    ctags = tags[coarse.tile_offset .. coarse.tile_offset + coarse.tile_total];
                    clen = coarse.patches.len;
                    cbounds = coarse.patches.items(.bounds);
                    coffsets = coarse.patches.items(.tile_offset);
                } else {
                    // Otherwise use base data
                    ctags = tags[0..self.base.tile_total];
                    clen = 1;
                    cbounds = bbounds_slice;
                    coffsets = boffsets_slice;
                }

                // 3.3 Generate new patches.
                // *************************

                // Start filling coarse children
                coarse_children.shrinkRetainingCapacity(0);

                try coarse_children.append(self.gpa, 0);

                for (0..clen) |cpid| {
                    // Reset arena for new "frame"
                    defer _ = arena.reset(.retain_capacity);

                    // Make aliases for patch variables
                    const cpbounds: IndexBox = cbounds[cpid];
                    const cpoffset: usize = coffsets[cpid];
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

                    try cppartitioner.build(cptags, self.patch_max_tiles, patch_efficiency);

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
                        cpspace.window(patch.bounds, bool, ptags, cptags);

                        // Run patitioning algorithm on patch to determine blocks
                        var ppartitioner = try PartitionSpace.init(scratch, pspace.size, &[_]IndexBox{});
                        defer ppartitioner.deinit();

                        try ppartitioner.build(ptags, self.block_max_tiles, block_efficiency);

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

                // if (coarse_exists) {
                //     const coarse: *Level = &self.levels.items[level_id - 1];

                //     try coarse.children.resize(self.gpa, target.patches.len);
                //     try coarse.parents.resize(self.gpa, target.patches.len);

                //     for (0..target.patches.len) |i| {
                //         coarse.children.items[i] = i;
                //     }

                //     for (0..clen) |cpid| {
                //         const start = coarse_children.items[cpid];
                //         const end = coarse_children.items[cpid + 1];

                //         coarse.patches.items(.children_offset)[cpid] = start;
                //         coarse.patches.items(.children_total)[cpid] = end - start;

                //         @memset(coarse.parents.items[start..end], cpid);
                //     }
                // }

                target.refine();

                // At this moment in time
                // - coarse is old
                // - target has been fully updated
                // - refined has been fully updated
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

                space.fillSubspace(cluster, bool, tags, true);
            }
        }

        fn computeOffsets(self: *Self) void {
            self.base.computeOffsets(self.tile_width);

            var tile_offset: usize = self.base.tile_total;
            var cell_offset: usize = self.base.cell_total;

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
            while (total > self.levels.items.len) {
                if (self.levels.items.len == 0) {
                    const size: [N]usize = scaled(self.base.index_size, 2);

                    try self.levels.append(self.gpa, Level.init(size));
                } else {
                    const size: [N]usize = scaled(self.levels.getLast().index_size, 2);

                    try self.levels.append(self.gpa, Level.init(size));
                }
            }

            self.active_levels = total;
        }
    };
}

test "mesh regridding" {
    // const expect = std.testing.expect;
    // const expectEqualSlices = std.testing.expectEqualSlices;

    const allocator = std.testing.allocator;

    const Mesh2 = Mesh(2, 0);

    const config: Mesh2.Config = .{
        .physical_bounds = .{
            .origin = [_]f64{ 0.0, 0.0 },
            .size = [_]f64{ 1.0, 1.0 },
        },
        .index_size = [_]usize{ 10, 10 },
        .tile_width = 16,
        .block_max_tiles = 80,
        .patch_max_tiles = 80,
    };

    var mesh: Mesh2 = Mesh2.init(allocator, config);
    defer mesh.deinit();

    var tags: []bool = try allocator.alloc(bool, mesh.tile_total);
    defer allocator.free(tags);

    // Tag all
    @memset(tags, true);

    try mesh.regrid(tags, .{
        .max_levels = 1,
        .block_efficiency = 0.7,
        .patch_efficiency = 0.1,
    });
}

test {}
