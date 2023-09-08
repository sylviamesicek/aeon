const std = @import("std");
const meta = std.meta;

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
const boundaries = @import("boundary.zig");
const levels = @import("level.zig");

// Public Exports

pub const BoundaryCondition = boundaries.BoundaryCondition;

pub fn Mesh(comptime N: usize, comptime O: usize) type {
    return struct {
        /// Allocator used for various arraylists stored in this struct.
        gpa: Allocator,
        /// Configuration of Mesh
        config: Config,
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
        pub const Level = levels.Level(N, O);
        pub const Block = levels.Block(N);
        pub const Patch = levels.Patch(N);
        const IndexBox = geometry.Box(N, usize);
        const RealBox = geometry.Box(N, f64);
        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const PartitionSpace = geometry.PartitionSpace(N);
        const Region = geometry.Region(N);
        const StencilSpace = basis.StencilSpace(N, O);

        pub const Config = struct {
            physical_bounds: RealBox,
            index_size: [N]usize,
            tile_width: usize,
            global_refinement: usize,

            pub fn baseTileSpace(self: Config) IndexSpace {
                var scale: usize = 1;

                for (0..self.global_refinement) |_| {
                    scale *= 2;
                }

                return IndexSpace.fromSize(self.index_size).scale(scale);
            }

            pub fn baseCellSpace(self: Config) IndexSpace {
                return self.baseTileSpace().scale(self.tile_width).extendUniform(2 * O);
            }

            pub fn check(self: Config) void {
                assert(self.tile_width >= 1);
                assert(self.tile_width >= O);
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

        pub fn init(allocator: Allocator, config: Config) Self {
            // Check config
            config.check();
            // Scale initial size by 2^global_refinement
            const tile_space: IndexSpace = config.baseTileSpace();
            const cell_space: IndexSpace = config.baseCellSpace();

            const base: Base = .{
                .index_size = tile_space.size,
                .tile_total = tile_space.total(),
                .cell_total = cell_space.total(),
            };

            return .{
                .gpa = allocator,
                .config = config,
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

        pub fn baseTileSlice(self: *const Self, mesh_slice: anytype) @TypeOf(mesh_slice) {
            return mesh_slice[0..self.baseTileTotal()];
        }

        pub fn baseCellSlice(self: *const Self, mesh_slice: anytype) @TypeOf(mesh_slice) {
            return mesh_slice[0..self.baseCellTotal()];
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

        pub fn buildTransferMap(self: *const Self, map: []usize) !void {
            assert(map.len == self.transferTotal());

            @memset(map, std.math.maxInt(usize));
            @memset(self.baseTransferSlice(usize, map), 0);

            for (self.levels.items, 0..) |*level, l| {
                const level_map: []usize = self.levelTransferSlice(l, usize, map);

                for (level.transfer_blocks.items(.bounds), level.transfer_blocks.items(.patch), 0..) |bounds, parent, id| {
                    const pbounds: IndexBox = level.transfer_patches.items(.bounds)[parent];
                    const tile_offset: usize = level.transfer_patches.items(.tile_offset)[parent];
                    const tile_total: usize = level.transfer_patches.items(.tile_total)[parent];
                    const tile_to_block: []usize = level_map[tile_offset..(tile_offset + tile_total)];

                    pbounds.space().fillSubspace(bounds.relativeTo(pbounds), usize, tile_to_block, id);
                }
            }
        }

        pub fn fillBoundary(self: *const Self, boundary: anytype, block_map: []const usize, field: []f64) !void {
            assert(block_map.len == self.tileTotal());
            assert(field.len == self.cellTotal());
            assert(boundaries.hasConditionDecl(N)(boundary));

            self.fillBaseExterior(boundary, field);

            for (0..self.active_levels) |i| {
                self.fillLevelBoundary(i, boundary, block_map, field);
            }
        }

        fn fillBaseExterior(self: *const Self, boundary: anytype, field: []f64) void {
            const regions = Region.orderedRegions();

            const base_field: []f64 = self.baseCellSlice(field);
            const stencil_space = self.baseStencilSpace();

            inline for (regions) |region| {
                stencil_space.fillBoundary(region, boundary, base_field);
            }
        }

        fn fillLevelBoundary(self: *const Self, level: usize, boundary: anytype, block_map: []const usize, field: []f64) !void {
            const target: *const Level = self.levels[level];
            const index_size: IndexBox = target.index_size;

            const blocks = target.blocks.slice();

            for (0..blocks.len) |block| {
                const bounds: IndexBox = blocks.items(.bounds)[block];
                const offset: usize = blocks.items(.cell_offset)[block];
                _ = offset;
                const total: usize = blocks.items(.cell_total)[block];
                _ = total;

                const regions = Region.orderedRegions();

                inline for (regions[1..]) |region| {
                    var exterior: bool = false;

                    for (0..N) |i| {
                        if (region.sides[i] == .left and bounds.origin[i] == 0) {
                            exterior = true;
                        } else if (region.sides[i] == .right and bounds.origin[i] + bounds.size[i] == index_size[i]) {
                            exterior == true;
                        }
                    }

                    if (exterior) {
                        self.fillLevelInterior(region, level, block, block_map, field);
                    } else {
                        self.fillLevelExterior(region, level, block, boundary, field);
                    }
                }
            }
        }

        fn fillLevelInterior(self: *const Self, comptime region: Region, level: usize, block: usize, block_map: []const usize, field: []f64) void {
            const target: *const Level = &self.levels[level];
            const target_field: []f64 = target.levelCellSlice(field);

            const blocks = target.blocks.slice();
            const patches = target.patches.slice();

            const block_bounds: IndexBox = blocks.items(.bounds)[block];
            const patch = blocks.items(.patch)[block];

            const block_field: []f64 = target.blockCellSlice(block, target_field);
            const block_stencil: StencilSpace = self.levelStencilSpace(level, block);

            const patch_bounds: IndexBox = patches.items(.bounds)[patch];
            const patch_space = patch_bounds.space();
            const patch_block_map: []const usize = target.patchTileSlice(patch, target.levelTileSlice(block_map));

            const bounds: IndexBox = block_bounds.relativeTo(patch_bounds);

            var inner_tiles = region.innerFaceIndices(1, bounds.size);

            while (inner_tiles.next()) |inner_tile| {
                var buffer_tile: [N]usize = bounds.globalFromLocal(inner_tile);

                for (0..N) |i| {
                    buffer_tile[i] += region.extentDir()[i];
                }

                var origin: [N]usize = undefined;

                for (0..N) |i| {
                    origin[i] = inner_tile[i] * self.config.tile_width;
                }

                const neighbor: usize = patch_block_map[patch_space.linearFromCartesian(buffer_tile)];

                if (neighbor == maxInt(usize)) {
                    var coarse_inner_tile: [N]usize = undefined;
                    var coarse_buffer_tile: [N]usize = undefined;

                    for (0..N) |i| {
                        coarse_inner_tile[i] = inner_tile[i] / 2;
                        coarse_buffer_tile[i] = buffer_tile[i] / 2;
                    }

                    if (level == 0) {
                        const coarse: *const Level = &self.levels[level - 1];
                        const coarse_patch = coarse.parents[patch];
                        const coarse_block_map: []const usize = coarse.levelTileSlice(block_map);
                        const cpblock_map: []const usize = coarse.patchTileSlice(coarse_patch, coarse_block_map);
                        const coarse_field: []const f64 = coarse.levelCellSlice(field);

                        const coarse_patch_bounds = coarse.patches.items(.bounds)[coarse_patch];
                        const coarse_patch_space = coarse_patch_bounds.space();

                        const coarse_neighbor = cpblock_map[coarse_patch_space.linearFromCartesian(coarse_buffer_tile)];

                        var neighbor_bounds = coarse.blocks.items(.bounds)[coarse_neighbor].relativeTo(coarse_patch_bounds);
                        const coarse_neighbor_stencil: StencilSpace = self.levelStencilSpace(level - 1, coarse.blocks.items(.bounds)[coarse_neighbor]);

                        neighbor_bounds.refine();

                        var norigin: [N]usize = neighbor_bounds.localFromGlobal(bounds.globalFromLocal(inner_tile));

                        for (0..N) |i| {
                            norigin[i] *= self.config.tile_width;
                        }

                        const neighbor_field: []const f64 = coarse.blockCellSlice(coarse_neighbor, coarse_field);

                        var indices = region.cartesianIndices(O, [1]usize{self.config.tile_width} ** N);

                        while (indices.next()) |index| {
                            var bcell: [N]usize = undefined;
                            var ncell: [N]usize = undefined;

                            for (0..N) |i| {
                                bcell[i] = origin[i] + index[i];
                                ncell[i] = norigin[i] + index[i];
                            }

                            block_stencil.setValue(bcell, block_field, coarse_neighbor_stencil.prolong(ncell, neighbor_field));
                        }
                    } else {
                        var base_bounds: IndexBox = .{
                            .origin = [1]usize{0} ** N,
                            .size = self.base.index_size,
                        };

                        base_bounds.refine();

                        var norigin: [N]usize = base_bounds.localFromGlobal(bounds.globalFromLocal(inner_tile));

                        for (0..N) |i| {
                            norigin[i] *= self.config.tile_width;
                        }

                        const base_field: []const f64 = self.baseCellSlice(field);
                        const base_stencil: StencilSpace = self.baseStencilSpace();

                        var indices = region.cartesianIndices(O, [1]usize{self.config.tile_width} ** N);

                        while (indices.next()) |index| {
                            var bcell: [N]usize = undefined;
                            var ncell: [N]usize = undefined;

                            for (0..N) |i| {
                                bcell[i] = origin[i] + index[i];
                                ncell[i] = norigin[i] + index[i];
                            }

                            block_stencil.setValue(bcell, block_field, base_stencil.prolong(ncell, base_field));
                        }
                    }
                } else {
                    // Copy from neighbor on same level
                    const neighbor_field: []const f64 = target.blockCellSlice(neighbor, target_field);
                    const neighbor_bounds: IndexBox = blocks.items(.bounds)[neighbor].relativeTo(patch_bounds);
                    const neighbor_stencil: StencilSpace = self.levelStencilSpace(level, blocks.items(.bounds)[neighbor]);

                    var norigin: [N]usize = neighbor_bounds.localFromGlobal(bounds.globalFromLocal(inner_tile));

                    for (0..N) |i| {
                        norigin[i] *= self.config.tile_width;
                    }

                    var indices = region.cartesianIndices(O, [1]usize{self.config.tile_width} ** N);

                    while (indices.next()) |index| {
                        var bcell: [N]usize = undefined;
                        var ncell: [N]usize = undefined;

                        for (0..N) |i| {
                            bcell[i] = origin[i] + index[i];
                            ncell[i] = norigin[i] + index[i];
                        }

                        block_stencil.setValue(bcell, block_field, neighbor_stencil.value(ncell, neighbor_field));
                    }
                }
            }
        }

        fn fillLevelExterior(self: *const Self, comptime region: Region, level: usize, block: usize, boundary: anytype, field: []f64) void {
            const target: *const Level = &self.levels[level];

            const level_field: []f64 = target.levelCellSlice(field);
            const block_field: []f64 = target.blockCellSlice(block, level_field);

            const stencil_space = self.levelStencilSpace(level, block);

            stencil_space.fillBoundary(region, boundary, block_field);
        }

        //         if (block_neighbor == maxInt(usize)) {
        //             const cpbounds: IndexBox = cbounds[cparent];
        //             const cpspace = cpbounds.space();

        //             var cglobal: [N]usize = global;

        //             for (0..N) |i| {
        //                 cglobal[i] /= 2;
        //             }

        //             const cblock_neighbor: usize = cpblock_map[cpspace.linearFromCartesian(cpbounds.localFromGlobal(cglobal))];

        //             self.interpolateGhostFromCoarse(region, level, block, cblock_neighbor, global, field);
        //         } else {
        //             self.copyGhostFromNeighbor(region, level, block, block_neighbor, global, field);
        //         }
        //     }
        // }

        fn baseStencilSpace(self: *const Self) StencilSpace {
            return .{
                .physical_bounds = self.physical_bounds,
                .index_size = IndexSpace.fromSize(self.base.index_size).scale(self.tile_width).size,
            };
        }

        fn levelStencilSpace(self: *const Self, level: usize, block: IndexBox) StencilSpace {
            const index_size: [N]usize = self.levels[level].index_size;

            var physical_bounds: RealBox = undefined;

            for (0..N) |i| {
                const sratio: f64 = @as(f64, @floatFromInt(block.size[i])) / @as(f64, @floatFromInt(index_size[i]));
                const oratio: f64 = @as(f64, @floatFromInt(block.origin[i])) / @as(f64, @floatFromInt(index_size[i] - 1));

                physical_bounds.size[i] = self.physical_bounds.size[i] * sratio;
                physical_bounds.origin[i] = self.physical_bounds.origin[i] + self.physical_bounds.size[i] * oratio;
            }

            return .{
                .physical_bounds = physical_bounds,
                .index_size = block.space().scale(self.tile_width).size,
            };
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
                            const start = coarse_children[tpid];
                            const end = coarse_children[tpid + 1];

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

                try target.setTotalChildren(self.gpa, clusters.len);
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

                    ctags = coarse.levelTileSlice(tags);
                    clen = coarse.patches.len;
                    cbounds = coarse.patches.items(.bounds);
                    coffsets = coarse.patches.items(.tile_offset);
                } else {
                    // Otherwise use base data
                    ctags = self.baseTileSlice(tags);
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
                    const cpspace: IndexSpace = cpbounds.space();
                    const cptags: []bool = ctags[cpoffset..(cpoffset + cpspace.total())];

                    // As well as clusters in this patch
                    const upclusters: []const IndexBox = clusters.items[cluster_offsets.items[cpid]..cluster_offsets.items[cpid + 1]];
                    const upcluster_index_map: []const usize = cluster_index_map.items[cluster_offsets.items[cpid]..cluster_offsets.items[cpid + 1]];

                    // Preprocess tags to include all elements from clusters (and one tile buffer region around cluster)
                    preprocessTagsOnPatch(cptags, cpspace, upclusters);

                    // Run partitioning algorithm on coarse patch to determine blocks.
                    var cppartitioner = try PartitionSpace.init(scratch, cpbounds.size, upclusters);
                    defer cppartitioner.deinit();

                    try cppartitioner.build(cptags, config.patch_max_tiles, config.patch_efficiency);

                    // Iterate computed patches
                    for (cppartitioner.partitions()) |patch| {
                        // Build a space from the patch size.
                        const pspace: IndexSpace = patch.bounds.space();

                        // Allocate sufficient space to hold children of this patch
                        var pchildren: []usize = try scratch.alloc(usize, patch.children_total);
                        defer scratch.free(pchildren);

                        // Fill with computed children of patch (using index map to find global index into refined children)
                        for (patch.children_offset..(patch.children_offset + patch.children_total), pchildren) |child_id, *child| {
                            child = upcluster_index_map[child_id];
                        }

                        // Build window into tags for this patch
                        var ptags: []bool = try scratch.alloc(bool, pspace.total());
                        defer scratch.free(ptags);
                        // Set ptags using window of uptags
                        cpspace.window(patch.bounds, bool, ptags, cptags);

                        // Run patitioning algorithm on patch to determine blocks
                        var ppartitioner = try PartitionSpace.init(scratch, pspace.size, &[_]IndexBox{});
                        defer ppartitioner.deinit();

                        try ppartitioner.build(ptags, config.block_max_tiles, config.block_efficiency);

                        // Offset blocks to be in global space
                        var pblocks: []IndexBox = try scratch.alloc(IndexBox, ppartitioner.partitions().len);
                        try scratch.free(pblocks);

                        // Compute global bounds of the patch
                        var pbounds: IndexBox = patch.bounds;

                        for (0..N) |i| {
                            pbounds.origin[i] += cpbounds.origin[i];
                        }

                        // Iterate computed blocks and offset to find global bounds of each block
                        for (ppartitioner.partitions(), pblocks) |block, *pblock| {
                            pblock = block;

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
            var tile_offset: usize = self.base.tile_total;
            var cell_offset: usize = self.base.cell_total;

            var transfer_tile_offset: usize = self.base.tile_total;
            var transfer_cell_offset: usize = self.base.cell_total;

            for (self.levels.items) |*level| {
                level.computeOffsets(self.tile_width);

                level.tile_offset = tile_offset;
                level.cell_offset = cell_offset;

                level.transfer_tile_offset = transfer_tile_offset;
                level.transfer_cell_offset = transfer_cell_offset;

                tile_offset += level.tile_total;
                cell_offset += level.cell_total;

                transfer_tile_offset += level.transfer_tile_total;
                transfer_cell_offset += level.transfer_cell_total;
            }

            self.cell_total = cell_offset;
            self.tile_total = tile_offset;

            self.transfer_tile_total = transfer_tile_offset;
            self.transfer_cell_total = transfer_cell_offset;
        }

        fn computeTotalLevels(self: *const Self, tags: []const bool, config: RegridConfig) usize {
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

        fn resizeActiveLevels(self: *Self, total: usize) !void {
            while (total > self.levels.items.len) {
                if (self.levels.items.len == 0) {
                    const size: [N]usize = IndexSpace.fromSize(self.base.index_size).scale(2).size;

                    try self.levels.append(self.gpa, Level.init(size));
                } else {
                    const size: [N]usize = IndexSpace.fromSize(self.levels.getLast().index_size).scale(2).size;

                    try self.levels.append(self.gpa, Level.init(size));
                }
            }

            self.active_levels = total;
        }
    };
}

test "mesh regridding" {
    const expect = std.testing.expect;
    _ = expect;
    const expectEqualSlices = std.testing.expectEqualSlices;
    _ = expectEqualSlices;

    const allocator = std.testing.allocator;

    const Mesh2 = Mesh(2, 0);

    const config: Mesh2.Config = .{
        .physical_bounds = .{
            .origin = [_]f64{ 0.0, 0.0 },
            .size = [_]f64{ 1.0, 1.0 },
        },
        .index_size = [_]usize{ 10, 10 },
        .tile_width = 16,
        .global_refinement = 2,
    };

    var mesh: Mesh2 = Mesh2.init(allocator, config);
    defer mesh.deinit();

    var tags: []bool = try allocator.alloc(bool, mesh.tileTotal());
    defer allocator.free(tags);

    // Tag all
    @memset(tags, true);

    try mesh.regrid(tags, .{
        .max_levels = 1,
        .block_max_tiles = 80,
        .block_efficiency = 0.7,
        .patch_max_tiles = 80,
        .patch_efficiency = 0.1,
    });
}

test {}
