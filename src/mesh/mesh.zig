const std = @import("std");
const meta = std.meta;

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const ArenaAllocator = std.heap.ArenaAllocator;

const assert = std.debug.assert;
const exp2 = std.math.exp2;

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const boundaries = @import("boundary.zig");

// Public Exports

pub const BoundaryCondition = boundaries.BoundaryCondition;

pub fn Mesh(comptime N: usize, comptime O: usize) type {
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
        /// Total number of tiles in mesh
        tile_total: usize,
        /// Total number of cells in mesh
        cell_total: usize,
        transfer_tile_total: usize,
        transfer_cell_total: usize,
        /// Base level (after performing global_refinement).
        base: Base,
        /// Refined levels
        levels: ArrayListUnmanaged(Level),
        /// Number of levels which are active.
        active_levels: usize,

        // Aliases
        const Self = @This();
        const IndexBox = geometry.Box(N, usize);
        const RealBox = geometry.Box(N, f64);
        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const PartitionSpace = geometry.PartitionSpace(N);
        const Region = geometry.Region(N, O);
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

        pub const Block = struct {
            bounds: IndexBox,
            patch: usize,
            cell_total: usize = 0,
            cell_offset: usize = 0,
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

        pub const TransferBlock = struct {
            bounds: IndexBox,
            patch: usize,

            cell_total: usize = 0,
            cell_offset: usize = 0,
        };

        pub const TransferPatch = struct {
            bounds: IndexBox,

            block_offset: usize,
            block_total: usize,
            patch_offset: usize,
            patch_total: usize,

            tile_offset: usize = 0,
            tile_total: usize = 0,
        };

        pub const Level = struct {
            index_size: [N]usize,

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

            /// Stores old blocks
            transfer_blocks: MultiArrayList(TransferBlock),
            /// Stores underlying patches
            transfer_patches: MultiArrayList(TransferPatch),
            transfer_tile_total: usize,
            transfer_cell_total: usize,
            transfer_tile_offset: usize,
            transfer_cell_offset: usize,

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

                    .transfer_blocks = .{},
                    .transfer_patches = .{},
                    .transfer_tile_total = 0,
                    .transfer_cell_total = 0,
                    .transfer_tile_offset = 0,
                    .transfer_cell_offset = 0,
                };
            }

            /// Frees a level
            fn deinit(self: *Level, allocator: Allocator) void {
                self.blocks.deinit(allocator);
                self.patches.deinit(allocator);
                self.children.deinit(allocator);
                self.parents.deinit(allocator);
                self.transfer_blocks.deinit(allocator);
                self.transfer_patches.deinit(allocator);
            }

            /// Gets a slice of children indices for each patch
            fn childrenSlice(self: *Level, patch: usize) []usize {
                const offset = self.patches.items(.children_offset)[patch];
                const count = self.patches.items(.children_total)[patch];
                return self.children.items[offset..(offset + count)];
            }

            /// Computes patch and block offsets and level totals for tiles and cells.
            fn computeOffsets(self: *Level, tile_width: usize) void {
                var tile_offset: usize = 0;

                for (self.patches.items(.bounds), self.patches.items(.tile_total), self.patches.items(.tile_offset)) |bounds, *total, *offset| {
                    offset.* = tile_offset;
                    total.* = bounds.space().total();
                    tile_offset += total.*;
                }

                var cell_offset: usize = 0;

                for (self.blocks.items(.bounds), self.blocks.items(.cell_total), self.blocks.items(.cell_offset)) |bounds, *total, *offset| {
                    offset.* = cell_offset;
                    total.* = bounds.space().scale(tile_width).extendUniform(2 * O).total();
                    cell_offset += total.*;
                }

                self.tile_total = tile_offset;
                self.cell_total = cell_offset;

                var transfer_patch_offset: usize = 0;

                for (self.transfer_patches.items(.patch_total), self.transfer_patches.items(.patch_offset)) |total, *offset| {
                    offset.* = transfer_patch_offset;
                    transfer_patch_offset += total;
                }

                var transfer_block_offset: usize = 0;

                for (self.transfer_patches.items(.block_total), self.transfer_patches.items(.block_offset)) |total, *offset| {
                    offset.* = transfer_block_offset;
                    transfer_block_offset += total;
                }

                var transfer_tile_offset: usize = 0;

                for (self.transfer_patches.items(.bounds), self.transfer_patches.items(.tile_total), self.transfer_patches.items(.tile_offset)) |bounds, *total, *offset| {
                    offset.* = transfer_tile_offset;
                    total.* = bounds.space().total();
                    transfer_tile_offset += total.*;
                }

                var transfer_cell_offset: usize = 0;

                for (self.transfer_blocks.items(.bounds), self.transfer_blocks.items(.cell_total), self.transfer_blocks.items(.cell_offset)) |bounds, *total, *offset| {
                    offset.* = transfer_cell_offset;
                    total.* = bounds.space().scale(tile_width).extendUniform(2 * O).total();
                    transfer_cell_offset += total.*;
                }

                self.tile_total = transfer_tile_offset;
                self.cell_total = transfer_cell_offset;
            }

            /// Refines every patch and block on this level.
            fn refine(self: *Level) void {
                for (self.patches.items(.bounds)) |*bounds| {
                    bounds.refine();
                }

                for (self.blocks.items(.bounds)) |*bounds| {
                    bounds.refine();
                }

                for (self.transfer_blocks.items(.bounds)) |*bounds| {
                    bounds.refine();
                }

                for (self.transfer_patches.items(.bounds)) |*bounds| {
                    bounds.refine();
                }
            }
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
                .physical_bounds = config.physical_bounds,
                .index_size = config.index_size,
                .global_refinement = config.global_refinement,
                .tile_width = config.tile_width,
                .tile_total = base.tile_total,
                .cell_total = base.cell_total,
                .transfer_tile_total = base.tile_total,
                .transfer_cell_total = base.cell_total,
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

        pub fn transferTotal(self: *const Self) usize {
            return self.transfer_tile_total;
        }

        pub fn baseTileTotal(self: *const Self) usize {
            return self.base.tile_total;
        }

        pub fn baseCellTotal(self: *const Self) usize {
            return self.base.cell_total;
        }

        pub fn baseTransferTotal(self: *const Self) usize {
            return self.base.tile_total;
        }

        pub fn levelTileTotal(self: *const Self, level: usize) usize {
            return self.levels.items[level].tile_total;
        }

        pub fn levelCellTotal(self: *const Self, level: usize) usize {
            return self.levels.items[level].cell_total;
        }

        pub fn levelTransferTotal(self: *const Self, level: usize) usize {
            return self.levels.items[level].transfer_tile_total;
        }

        pub fn baseTileSlice(self: *const Self, comptime T: type, slice: []T) []T {
            return slice[0..self.baseTileTotal()];
        }

        pub fn baseCellSlice(self: *const Self, comptime T: type, slice: []T) []T {
            return slice[0..self.baseCellTotal()];
        }

        pub fn baseTransferSlice(self: *const Self, comptime T: type, slice: []T) []T {
            return slice[0..self.baseTransferTotal()];
        }

        pub fn levelTileSlice(self: *const Self, level: usize, comptime T: type, slice: []T) []T {
            const l: *const Level = &self.levels.items[level];
            return slice[l.tile_offset..(l.tile_offset + l.tile_total)];
        }

        pub fn levelCellSlice(self: *const Self, level: usize, comptime T: type, slice: []T) []T {
            const l: *const Level = &self.levels.items[level];
            return slice[l.cell_offset..(l.cell_offset + l.cell_total)];
        }

        pub fn levelTransferSlice(self: *const Self, level: usize, comptime T: type, slice: []T) []T {
            const l: *const Level = &self.levels.items[level];
            return slice[l.transfer_tile_offset..(l.transfer_tile_offset + l.transfer_tile_total)];
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

        // *************************
        // Fill Ghost **************
        // *************************

        pub fn fillGhostCells(self: *const Self, boundary: anytype, block_map: []const usize, field: []f64) !void {
            assert(block_map.len == self.tileTotal());
            assert(field.len == self.cellTotal());
            assert(boundaries.hasConditionDecl(N)(boundary));

            // Fill base first
            const base_field: []f64 = field[0..self.baseCellTotal()];
            try self.fillBaseGhostCells(base_field);

            if (self.active_levels > 0) {
                const target_field: []f64 = self.levelCellSlice(1, f64, field);
                try self.fillLevelGhostCells(base_field, boundary, target_field);
            }

            for (2..self.active_levels) |level_id| {
                const coarse_field: []f64 = self.levelCellSlice(level_id - 1, f64, field);
                const target_field: []f64 = self.levelCellSlice(level_id, f64, field);
                try self.fillLevelGhostCells(boundary, coarse_field, target_field);
            }
        }

        fn fillBaseGhostCells(self: *const Self, boundary: anytype, base_field: []f64) !void {
            const regions = Region.orderedRegions();

            const block: IndexBox = .{
                .origin = [1]usize{0} ** N,
                .size = self.base.index_size,
            };

            inline for (regions) |region| {
                try self.fillExteriorGhostCells(region, 0, block, boundary, base_field);
            }
        }

        fn fillLevelGhostCells(self: *const Self, boundary: anytype, coarse_field: []const f64, target_field: []f64) !void {
            _ = boundary;
            _ = target_field;
            _ = coarse_field;
            _ = self;
        }

        fn fillExteriorGhostCells(self: *const Self, comptime region: Region, level: usize, block: IndexBox, boundary: anytype, field: []f64) !void {
            const stencil_space: StencilSpace = self.computeStencilSpace(level, block);

            var inner_face_cells = region.innerFaceIndices(stencil_space.index_size);

            while (inner_face_cells.next()) |inner_face_cell| {
                var cell: [N]usize = inner_face_cell;

                for (0..N) |i| {
                    cell[i] -= O;
                }

                comptime var extent_indices = region.extentOffsets();

                inline while (extent_indices) |extents| {
                    var target: [N]usize = cell;

                    for (0..N) |i| {
                        target[i] += extents[i];
                    }

                    stencil_space.setValue(target, field, 0.0);

                    const position = stencil_space.boundaryPosition(extents, cell);

                    var value: f64 = 0.0;
                    var normals: [N]usize = undefined;
                    var rhs: f64 = 0.0;

                    for (0..N) |i| {
                        if (extents[i] != 0) {
                            const condition: BoundaryCondition = boundary.condition(position, Face{
                                .side = extents[i] > 0,
                                .axis = i,
                            });

                            value += condition.value;
                            normals[i] = condition.normal;
                            rhs += condition.rhs;
                        }
                    }

                    var sum = stencil_space.boundaryValue(extents, cell, field);
                    var coef: f64 = stencil_space.boundaryValueCoef(extents);

                    inline for (0..N) |i| {
                        if (extents[i] != 0) {
                            var ranks: [N]usize = [1]usize{0} ** N;
                            ranks[i] = 1;

                            sum += stencil_space.boundaryDerivative(ranks, extents, cell, field);
                            coef += stencil_space.boundaryDerivativeCoef(ranks, extents);
                        }
                    }

                    stencil_space.setValue(target, field, (rhs - sum) / coef);
                }
            }
        }

        fn computeStencilSpace(self: *const Self, level: usize, block: IndexBox) StencilSpace {
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
            self.active_levels = total_levels;
            try self.cacheLevels(total_levels);

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
                // - target is old (but children and parents have been updated)
                // - refined has been fully updated

                // To assemble clusters per patch we iterate children of coarse, then children of target

                clusters.shrinkRetainingCapacity(0);
                cluster_index_map.shrinkRetainingCapacity(0);
                cluster_offsets.shrinkRetainingCapacity(0);

                try cluster_offsets.append(self.gpa, 0);

                if (refined_exists and coarse_exists) {
                    const refined: *const Level = &self.levels.items[level_id + 1];
                    const coarse: *Level = &self.levels.items[level_id - 1];

                    for (0..coarse.patches.len) |cpid| {
                        for (coarse.childrenSlice(cpid)) |tpid| {
                            for (target.childrenSlice(tpid)) |child| {
                                var patch: IndexBox = refined.patches.items(.bounds)[child];

                                try patch.coarsen();
                                try patch.coarsen();

                                try clusters.append(self.gpa, patch);
                                try cluster_index_map.append(self.gpa, child);
                            }
                        }

                        try cluster_offsets.append(self.gpa, clusters.items.len);
                    }
                } else if (refined_exists) {
                    const refined: *const Level = &self.levels.items[level_id + 1];

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

                // Now we can clear the target arrays
                target.transfer_blocks.shrinkRetainingCapacity(0);
                target.transfer_patches.shrinkRetainingCapacity(0);

                if (coarse_exists) {
                    const coarse: *const Level = &self.levels.items[level_id - 1];

                    for (0..coarse.patches.len) |cpid| {
                        const upbounds: IndexBox = coarse.patches.items(.bounds)[cpid];

                        try target.transfer_patches.append(self.gpa, TransferPatch{
                            .bounds = upbounds,
                            .block_offset = 0,
                            .block_total = 0,
                            .patch_offset = 0,
                            .patch_total = 0,
                        });
                    }

                    for (target.blocks.items(.bounds), target.blocks.items(.patch)) |bounds, patch| {
                        const cpid: usize = coarse.parents.items[patch];
                        target.transfer_patches.items(.block_total)[cpid] += 1;

                        try target.transfer_blocks.append(self.gpa, TransferBlock{
                            .bounds = bounds,
                            .patch = cpid,
                        });
                    }
                } else {
                    try target.transfer_patches.append(self.gpa, TransferPatch{
                        .bounds = bbounds,
                        .block_offset = 0,
                        .block_total = 0,
                        .patch_offset = 0,
                        .patch_total = 0,
                    });

                    for (target.blocks.items(.bounds)) |bounds| {
                        target.transfer_patches.items(.block_total)[0] += 1;

                        try target.transfer_blocks.append(self.gpa, TransferBlock{
                            .bounds = bounds,
                            .patch = 0,
                        });
                    }
                }

                target.patches.shrinkRetainingCapacity(0);
                target.blocks.shrinkRetainingCapacity(0);
                target.children.shrinkRetainingCapacity(0);
                target.parents.shrinkRetainingCapacity(0);

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

                    ctags = self.levelTileSlice(level_id - 1, bool, tags);
                    clen = coarse.patches.len;
                    cbounds = coarse.patches.items(.bounds);
                    coffsets = coarse.patches.items(.tile_offset);
                } else {
                    // Otherwise use base data
                    ctags = self.baseTileSlice(bool, tags);
                    clen = 1;
                    cbounds = bbounds_slice;
                    coffsets = boffsets_slice;
                }

                // 3.3 Generate new patches.
                // *************************

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
                    preprocessTagsOnPatch(cptags, cpbounds, upclusters);

                    // Run partitioning algorithm
                    var partition_space = try PartitionSpace.init(scratch, cpbounds.size, upclusters);
                    defer partition_space.deinit();

                    try partition_space.build(cptags, config.patch_max_tiles, config.patch_efficiency);

                    // Add patches to target
                    const patch_children_offset: usize = target.children.items.len;

                    for (partition_space.parts.items(.bounds), partition_space.parts.items(.children_offset), partition_space.parts.items(.children_total)) |bounds, offset, total| {
                        // Compute refined bounds in global space.
                        var pbounds: IndexBox = undefined;

                        for (0..N) |axis| {
                            pbounds.origin[axis] = cpbounds.origin[axis] + bounds.origin[axis];
                            pbounds.size[axis] = bounds.size[axis];
                        }

                        // Append to patch list
                        try target.patches.append(self.gpa, Patch{
                            .bounds = pbounds,
                            .children_offset = patch_children_offset + offset,
                            .children_total = total,
                        });
                    }

                    // Append mapped children indices to buffer.
                    for (partition_space.children) |index| {
                        try target.children.append(self.gpa, upcluster_index_map[index]);
                    }

                    // Set parents to corret values
                    for (partition_space.parts.items(.children_offset), partition_space.parts.items(.children_total), 0..) |offset, total, i| {
                        for (target.children.items[patch_children_offset..][offset..(offset + total)]) |child| {
                            target.parents.items[child] = i;
                        }
                    }

                    try coarse_children.append(self.gpa, target.patches.len);

                    target.transfer_patches.items(.patch_total)[cpid] += partition_space.parts.len;
                }

                if (coarse_exists) {
                    const coarse: *Level = &self.levels.items[level_id - 1];

                    try coarse.children.resize(self.gpa, target.patches.len);
                    try coarse.parents.resize(self.gpa, target.patches.len);

                    for (0..target.patches.len) |i| {
                        coarse.children.items[i] = i;
                    }

                    for (0..clen) |cpid| {
                        const start = coarse_children.items[cpid];
                        const end = coarse_children.items[cpid + 1];

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
                for (0..clen) |cpid| {
                    // Aliases for underlying variables
                    const cpbounds: IndexBox = cbounds[cpid];
                    const cpoffset: usize = coffsets[cpid];
                    const cpspace: IndexSpace = cpbounds.space();
                    const cptags: []const bool = ctags[cpoffset..(cpoffset + cpspace.total())];

                    for (coarse_children.items[cpid]..coarse_children.items[cpid + 1]) |patch| {
                        // Reset arena for new frame.
                        defer _ = arena.reset(.retain_capacity);

                        const bounds: IndexBox = target.patches.items(.bounds)[patch];

                        const space: IndexSpace = bounds.space();

                        // Build patch tags
                        var ptags: []bool = try scratch.alloc(bool, space.total());
                        defer scratch.free(ptags);
                        // Set ptags using window of uptags
                        cpspace.fillWindow(bounds.relativeTo(cpbounds), bool, ptags, cptags);

                        // Run partitioning algorithm
                        var partition_space = try PartitionSpace.init(scratch, bounds.size, &[_]IndexBox{});
                        defer partition_space.deinit();

                        try partition_space.build(ptags, config.block_max_tiles, config.block_efficiency);

                        // For each resulting block
                        for (partition_space.parts.items(.bounds)) |pbounds| {
                            // Compute refined global bounds
                            var rbounds: IndexBox = undefined;

                            for (0..N) |axis| {
                                rbounds.origin[axis] = bounds.origin[axis] + pbounds.origin[axis];
                                rbounds.size[axis] = pbounds.size[axis];
                            }

                            // Add to list of blocks with appropriate patch id.
                            try target.blocks.append(self.gpa, Block{
                                .bounds = rbounds,
                                .patch = patch,
                            });
                        }
                    }
                }

                target.refine();

                // At this moment in time
                // - coarse is old (but heirarchy is updated)
                // - target has been fully updated
                // - refined has been fully updated
            }

            // 4. Recompute level offsets and totals.
            // **************************************

            self.computeOffsets();
        }

        fn preprocessTagsOnPatch(tags: []bool, bounds: IndexBox, clusters: []const IndexBox) void {
            for (clusters) |upcluster| {
                var cluster: IndexBox = upcluster;

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

        fn cacheLevels(self: *Self, total: usize) !void {
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
