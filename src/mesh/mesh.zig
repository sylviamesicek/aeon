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

const array = @import("../array.zig");
const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");

const boundaries = @import("boundary.zig");
const operator = @import("operator.zig");
const levels = @import("level.zig");

// Public Exports

pub const ApproxEngine = operator.ApproxEngine;
pub const BoundaryCondition = boundaries.BoundaryCondition;

pub const isInputStruct = operator.isInputStruct;
pub const isOperator = operator.isOperator;

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
        const InterpolationSpace = basis.InterpolationSpace(N, O);

        const Array = array.Array(N, usize);

        const add = Array.add;
        const sub = Array.sub;
        const scaled = Array.scaled;
        const splat = Array.splat;

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

        // ************************
        // Fill operation *********
        // ************************

        pub fn fillBoundaries(self: *const Self, boundary: anytype, block_map: []const usize, field: []f64) void {
            assert(block_map.len == self.tileTotal());
            assert(field.len == self.cellTotal());
            assert(boundaries.hasConditionDecl(N)(boundary));

            self.fillBaseBoundary(boundary, field);

            for (0..self.active_levels) |i| {
                self.fillLevelBoundaries(i, boundary, block_map, field);
            }
        }

        pub fn fillBaseBoundary(self: *const Self, boundary: anytype, field: []f64) void {
            const regions = Region.orderedRegions();

            const base_field: []f64 = self.baseCellSlice(field);
            const stencil_space = self.baseStencilSpace();

            inline for (regions) |region| {
                stencil_space.fillBoundary(region, boundary, base_field);
            }
        }

        pub fn fillLevelBoundaries(self: *const Self, level: usize, boundary: anytype, block_map: []const usize, field: []f64) void {
            const target: *const Level = self.levels[level];
            const index_size: IndexBox = target.index_size;

            const blocks = target.blocks.slice();

            for (0..blocks.len) |block| {
                const bounds: IndexBox = blocks.items(.bounds)[block];

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

        fn fillLevelInterior(self: *const Self, comptime region: Region, level: usize, block: usize, global_block_map: []const usize, global_field: []f64) void {
            // Cache target and target field
            const target: *const Level = &self.levels[level];
            const field: []f64 = target.levelCellSlice(global_field);
            const block_map: []const usize = target.levelTileSlice(global_block_map);

            const blocks = target.blocks.slice();
            const patches = target.patches.slice();

            // Get block bounds and containing patch
            const block_bounds: IndexBox = blocks.items(.bounds)[block];
            const block_field: []f64 = target.blockCellSlice(block, field);
            const block_interp: InterpolationSpace = InterpolationSpace.fromSize(block_bounds.size);

            const patch = blocks.items(.patch)[block];
            const patch_bounds: IndexBox = patches.items(.bounds)[patch];
            const patch_space = patch_bounds.space();
            const patch_block_map: []const usize = target.patchTileSlice(patch, block_map);

            const relative_bounds: IndexBox = block_bounds.relativeTo(patch_bounds);

            var tiles = region.innerFaceIndices(1, relative_bounds.size);

            while (tiles.next()) |tile| {
                const relative_tile: [N]usize = relative_bounds.globalFromLocal(tile);
                const buffer_tile: [N]usize = add(relative_tile, region.extentDir());

                const origin: [N]usize = scaled(tile, self.config.tile_width);
                const neighbor: usize = patch_block_map[patch_space.linearFromCartesian(buffer_tile)];

                if (neighbor == maxInt(usize)) {
                    var coarse_tile: [N]usize = undefined;
                    var coarse_buffer_tile: [N]usize = undefined;

                    for (0..N) |i| {
                        coarse_tile[i] = tile[i] / 2;
                        coarse_buffer_tile[i] = buffer_tile[i] / 2;
                    }

                    // Check if coarse exists
                    if (level != 0) {
                        // If so cache various coarse variables
                        const coarse: *const Level = &self.levels[level - 1];
                        const coarse_block_map: []const usize = coarse.levelTileSlice(block_map);
                        const coarse_field: []const f64 = coarse.levelCellSlice(global_field);

                        const coarse_patch = coarse.parents[patch];
                        const coarse_patch_block_map: []const usize = coarse.patchTileSlice(coarse_patch, coarse_block_map);
                        const coarse_patch_bounds = coarse.patches.items(.bounds)[coarse_patch];
                        const coarse_patch_space = coarse_patch_bounds.space();

                        const coarse_neighbor = coarse_patch_block_map[coarse_patch_space.linearFromCartesian(coarse_buffer_tile)];
                        const coarse_neighbor_interp: InterpolationSpace = InterpolationSpace.fromSize(coarse.blocks.items(.bounds)[coarse_neighbor].size);
                        const coarse_neighbor_field: []const f64 = coarse.blockCellSlice(coarse_neighbor, coarse_field);

                        var coarse_relative_neighbor_bounds = coarse.blocks.items(.bounds)[coarse_neighbor].relativeTo(coarse_patch_bounds);
                        coarse_relative_neighbor_bounds.refine();

                        // Neighbor origin in subcell space
                        const coarse_neighbor_origin: [N]usize = scaled(coarse_relative_neighbor_bounds.localFromGlobal(relative_tile), self.config.tile_width);

                        var indices = region.cartesianIndices(O, splat(self.config.tile_width));

                        while (indices.next()) |index| {
                            // Cell in subcell space
                            const block_cell: [N]usize = add(origin, index);
                            // Cell in neighbor in subcell space
                            const neighbor_cell: [N]usize = add(coarse_neighbor_origin, index);

                            block_interp.setValue(
                                block_cell,
                                block_field,
                                coarse_neighbor_interp.prolong(neighbor_cell, coarse_neighbor_field),
                            );
                        }
                    } else {
                        var base_bounds: IndexBox = .{
                            .origin = splat(0),
                            .size = self.base.index_size,
                        };

                        base_bounds.refine();

                        const base_origin: [N]usize = scaled(base_bounds.localFromGlobal(relative_tile), self.config.tile_width);

                        const base_field: []const f64 = self.baseCellSlice(field);
                        const base_interp: InterpolationSpace = InterpolationSpace.fromSize(self.base.index_size);

                        var indices = region.cartesianIndices(O, splat(self.config.tile_width));

                        while (indices.next()) |index| {
                            const block_cell: [N]usize = add(origin, index);
                            const base_cell: [N]usize = add(base_origin, index);

                            block_interp.setValue(
                                block_cell,
                                block_field,
                                base_interp.prolong(base_cell, base_field),
                            );
                        }
                    }
                } else {
                    // Copy from neighbor on same level
                    const neighbor_field: []const f64 = target.blockCellSlice(neighbor, field);
                    const neighbor_bounds: IndexBox = blocks.items(.bounds)[neighbor].relativeTo(patch_bounds);
                    const neighbor_interp: InterpolationSpace = InterpolationSpace.fromSize(blocks.items(.bounds)[neighbor].size);

                    const neighbor_origin: [N]usize = scaled(neighbor_bounds.localFromGlobal(relative_tile), self.config.tile_width);

                    var indices = region.cartesianIndices(O, splat(self.config.tile_width));

                    while (indices.next()) |index| {
                        const cell: [N]usize = add(origin, index);
                        const neighbor_cell: [N]usize = add(neighbor_origin, index);

                        block_interp.setValue(
                            cell,
                            block_field,
                            neighbor_interp.value(neighbor_cell, neighbor_field),
                        );
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

        fn baseStencilSpace(self: *const Self) StencilSpace {
            return .{
                .physical_bounds = self.physical_bounds,
                .index_size = scaled(self.base.index_size, self.config.tile_width),
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
                .size = scaled(block.size, self.config.tile_width),
            };
        }

        // *************************************
        // Restriction and Prolongation ********
        // *************************************

        pub fn restrict(self: *const Self, block_map: []const usize, field: []f64) void {
            assert(block_map.len == self.tileTotal());
            assert(field.len == self.cellTotal());

            self.restrictToBase(field);

            for (0..self.active_levels - 1) |level| {
                self.restrictToLevel(level, block_map, field);
            }
        }

        pub fn restrictToBase(self: *const Self, global_field: []f64) void {
            // A more refined level must exist for restriction
            if (self.active_levels == 0) {
                return;
            }

            // Get interpolation space for base
            const interp_space: InterpolationSpace = InterpolationSpace.fromSize(self.base.index_size);
            const field: []f64 = self.baseCellSlice(global_field);

            // Refined level
            const refined: *const Level = &self.levels[0];

            const refined_blocks = refined.blocks.slice();
            const refined_field: []const f64 = refined.levelCellSlice(global_field);

            for (0..refined_blocks.len) |refined_block| {
                // Get refined block bounds in supercell space.
                var refined_block_bounds: IndexBox = refined_blocks.items(.bounds)[refined_block];
                refined_block_bounds.coarsen();

                const refined_interp_space: InterpolationSpace = InterpolationSpace.fromSize(refined_blocks.items(.bounds)[refined_block].size);
                const refined_block_field: []const f64 = refined.blockCellSlice(refined_block, refined_field);

                // Origin of the refined block in supercell space
                const refined_origin: [N]usize = scaled(refined_block_bounds.origin, self.config.tile_width);

                // Iterate supercell space
                var indices = refined_block_bounds.space().cartesianIndices();

                while (indices.next()) |index| {
                    const supercell: [N]usize = add(refined_origin, index);

                    interp_space.setValue(
                        supercell,
                        field,
                        refined_interp_space.restrict(index, refined_block_field),
                    );
                }
            }
        }

        pub fn restrictToLevel(self: *const Self, level: usize, global_block_map: []const usize, global_field: []f64) void {
            // A more refined level must exist in order to perform restriction
            if (self.active_levels <= level + 1) {
                return;
            }

            // Cache target and refined variables
            const target: *const Level = &self.levels[level];
            const refined: *const Level = &self.levels[level + 1];

            const field: []f64 = target.levelCellSlice(global_field);
            const block_map: []const usize = target.levelTileSlice(global_block_map);

            const patches = target.patches.slice();
            const blocks = target.blocks.slice();

            const refined_field: []const f64 = refined.levelCellSlice(global_field);
            const refined_patches = refined.patches.slice();
            const refined_blocks = refined.blocks.slice();

            // Iterate over refined patches
            for (0..refined_patches.len) |refined_patch| {
                // Get target patch under each refined patch
                const patch: usize = target.parents.items[refined_patch];
                const patch_block_map: usize = target.patchTileSlice(patch, block_map);
                const patch_bounds: IndexBox = patches.items(.bounds)[patch];
                const patch_space: IndexSpace = patch_bounds.space();

                // Iterate blocks on refined patch
                const offset: usize = refined_patches.items(.block_offset)[refined_patch];
                const total: usize = refined_patches.items(.block_total)[refined_patch];

                for (offset..(offset + total)) |refined_block| {
                    // Get bounds of refined block in supercell space
                    var refined_block_bounds: IndexBox = refined_blocks.items(.block_bounds)[refined_block];
                    refined_block_bounds.coarsen();

                    // Compute bounds relatives to coarse patch bounds
                    const refined_relative_bounds: IndexBox = refined_block_bounds.relativeTo(patch_bounds);
                    const refined_block_space: IndexSpace = refined_relative_bounds.space();
                    const refined_interp_space: InterpolationSpace = InterpolationSpace.fromSize(refined_relative_bounds.size);
                    const refined_block_field: []const f64 = refined.blockCellSlice(refined_block, refined_field);

                    // Iterate tiles in refined block
                    var tiles = refined_block_space.cartesianIndices();

                    while (tiles.next()) |tile| {
                        // Tile in patch space
                        const relative_tile = refined_relative_bounds.globalFromLocal(tile);
                        // Origin of tile in supercell space
                        const refined_origin: [N]usize = scaled(tile, self.config.tile_width);
                        // Block under refined tile
                        const block = patch_block_map[patch_space.linearFromCartesian(relative_tile)];
                        const block_bounds: IndexBox = blocks.items(.bounds)[block];
                        const block_field: []f64 = target.blockCellSlice(block, field);
                        const interp_space: InterpolationSpace = InterpolationSpace.fromSize(block_bounds.size);

                        // Bounds of block relative to patch
                        const relative_bounds: IndexBox = block_bounds.relativeTo(patch_bounds);

                        // Origin of tile in supercell space on block
                        const origin: [N]usize = scaled(relative_bounds.localFromGlobal(relative_tile), self.config.tile_width);

                        // Iterate over supercells
                        var indices = IndexSpace.fromSize(splat(self.config.tile_width)).cartesianIndices();

                        while (indices.next()) |index| {
                            // Refined cell in supercell space
                            const refined_cell = add(refined_origin, index);
                            // Cell in supercell space
                            const cell = add(origin, index);

                            interp_space.setValue(
                                cell,
                                block_field,
                                refined_interp_space.restrict(refined_cell, refined_block_field),
                            );
                        }
                    }
                }
            }
        }

        pub fn prolong(self: *const Self, block_map: []const usize, field: []f64) void {
            assert(block_map.len == self.tileTotal());
            assert(field.len == self.cellTotal());

            self.prolongFromBase(field);

            for (0..self.active_levels - 1) |level| {
                self.prolongFromLevel(level, block_map, field);
            }
        }

        fn prolongFromBase(self: *const Self, field: []f64) void {
            if (self.active_levels == 0) {
                return;
            }

            const stencil_space: InterpolationSpace = InterpolationSpace.fromSize(self.base.index_size);

            const target: *const Level = &self.levels[0];
            const blocks = target.blocks.slice();

            const base_field: []const f64 = self.baseCellSlice(field);
            const level_field: []f64 = target.levelCellSlice(field);

            for (0..blocks.len) |b| {
                const block_bounds: IndexBox = blocks.items(.bounds)[b];
                const block_stencil_space: InterpolationSpace = InterpolationSpace.fromSize(block_bounds.size);
                const block_field: []const f64 = target.blockCellSlice(b, level_field);

                const origin: [N]usize = scaled(block_bounds.origin, self.config.tile_width);

                var indices = block_bounds.space().scale(self.config.tile_width).cartesianIndices();

                while (indices.next()) |index| {
                    const cell: [N]usize = add(origin, index);

                    block_stencil_space.setValue(index, block_field, stencil_space.prolong(cell, base_field));
                }
            }
        }

        fn prolongFromLevel(self: *const Self, level: usize, global_block_map: []const usize, global_field: []f64) void {
            // If a more refined level doesn't exist, then this function is done.
            if (self.active_levels <= level + 1) {
                return;
            }
            // Cache various variables
            const coarse: *const Level = &self.levels[level];
            const target: *const Level = &self.levels[level + 1];

            const coarse_field: []f64 = coarse.levelCellSlice(global_field);
            const target_field: []const f64 = target.levelCellSlice(global_field);

            const coarse_block_map: []const usize = coarse.levelTileSlice(global_block_map);
            const coarse_patches = coarse.patches.slice();
            const coarse_blocks = coarse.blocks.slice();

            const patches = target.patches.slice();
            const blocks = target.blocks.slice();

            // Iterate patches on refined level
            for (0..patches.len) |patch| {
                // Cache underlying patch data
                const coarse_patch: usize = coarse.parents.items[patch];
                const coarse_patch_block_map: usize = target.patchTileSlice(coarse_patch, coarse_block_map);
                const coarse_patch_bounds: IndexBox = coarse_patches.items(.bounds)[coarse_patch];
                const coarse_patch_space: IndexSpace = coarse_patch_bounds.space();

                const offset: usize = patches.items(.block_offset)[patch];
                const total: usize = patches.items(.block_total)[patch];

                // Iterate blocks on refined patch
                for (offset..(offset + total)) |block| {
                    // Get bounds and field on this block
                    const block_field: []f64 = target.blockCellSlice(block, target_field);
                    // Transform bounds into coarse level
                    var block_bounds: IndexBox = blocks.items(.block_bounds)[block];
                    block_bounds.coarsen();
                    // Compute interpolation space for this block
                    const interp_space: InterpolationSpace = InterpolationSpace.fromSize(blocks.items(.block_bounds)[block].size);

                    // Compute the bounds relative to the coarse patch
                    const relative_block_bounds = block_bounds.relativeTo(coarse_patch_bounds);

                    // Iterate tiles of block
                    var tiles = block_bounds.space().cartesianIndices();

                    while (tiles.next()) |tile| {
                        // Get coarse block under this tile
                        const relative_tile: [N]usize = relative_block_bounds.globalFromLocal(tile);
                        const coarse_block = coarse_patch_block_map[coarse_patch_space.linearFromCartesian(relative_tile)];
                        const coarse_block_bounds: IndexBox = coarse_blocks.items(.bounds)[coarse_block];
                        const coarse_relative_block_bounds: IndexBox = coarse_block_bounds.relativeTo(coarse_patch_bounds);
                        const coarse_interp_space: InterpolationSpace = InterpolationSpace.fromSize(coarse_relative_block_bounds.size);
                        const coarse_block_field: []const f64 = coarse.blockCellSlice(coarse_block, coarse_field);

                        // Get coarse origin (in subcell space)
                        const coarse_origin = scaled(coarse_relative_block_bounds.globalFromLocal(tile), 2 * self.config.tile_width);

                        // Get origin (in subcell space)
                        const origin = scaled(tile, 2 * self.config.tile_width);

                        // Iterate subcell space
                        var indices = IndexSpace.fromSize(splat(2 * self.config.tile_width)).cartesianIndices();

                        while (indices.next()) |index| {
                            // Get coarse subcell
                            const coarse_cell = add(coarse_origin, index);
                            // Get target subcell
                            const cell = add(origin, index);

                            // Prolong
                            interp_space.setValue(
                                cell,
                                block_field,
                                coarse_interp_space.prolong(coarse_cell, coarse_block_field),
                            );
                        }
                    }
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

                        try ppartitioner.build(ptags, config.block_max_tiles, config.block_efficiency);

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
            var tile_offset: usize = self.base.tile_total;
            var cell_offset: usize = self.base.cell_total;

            for (self.levels.items) |*level| {
                level.computeOffsets(self.config.tile_width);

                level.tile_offset = tile_offset;
                level.cell_offset = cell_offset;

                tile_offset += level.tile_total;
                cell_offset += level.cell_total;
            }

            self.cell_total = cell_offset;
            self.tile_total = tile_offset;
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
