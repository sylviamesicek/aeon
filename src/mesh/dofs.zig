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

// root imports

const basis = @import("../basis/basis.zig");
const geometry = @import("../geometry/geometry.zig");
const lac = @import("../lac/lac.zig");
const system = @import("../system.zig");

/// Provides an interface for interacting with degrees of freedom defined on a mesh.
/// This includes all fields which are represented as slices of f64 (not full systems).
/// This type provides support for restriction, prolongation, filling boundaries, etc.
pub fn DofHandler(comptime N: usize, comptime O: usize) type {
    return struct {
        mesh: *const Mesh,

        // Aliases
        const Self = @This();
        const Mesh = @import("mesh.zig").Mesh(N, O);
        const Level = Mesh.Level;
        const IndexBox = geometry.Box(N, usize);
        const RealBox = geometry.Box(N, f64);
        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const PartitionSpace = geometry.PartitionSpace(N);
        const Region = geometry.Region(N);
        const CellSpace = basis.CellSpace(N, O);
        const StencilSpace = basis.StencilSpace(N, O);

        // Mixins
        const Index = @import("../index.zig").Index(N);

        const add = Index.add;
        const sub = Index.sub;
        const scaled = Index.scaled;
        const splat = Index.splat;
        const toSigned = Index.toSigned;

        // ************************
        // Fill operations ********
        // ************************

        pub fn fillBoundaries(self: *const Self, comptime full: bool, boundary: anytype, block_map: []const usize, field: []f64) void {
            const mesh = self.mesh;

            assert(block_map.len == mesh.tileTotal());
            assert(field.len == mesh.cellTotal());

            self.fillBaseBoundary(boundary, field);

            for (0..mesh.active_levels) |i| {
                self.fillLevelBoundary(full, i, boundary, block_map, field, full);
            }
        }

        pub fn fillBaseBoundary(self: *const Self, boundary: anytype, field: []f64) void {
            const mesh = self.mesh;

            const regions = Region.orderedRegions();

            const base_field: []f64 = mesh.baseCellSlice(field);
            const stencil_space = mesh.baseStencilSpace();

            inline for (regions) |region| {
                stencil_space.fillBoundary(region, O, boundary, base_field);
            }
        }

        pub fn fillLevelBoundary(self: *const Self, comptime full: bool, level: usize, boundary: anytype, block_map: []const usize, field: []f64) void {
            const mesh = self.mesh;

            const target: *const Level = &mesh.levels.items[level];
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
                        self.fillLevelInterior(region, full, level, block, block_map, field);
                    } else {
                        self.fillLevelExterior(region, full, level, block, boundary, field);
                    }
                }
            }
        }

        fn fillLevelInterior(self: *const Self, comptime region: Region, comptime full: bool, level: usize, block: usize, global_block_map: []const usize, global_field: []f64) void {
            const E = if (full) 2 * O else O;

            const mesh = self.mesh;

            // Cache target and target field
            const target: *const Level = &mesh.levels[level];
            const field: []f64 = target.levelCellSlice(global_field);
            const block_map: []const usize = target.levelTileSlice(global_block_map);

            const blocks = target.blocks.slice();
            const patches = target.patches.slice();

            // Get block bounds and containing patch
            const block_bounds: IndexBox = blocks.items(.bounds)[block];
            const block_field: []f64 = target.blockCellSlice(block, field);
            const block_cell_space: CellSpace = CellSpace.fromSize(block_bounds.size);

            const patch = blocks.items(.patch)[block];
            const patch_bounds: IndexBox = patches.items(.bounds)[patch];
            const patch_space = patch_bounds.space();
            const patch_block_map: []const usize = target.patchTileSlice(patch, block_map);

            const relative_bounds: IndexBox = block_bounds.relativeTo(patch_bounds);

            var tiles = region.innerFaceIndices(1, relative_bounds.size);

            while (tiles.next()) |tile| {
                const relative_tile: [N]usize = relative_bounds.globalFromLocal(tile);
                const buffer_tile: [N]usize = add(relative_tile, region.extentDir());

                const origin: [N]usize = scaled(tile, mesh.config.tile_width);
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
                        const coarse: *const Level = &mesh.levels[level - 1];
                        const coarse_block_map: []const usize = coarse.levelTileSlice(block_map);
                        const coarse_field: []const f64 = coarse.levelCellSlice(global_field);

                        const coarse_patch = coarse.parents[patch];
                        const coarse_patch_block_map: []const usize = coarse.patchTileSlice(coarse_patch, coarse_block_map);
                        const coarse_patch_bounds = coarse.patches.items(.bounds)[coarse_patch];
                        const coarse_patch_space = coarse_patch_bounds.space();

                        const coarse_neighbor = coarse_patch_block_map[coarse_patch_space.linearFromCartesian(coarse_buffer_tile)];
                        const coarse_neighbor_cell_space: CellSpace = CellSpace.fromSize(coarse.blocks.items(.bounds)[coarse_neighbor].size);
                        const coarse_neighbor_field: []const f64 = coarse.blockCellSlice(coarse_neighbor, coarse_field);

                        var coarse_relative_neighbor_bounds = coarse.blocks.items(.bounds)[coarse_neighbor].relativeTo(coarse_patch_bounds);
                        coarse_relative_neighbor_bounds.refine();

                        // Neighbor origin in subcell space
                        const coarse_neighbor_origin: [N]usize = scaled(coarse_relative_neighbor_bounds.localFromGlobal(relative_tile), mesh.config.tile_width);

                        var indices = region.cartesianIndices(E, splat(mesh.config.tile_width));

                        while (indices.next()) |index| {
                            // Cell in subcell space
                            const block_cell: [N]isize = CellSpace.offsetFromOrigin(origin, index);
                            // Cell in neighbor in subcell space
                            const neighbor_cell: [N]isize = CellSpace.offsetFromOrigin(coarse_neighbor_origin, index);

                            block_cell_space.setValue(
                                block_cell,
                                block_field,
                                coarse_neighbor_cell_space.prolong(neighbor_cell, coarse_neighbor_field),
                            );
                        }
                    } else {
                        var base_bounds: IndexBox = .{
                            .origin = splat(0),
                            .size = mesh.base.index_size,
                        };

                        base_bounds.refine();

                        const base_origin: [N]usize = scaled(base_bounds.localFromGlobal(relative_tile), mesh.config.tile_width);

                        const base_field: []const f64 = mesh.baseCellSlice(field);
                        const base_cell_space: CellSpace = CellSpace.fromSize(mesh.base.index_size);

                        var indices = region.cartesianIndices(E, splat(mesh.config.tile_width));

                        while (indices.next()) |index| {
                            const block_cell: [N]isize = CellSpace.offsetFromOrigin(origin, index);
                            const base_cell: [N]isize = CellSpace.offsetFromOrigin(base_origin, index);

                            block_cell_space.setValue(
                                block_cell,
                                block_field,
                                base_cell_space.prolong(base_cell, base_field),
                            );
                        }
                    }
                } else {
                    // Copy from neighbor on same level
                    const neighbor_field: []const f64 = target.blockCellSlice(neighbor, field);
                    const neighbor_bounds: IndexBox = blocks.items(.bounds)[neighbor].relativeTo(patch_bounds);
                    const neighbor_cell_space: CellSpace = CellSpace.fromSize(blocks.items(.bounds)[neighbor].size);

                    const neighbor_origin: [N]usize = scaled(neighbor_bounds.localFromGlobal(relative_tile), mesh.config.tile_width);

                    var indices = region.cartesianIndices(E, splat(mesh.config.tile_width));

                    while (indices.next()) |index| {
                        const cell: [N]isize = CellSpace.offsetFromOrigin(origin, index);
                        const neighbor_cell: [N]isize = CellSpace.offsetFromOrigin(neighbor_origin, index);

                        block_cell_space.setValue(
                            cell,
                            block_field,
                            neighbor_cell_space.value(neighbor_cell, neighbor_field),
                        );
                    }
                }
            }
        }

        fn fillLevelExterior(self: *const Self, comptime region: Region, comptime full: bool, level: usize, block: usize, boundary: anytype, field: []f64) void {
            const mesh = self.mesh;
            const target: *const Level = &mesh.levels.items[level];

            const level_field: []f64 = target.levelCellSlice(field);
            const block_field: []f64 = target.blockCellSlice(block, level_field);

            const stencil_space = mesh.levelStencilSpace(level, block);
            const E = if (full) 2 * O else O;

            stencil_space.fillBoundary(region, E, boundary, block_field);
        }

        // *************************************
        // Restriction and Prolongation ********
        // *************************************

        pub fn restrict(self: *const Self, block_map: []const usize, field: []f64) void {
            const mesh = self.mesh;

            assert(block_map.len == mesh.tileTotal());
            assert(field.len == mesh.cellTotal());

            self.restrictToBase(field);

            for (0..self.active_levels - 1) |level| {
                self.restrictToLevel(level, block_map, field);
            }
        }

        pub fn restrictToBase(self: *const Self, global_field: []f64) void {
            const mesh = self.mesh;

            // A more refined level must exist for restriction
            if (mesh.active_levels == 0) {
                return;
            }

            // Get interpolation space for base
            const cell_space: CellSpace = CellSpace.fromSize(mesh.base.index_size);
            const field: []f64 = mesh.baseCellSlice(global_field);

            // Refined level
            const refined: *const Level = &mesh.levels[0];

            const refined_blocks = refined.blocks.slice();
            const refined_field: []const f64 = refined.levelCellSlice(global_field);

            for (0..refined_blocks.len) |refined_block| {
                // Get refined block bounds in supercell space.
                var refined_block_bounds: IndexBox = refined_blocks.items(.bounds)[refined_block];
                refined_block_bounds.coarsen();

                const refined_cell_space: CellSpace = CellSpace.fromSize(refined_blocks.items(.bounds)[refined_block].size);
                const refined_block_field: []const f64 = refined.blockCellSlice(refined_block, refined_field);

                // Origin of the refined block in supercell space
                const refined_origin: [N]usize = scaled(refined_block_bounds.origin, mesh.config.tile_width);

                // Iterate supercell space
                var indices = refined_block_bounds.space().cartesianIndices();

                while (indices.next()) |index| {
                    const supercell: [N]isize = toSigned(add(refined_origin, index));

                    cell_space.setValue(
                        supercell,
                        field,
                        refined_cell_space.restrict(index, refined_block_field),
                    );
                }
            }
        }

        pub fn restrictToLevel(self: *const Self, level: usize, global_block_map: []const usize, global_field: []f64) void {
            const mesh = self.mesh;

            // A more refined level must exist in order to perform restriction
            if (mesh.active_levels <= level + 1) {
                return;
            }

            // Cache target and refined variables
            const target: *const Level = &mesh.levels[level];
            const refined: *const Level = &mesh.levels[level + 1];

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
                    const refined_cell_space: CellSpace = CellSpace.fromSize(refined_relative_bounds.size);
                    const refined_block_field: []const f64 = refined.blockCellSlice(refined_block, refined_field);

                    // Iterate tiles in refined block
                    var tiles = refined_block_space.cartesianIndices();

                    while (tiles.next()) |tile| {
                        // Tile in patch space
                        const relative_tile = refined_relative_bounds.globalFromLocal(tile);
                        // Origin of tile in supercell space
                        const refined_origin: [N]usize = scaled(tile, mesh.config.tile_width);
                        // Block under refined tile
                        const block = patch_block_map[patch_space.linearFromCartesian(relative_tile)];
                        const block_bounds: IndexBox = blocks.items(.bounds)[block];
                        const block_field: []f64 = target.blockCellSlice(block, field);
                        const block_cell_space: CellSpace = CellSpace.fromSize(block_bounds.size);

                        // Bounds of block relative to patch
                        const relative_bounds: IndexBox = block_bounds.relativeTo(patch_bounds);

                        // Origin of tile in supercell space on block
                        const origin: [N]usize = scaled(relative_bounds.localFromGlobal(relative_tile), mesh.config.tile_width);

                        // Iterate over supercells
                        var indices = IndexSpace.fromSize(splat(mesh.config.tile_width)).cartesianIndices();

                        while (indices.next()) |index| {
                            // Refined cell in supercell space
                            const refined_cell = toSigned(add(refined_origin, index));
                            // Cell in supercell space
                            const cell = toSigned(add(origin, index));

                            block_cell_space.setValue(
                                cell,
                                block_field,
                                refined_cell_space.restrict(refined_cell, refined_block_field),
                            );
                        }
                    }
                }
            }
        }

        pub fn prolong(self: *const Self, block_map: []const usize, field: []f64) void {
            const mesh = self.mesh;

            assert(block_map.len == mesh.tileTotal());
            assert(field.len == mesh.cellTotal());

            self.prolongFromBase(field);

            for (0..self.active_levels - 1) |level| {
                self.prolongFromLevel(level, block_map, field);
            }
        }

        pub fn prolongFromBase(self: *const Self, field: []f64) void {
            const mesh = self.mesh;

            if (mesh.active_levels == 0) {
                return;
            }

            const cell_space: CellSpace = CellSpace.fromSize(mesh.base.index_size);

            const target: *const Level = &mesh.levels.items[0];
            const blocks = target.blocks.slice();

            const base_field: []const f64 = mesh.baseCellSlice(field);
            const level_field: []f64 = target.levelCellSlice(field);

            for (0..blocks.len) |b| {
                const block_bounds: IndexBox = blocks.items(.bounds)[b];
                const block_cell_space: CellSpace = CellSpace.fromSize(block_bounds.size);
                const block_field: []const f64 = target.blockCellSlice(b, level_field);

                const origin: [N]usize = scaled(block_bounds.origin, mesh.config.tile_width);

                var indices = block_bounds.space().scale(mesh.config.tile_width).cartesianIndices();

                while (indices.next()) |index| {
                    const cell = toSigned(add(origin, index));

                    block_cell_space.setValue(toSigned(index), block_field, cell_space.prolong(cell, base_field));
                }
            }
        }

        pub fn prolongFromLevel(self: *const Self, level: usize, global_block_map: []const usize, global_field: []f64) void {
            const mesh = self.mesh;

            // If a more refined level doesn't exist, then this function is done.
            if (mesh.active_levels <= level + 1) {
                return;
            }
            // Cache various variables
            const coarse: *const Level = &mesh.levels.items[level];
            const target: *const Level = &mesh.levels.items[level + 1];

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
                    const cell_space: CellSpace = CellSpace.fromSize(blocks.items(.block_bounds)[block].size);

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
                        const coarse_cell_space: CellSpace = CellSpace.fromSize(coarse_relative_block_bounds.size);
                        const coarse_block_field: []const f64 = coarse.blockCellSlice(coarse_block, coarse_field);

                        // Get coarse origin (in subcell space)
                        const coarse_origin = scaled(coarse_relative_block_bounds.globalFromLocal(tile), 2 * mesh.config.tile_width);

                        // Get origin (in subcell space)
                        const origin = scaled(tile, 2 * mesh.config.tile_width);

                        // Iterate subcell space
                        var indices = IndexSpace.fromSize(splat(2 * mesh.config.tile_width)).cartesianIndices();

                        while (indices.next()) |index| {
                            // Get coarse subcell
                            const coarse_cell = toSigned(add(coarse_origin, index));
                            // Get target subcell
                            const cell = toSigned(add(origin, index));

                            // Prolong
                            cell_space.setValue(
                                cell,
                                block_field,
                                coarse_cell_space.prolong(coarse_cell, coarse_block_field),
                            );
                        }
                    }
                }
            }
        }
    };
}
