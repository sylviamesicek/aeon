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
const index = @import("../index.zig");
const mesh = @import("../mesh/mesh.zig");
const system = @import("../system.zig");

// submodules

const boundary = @import("boundary.zig");
const multigrid = @import("multigrid.zig");
const operator = @import("operator.zig");

// Public Exports

pub const BoundaryCondition = boundary.BoundaryCondition;
pub const SystemBoundaryCondition = boundary.SystemBoundaryCondition;
pub const isSystemBoundaryCondition = boundary.isSystemBoundaryCondition;

pub const Engine = operator.Engine;
pub const OperatorEngine = operator.OperatorEngine;
pub const FunctionEngine = operator.FunctionEngine;
pub const EngineType = operator.EngineType;
pub const isMeshFunction = operator.isMeshFunction;
pub const isMeshOperator = operator.isMeshOperator;
pub const isMeshBoundary = operator.isMeshBoundary;

pub const MultigridSolver = multigrid.MultigridSolver;

// *************************
// Dof Handler *************
// *************************

/// Manages dofs defined on a mesh.
pub fn DofHandler(comptime N: usize, comptime O: usize) type {
    return struct {
        gpa: Allocator,
        mesh: *const Mesh,

        const Self = @This();
        const Face = geometry.Face(N);
        const IndexBox = geometry.Box(N, usize);
        const IndexSpace = geometry.IndexSpace(N);
        const Level = Mesh.Level;
        const Mesh = mesh.Mesh(N, O);
        const Region = geometry.Region(N);
        const CellSpace = basis.CellSpace(N, O);
        const StencilSpace = basis.StencilSpace(N, O);

        const Index = index.Index(N);
        const add = Index.add;
        const scaled = Index.scaled;
        const splat = Index.splat;
        const toSigned = Index.toSigned;

        /// Initialises a dof handler
        pub fn init(allocator: Allocator, grid: *const Mesh) Self {
            return .{
                .gpa = allocator,
                .mesh = grid,
            };
        }

        /// Deinit procedure for backwards capatability.
        pub fn deinit(self: *Self) void {
            _ = self;
        }

        // *********************
        // Fill operations *****
        // *********************

        pub fn fillBoundary(
            self: *const Self,
            comptime full: bool,
            block_map: []const usize,
            bound: anytype,
            sys: system.SystemSlice(@TypeOf(bound).System),
        ) void {
            _ = sys;
            const T = @TypeOf(bound);

            if (comptime !isMeshBoundary(N)(T)) {
                @compileError("Bound type must satisfy the isMeshBoundary trait.");
            }

            assert(block_map.len == self.mesh.tile_total);

            self.fillBaseBoundary(bound, system);

            _ = full;
        }

        pub fn fillBaseBoundary(
            self: *const Self,
            bound: anytype,
            sys: system.SystemSlice(@TypeOf(bound).System),
        ) void {
            const base_sys = system.systemStructSlice(sys, 0, self.mesh.base.cell_total);

            const stencil_space = self.mesh.baseStencilSpace();
            const regions = comptime Region.orderedRegions();

            inline for (comptime regions[1..]) |region| {
                fillBoundaryRegion(region, false, stencil_space, bound, base_sys);
            }
        }

        pub fn fillLevelBoundary(
            self: *const Self,
            comptime full: bool,
            level: usize,
            block_map: []const usize,
            bound: anytype,
            sys: system.SystemSlice(@TypeOf(bound).System),
        ) void {
            const target: *const Level = self.mesh.getLevel(level);
            const index_size = target.index_size;

            const blocks = target.blocks.slice();

            for (0..blocks.len) |block| {
                const bounds: IndexBox = blocks.items(.bounds)[block];

                const regions = comptime Region.orderedRegions();

                inline for (comptime regions[1..]) |region| {
                    var exterior: bool = false;

                    for (0..N) |i| {
                        if (region.sides[i] == .left and bounds.origin[i] == 0) {
                            exterior = true;
                        } else if (region.sides[i] == .right and bounds.origin[i] + bounds.size[i] == index_size[i]) {
                            exterior == true;
                        }
                    }

                    if (exterior) {
                        self.fillBlockBoundary(
                            region,
                            full,
                            level,
                            block,
                            block_map,
                            sys,
                        );
                    } else {
                        const level_offset = target.cell_offset + blocks.items(.cell_offset)[block];
                        const level_total = blocks.items(.cell_total)[block];
                        const level_sys = system.systemStructSlice(sys, level_offset, level_total);
                        const level_stencil_space = self.mesh.levelStencilSpace(level, block);

                        fillBoundaryRegion(
                            region,
                            full,
                            level_stencil_space,
                            bound,
                            level_sys,
                        );
                    }
                }
            }
        }

        fn fillBlockBoundary(
            self: *const Self,
            comptime region: Region,
            comptime full: bool,
            level: usize,
            block: usize,
            block_map: []const usize,
            sys: anytype,
        ) void {
            const target: *const Level = self.mesh.getLevel(level);

            const block_sys = system.systemStructSlice(
                sys,
                target.blockCellOffset(block),
                target.blockCellTotal(block),
            );

            const blocks = target.blocks.slice();
            const patches = target.patches.slice();

            const block_bounds = block.items(.bounds)[block];
            const block_cell_space = CellSpace.fromSize(block_bounds.size);

            const patch = blocks.items(.patch)[blocks];
            const patch_bounds: IndexBox = patches.items(.bounds)[patch];
            const patch_space = IndexSpace.fromBox(patch_bounds);
            const patch_block_map: []const usize = self.mesh.levelTileSlice(level, patch, block_map);

            const relative_bounds: IndexBox = block_bounds.relativeTo(patch_bounds);

            var tiles = region.innerFaceIndices(1, relative_bounds.size);

            while (tiles.next()) |tile_signed| {
                var tile: [N]usize = undefined;

                for (0..N) |i| {
                    tile[i] = @intCast(tile_signed[i]);
                }

                const relative_tile: [N]usize = relative_bounds.globalFromLocal(tile);
                const buffer_tile: [N]usize = add(relative_tile, region.extentDir());

                const origin: [N]usize = scaled(tile, self.mesh.tile_width);
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
                        const coarse: *const Level = &self.mesh.getLevel(level - 1);

                        const coarse_patch = coarse.parents[patch];
                        const coarse_patch_block_map: []const usize = self.mesh.levelTileSlice(level - 1, coarse_patch, block_map);
                        const coarse_patch_bounds = coarse.patches.items(.bounds)[coarse_patch];
                        const coarse_patch_space = coarse_patch_bounds.space();

                        const coarse_block = coarse_patch_block_map[coarse_patch_space.linearFromCartesian(coarse_buffer_tile)];
                        const coarse_block_cell_space: CellSpace = CellSpace.fromSize(coarse.blocks.items(.bounds)[coarse_block].size);

                        const coarse_block_offset = coarse.cell_offset + coarse.blocks.items(.cell_offset)[coarse_block];
                        const coarse_block_total = target.blocks.items(.cell_total)[coarse_block];
                        const coarse_block_sys = system.systemStructSlice(sys, coarse_block_offset, coarse_block_total);

                        var coarse_relative_block_bounds = coarse.blocks.items(.bounds)[coarse_block].relativeTo(coarse_patch_bounds);
                        coarse_relative_block_bounds.refine();

                        // Neighbor origin in subcell space
                        const coarse_neighbor_origin: [N]usize = scaled(coarse_relative_block_bounds.localFromGlobal(relative_tile), self.mesh.tile_width);

                        var indices = region.cartesianIndices(if (full) 2 * O else O, splat(self.mesh.tile_width));

                        while (indices.next()) |ind| {
                            // Cell in subcell space
                            const block_cell: [N]isize = CellSpace.offsetFromOrigin(origin, ind);
                            // Cell in neighbor in subcell space
                            const neighbor_cell: [N]isize = CellSpace.offsetFromOrigin(coarse_neighbor_origin, ind);

                            inline for (comptime system.systemFieldNames(@TypeOf(sys))) |name| {
                                block_cell_space.setValue(
                                    block_cell,
                                    @field(block_sys, name),
                                    coarse_block_cell_space.prolong(
                                        neighbor_cell,
                                        @field(coarse_block_sys, name),
                                    ),
                                );
                            }
                        }
                    } else {
                        var base_bounds: IndexBox = .{
                            .origin = splat(0),
                            .size = self.mesh.base.index_size,
                        };

                        base_bounds.refine();

                        const base_origin: [N]usize = scaled(base_bounds.localFromGlobal(relative_tile), self.mesh.tile_width);
                        const base_cell_space: CellSpace = CellSpace.fromSize(self.mesh.base.index_size);
                        const base_sys = system.systemStructSlice(sys, 0, self.mesh.base.tile_total);

                        var indices = region.cartesianIndices(if (full) 2 * O else O, splat(self.mesh.tile_width));

                        while (indices.next()) |idx| {
                            const block_cell: [N]isize = CellSpace.offsetFromOrigin(origin, idx);
                            const base_cell: [N]isize = CellSpace.offsetFromOrigin(base_origin, idx);

                            inline for (comptime system.systemFieldNames(@TypeOf(sys))) |name| {
                                block_cell_space.setValue(
                                    block_cell,
                                    @field(block_sys, name),
                                    base_cell_space.prolong(
                                        base_cell,
                                        @field(base_sys, name),
                                    ),
                                );
                            }
                        }
                    }
                } else {
                    // Copy from neighbor on same level
                    const neighbor_sys = system.systemStructSlice(
                        sys,
                        target.blockCellOffset(neighbor),
                        target.blockCellTotal(neighbor),
                    );
                    const neighbor_bounds: IndexBox = blocks.items(.bounds)[neighbor].relativeTo(patch_bounds);
                    const neighbor_cell_space: CellSpace = CellSpace.fromSize(blocks.items(.bounds)[neighbor].size);

                    const neighbor_origin: [N]usize = scaled(neighbor_bounds.localFromGlobal(relative_tile), self.mesh.tile_width);

                    var indices = region.cartesianIndices(if (full) 2 * O else O, splat(self.mesh.tile_width));

                    while (indices.next()) |idx| {
                        const block_cell: [N]isize = CellSpace.offsetFromOrigin(origin, idx);
                        const neighbor_cell: [N]isize = CellSpace.offsetFromOrigin(neighbor_origin, idx);

                        inline for (comptime system.systemFieldNames(@TypeOf(sys))) |name| {
                            block_cell_space.setValue(
                                block_cell,
                                @field(block_sys, name),
                                neighbor_cell_space.prolong(
                                    neighbor_cell,
                                    @field(neighbor_sys, name),
                                ),
                            );
                        }
                    }
                }
            }
        }

        pub fn fillBoundaryRegion(
            comptime region: Region,
            comptime full: bool,
            stencil_space: StencilSpace,
            bound: anytype,
            sys: system.SystemSlice(@TypeOf(bound).System),
        ) void {
            const T = @TypeOf(bound);

            if (comptime !isMeshBoundary(N)(T)) {
                @compileError("Bound must satisfy the isMeshBoundary trait.");
            }

            var inner_face_cells = region.innerFaceIndices(stencil_space.size);

            while (inner_face_cells.next()) |cell| {
                comptime var extent_indices = region.extentOffsets(if (full) 2 * O else O);

                inline while (comptime extent_indices.next()) |extents| {
                    // Compute target cell for the given extent indices
                    var target: [N]isize = undefined;

                    inline for (0..N) |i| {
                        target[i] = cell[i] + extents[i];
                    }

                    // Compute position of boundary given extents
                    const pos: [N]f64 = stencil_space.boundaryPosition(extents, cell);

                    // Compute conditions for the given field.
                    var conditions: [N]SystemBoundaryCondition(T.System) = undefined;

                    inline for (0..N) |i| {
                        if (extents[i] != 0) {
                            conditions[i] = bound.condition(pos, Face{
                                .side = extents[i] > 0,
                                .axis = i,
                            });
                        }
                    }

                    inline for (comptime system.systemFieldNames(T.System)) |name| {
                        // Set the fields of the system to be zero at the target.
                        stencil_space.cellSpace().setValue(target, @field(sys, name), 0.0);

                        var v: f64 = 0.0;
                        var normals: [N]f64 = undefined;
                        var rhs: f64 = 0.0;

                        for (0..N) |i| {
                            if (extents[i] != 0) {
                                const condition: BoundaryCondition = @field(conditions[i], name);

                                v += condition.value;
                                normals[i] = if (extents[i] > 0) condition.normal else -condition.normal;
                                rhs += condition.rhs;
                            }
                        }

                        var sum: f64 = v * stencil_space.boundaryValue(
                            extents,
                            cell,
                            @field(sys, name),
                        );
                        var coef: f64 = v * stencil_space.boundaryValueCoef(extents);

                        inline for (0..N) |i| {
                            if (extents[i] != 0) {
                                comptime var ranks: [N]usize = [1]usize{0} ** N;
                                ranks[i] = 1;

                                sum += normals[i] * stencil_space.boundaryDerivative(ranks, extents, cell, @field(sys, name));
                                coef += normals[i] * stencil_space.boundaryDerivativeCoef(ranks, extents);
                            }
                        }

                        stencil_space.cellSpace().setValue(target, @field(sys, name), (rhs - sum) / coef);
                    }
                }
            }
        }

        // *************************************
        // Restriction and Prolongation ********
        // *************************************

        pub fn restrict(
            self: *const Self,
            block_map: []const usize,
            sys: anytype,
        ) void {
            if (comptime !system.isSystemSlice(sys)) {
                @compileError("System must satisfy isSystemSlice trait.");
            }

            assert(block_map.len == self.mesh.tileTotal());

            self.restrictToBase(sys);

            for (0..self.mesh.active_levels - 1) |level| {
                self.restrictToLevel(level, block_map, sys);
            }
        }

        pub fn restrictToBase(self: *const Self, sys: anytype) void {
            // A more refined level must exist for restriction
            if (self.mesh.active_levels == 0) {
                return;
            }

            // Get interpolation space for base
            const cell_space: CellSpace = CellSpace.fromSize(self.mesh.base.index_size);
            const base_sys = system.systemStructSlice(sys, 0, self.mesh.cell_total);

            // Refined level
            const refined: *const Level = self.mesh.getLevel(0);
            const refined_blocks = refined.blocks.slice();

            for (0..refined_blocks.len) |refined_block| {
                // Get refined block bounds in supercell space.
                var refined_block_bounds: IndexBox = refined_blocks.items(.bounds)[refined_block];
                refined_block_bounds.coarsen();

                const refined_cell_space: CellSpace = CellSpace.fromSize(refined_blocks.items(.bounds)[refined_block].size);
                const refined_block_sys = system.systemStructSlice(
                    sys,
                    refined.blockCellOffset(refined_block),
                    refined.blockCellTotal(refined_block),
                );

                // Origin of the refined block in supercell space
                const refined_origin: [N]usize = scaled(refined_block_bounds.origin, self.mesh.tile_width);

                // Iterate supercell space
                var indices = IndexSpace.fromBox(refined_block_bounds).cartesianIndices();

                while (indices.next()) |idx| {
                    const supercell: [N]isize = toSigned(add(refined_origin, idx));

                    inline for (comptime system.systemFieldNames(@TypeOf(sys))) |name| {
                        cell_space.setValue(
                            supercell,
                            @field(base_sys, name),
                            refined_cell_space.restrict(
                                idx,
                                @field(refined_block_sys, name),
                            ),
                        );
                    }
                }
            }
        }

        pub fn restrictToLevel(self: *const Self, level: usize, block_map: []const usize, sys: anytype) void {
            // A more refined level must exist in order to perform restriction
            if (self.mesh.active_levels <= level + 1) {
                return;
            }

            // Cache target and refined variables
            const target: *const Level = self.mesh.getLevel(level);
            const refined: *const Level = self.mesh.getLevel(level + 1);

            const patches = target.patches.slice();
            const blocks = target.blocks.slice();

            const refined_patches = refined.patches.slice();
            const refined_blocks = refined.blocks.slice();

            // Iterate over refined patches
            for (0..refined_patches.len) |refined_patch| {
                // Get target patch under each refined patch
                const patch: usize = target.parents.items[refined_patch];
                const patch_block_map: usize = self.mesh.levelTileSlice(level + 1, refined_patch, block_map);
                const patch_bounds: IndexBox = patches.items(.bounds)[patch];
                const patch_space: IndexSpace = IndexSpace.fromBox(patch_bounds);

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

                    const refined_block_sys = system.systemStructSlice(
                        sys,
                        refined.blockCellOffset(refined_block),
                        refined.blockCellTotal(refined_block),
                    );

                    // Iterate tiles in refined block
                    var tiles = refined_block_space.cartesianIndices();

                    while (tiles.next()) |tile| {
                        // Tile in patch space
                        const relative_tile = refined_relative_bounds.globalFromLocal(tile);
                        // Origin of tile in supercell space
                        const refined_origin: [N]usize = scaled(tile, self.mesh.tile_width);
                        // Block under refined tile
                        const block = patch_block_map[patch_space.linearFromCartesian(relative_tile)];
                        const block_bounds: IndexBox = blocks.items(.bounds)[block];
                        const block_cell_space: CellSpace = CellSpace.fromSize(block_bounds.size);
                        const block_sys = system.systemStructSlice(
                            sys,
                            target.blockCellOffset(block),
                            target.blockCellTotal(block),
                        );

                        // Bounds of block relative to patch
                        const relative_bounds: IndexBox = block_bounds.relativeTo(patch_bounds);

                        // Origin of tile in supercell space on block
                        const origin: [N]usize = scaled(relative_bounds.localFromGlobal(relative_tile), self.mesh.config.tile_width);

                        // Iterate over supercells
                        var indices = IndexSpace.fromSize(splat(self.mesh.config.tile_width)).cartesianIndices();

                        while (indices.next()) |idx| {
                            // Refined cell in supercell space
                            const refined_cell = toSigned(add(refined_origin, idx));
                            // Cell in supercell space
                            const cell = toSigned(add(origin, idx));

                            inline for (comptime system.systemFieldNames(@TypeOf(sys))) |name| {
                                block_cell_space.setValue(
                                    cell,
                                    @field(block_sys, name),
                                    refined_cell_space.restrict(
                                        refined_cell,
                                        @field(refined_block_sys, name),
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }

        pub fn prolong(self: *const Self, block_map: []const usize, sys: anytype) void {
            if (comptime !system.isSystemSlice(sys)) {
                @compileError("System must satisfy isSystemSlice trait.");
            }

            assert(block_map.len == self.mesh.tileTotal());

            self.prolongFromBase(sys);

            for (0..self.mesh.active_levels - 1) |level| {
                self.prolongFromLevel(level, block_map, sys);
            }
        }

        pub fn prolongFromBase(self: *const Self, sys: anytype) void {
            if (self.mesh.active_levels == 0) {
                return;
            }

            const cell_space: CellSpace = CellSpace.fromSize(self.mesh.base.index_size);
            const base_sys = system.systemStructSlice(sys, 0, self.mesh.cell_total);

            const target: *const Level = self.mesh.getLevel(0);
            const blocks = target.blocks.slice();

            for (0..blocks.len) |b| {
                const block_bounds: IndexBox = blocks.items(.bounds)[b];
                const block_cell_space: CellSpace = CellSpace.fromSize(block_bounds.size);

                const block_sys = system.systemStructSlice(
                    sys,
                    target.blockCellOffset(b),
                    target.blockCellTotal(b),
                );

                const origin: [N]usize = scaled(block_bounds.origin, self.mesh.tile_width);

                var indices = IndexSpace.fromSize(scaled(block_bounds.size, self.mesh.tile_width)).cartesianIndices();

                while (indices.next()) |idx| {
                    const cell = cell_space.offsetFromOrigin(origin, toSigned(idx));

                    inline for (comptime system.systemFieldNames(@TypeOf(sys))) |name| {
                        block_cell_space.setValue(
                            toSigned(idx),
                            @field(block_sys, name),
                            cell_space.prolong(
                                cell,
                                @field(base_sys, name),
                            ),
                        );
                    }
                }
            }
        }

        pub fn prolongFromLevel(self: *const Self, level: usize, block_map: []const usize, sys: anytype) void {
            // If a more refined level doesn't exist, then this function is done.
            if (self.mesh.active_levels <= level + 1) {
                return;
            }
            // Cache various variables
            const coarse: *const Level = self.mesh.getLevel(level);
            const target: *const Level = self.mesh.getLevel(level + 1);

            const coarse_patches = coarse.patches.slice();
            const coarse_blocks = coarse.blocks.slice();

            const patches = target.patches.slice();
            const blocks = target.blocks.slice();

            // Iterate patches on refined level
            for (0..patches.len) |patch| {
                // Cache underlying patch data
                const coarse_patch: usize = coarse.parents.items[patch];
                const coarse_patch_block_map = self.mesh.levelTileSlice(level, coarse_patch, block_map);
                const coarse_patch_bounds: IndexBox = coarse_patches.items(.bounds)[coarse_patch];
                const coarse_patch_space: IndexSpace = coarse_patch_bounds.space();

                const offset: usize = patches.items(.block_offset)[patch];
                const total: usize = patches.items(.block_total)[patch];

                // Iterate blocks on refined patch
                for (offset..(offset + total)) |block| {
                    const block_sys = system.systemStructSlice(
                        sys,
                        target.blockCellOffset(block),
                        target.blockCellTotal(block),
                    );
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
                        const coarse_block_sys = system.systemStructSlice(
                            sys,
                            coarse.blockCellOffset(coarse_block),
                            coarse.blockCellTotal(coarse_block),
                        );

                        // Get coarse origin (in subcell space)
                        const coarse_origin = scaled(coarse_relative_block_bounds.globalFromLocal(tile), 2 * mesh.config.tile_width);

                        // Get origin (in subcell space)
                        const origin = scaled(tile, 2 * mesh.config.tile_width);

                        // Iterate subcell space
                        var indices = IndexSpace.fromSize(splat(2 * mesh.config.tile_width)).cartesianIndices();

                        while (indices.next()) |idx| {
                            // Get coarse subcell
                            const coarse_cell = toSigned(add(coarse_origin, idx));
                            // Get target subcell
                            const cell = toSigned(add(origin, idx));

                            // Prolong
                            inline for (comptime system.systemFieldNames(@TypeOf(sys))) |name| {
                                cell_space.setValue(
                                    cell,
                                    @field(block_sys, name),
                                    coarse_cell_space.prolong(
                                        coarse_cell,
                                        @field(coarse_block_sys, name),
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }

        // *************************
        // Smoothing ***************
        // *************************

        /// Runs one iteration of Jacobi's method on the full mesh. Assuming all boundaries have been filled at least partially.
        pub fn smooth(
            self: *const Self,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            input: system.SystemSliceConst(@TypeOf(oper).System),
            rhs: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            self.smoothBase(oper, result, input, rhs, context);

            for (0..self.mesh.active_levels) |level| {
                self.smoothLevel(level, oper, result, input, rhs, context);
            }
        }

        /// Runs one iteration of Jacobi's method on the base level, assuming all boundaries have been filled at least partially.
        pub fn smoothBase(
            self: *const Self,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            input: system.SystemSliceConst(@TypeOf(oper).System),
            rhs: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            if (!(operator.isMeshOperator(N, O)(@TypeOf(oper)))) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            const base_offset = 0;
            const base_total = self.meshbase.cell_total;

            const base_result = system.systemStructSlice(result, base_offset, base_total);
            const base_input = system.systemStructSlice(input, base_offset, base_total);
            const base_rhs = system.systemStructSlice(rhs, base_offset, base_total);
            const base_context = system.systemStructSlice(context, base_offset, base_total);

            const stencil_space: StencilSpace = self.mesh.baseStencilSpace();

            var cells = stencil_space.cellSpace().cells();

            while (cells.next()) |cell| {
                const engine = EngineType(N, O, @TypeOf(oper)){
                    .inner = .{
                        .space = stencil_space,
                        .cell = cell,
                    },
                    .context = base_context,
                    .operated = base_input,
                };

                const app: system.SystemValue(@TypeOf(oper).System) = oper.apply(engine);
                const diag: system.SystemValue(@TypeOf(oper).System) = oper.applyDiagonal(engine);

                inline for (system.systemFieldNames(@TypeOf(oper).System)) |name| {
                    const f: f64 = stencil_space.value(cell, @field(base_input, name));
                    const r: f64 = stencil_space.value(cell, @field(base_rhs, name));
                    const a: f64 = @field(app, name);
                    const d: f64 = @field(diag, name);

                    stencil_space.cellSpace().setValue(cell, @field(base_result, name), f + (r - a) / d);
                }
            }
        }

        /// Runs one iteration of Jacobi's method on the given level, assuming all boundaries have been filled at least partially.
        pub fn smoothLevel(
            self: *const Self,
            level: usize,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            input: system.SystemSliceConst(@TypeOf(oper).System),
            rhs: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            if (!(operator.isMeshOperator(N, O)(@TypeOf(oper)))) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            const target: *const Level = &self.levels[level];

            const level_offset = target.cell_offset;
            const level_total = target.cell_total;

            const level_result = system.systemStructSlice(result, level_offset, level_total);
            const level_input = system.systemStructSlice(input, level_offset, level_total);
            const level_rhs = system.systemStructSlice(rhs, level_offset, level_total);
            const level_context = system.systemStructSlice(context, level_offset, level_total);

            for (target.blocks.items(.bounds), target.blocks.items(.cell_offset), target.blocks.items(.cell_total)) |block, offset, total| {
                const block_result = system.systemStructSlice(level_result, offset, total);
                const block_input = system.systemStructSlice(level_input, offset, total);
                const block_rhs = system.systemStructSlice(level_rhs, offset, total);
                const block_context = system.systemStructSlice(level_context, offset, total);

                const stencil_space: StencilSpace = self.levelStencilSpace(level, block);

                var cells = stencil_space.cellSpace().cells();

                while (cells.next()) |cell| {
                    const engine = EngineType(N, O, @TypeOf(oper)){
                        .inner = .{
                            .space = stencil_space,
                            .cell = cell,
                        },
                        .context = block_context,
                        .operated = block_input,
                    };

                    const app: system.SystemValue(@TypeOf(oper).System) = oper.apply(engine);
                    const diag: system.SystemValue(@TypeOf(oper).System) = oper.applyDiagonal(engine);

                    inline for (system.systemFieldNames(@TypeOf(oper).System)) |name| {
                        const f: f64 = stencil_space.value(cell, @field(block_input, name));
                        const r: f64 = stencil_space.value(cell, @field(block_rhs, name));
                        const a: f64 = @field(app, name);
                        const d: f64 = @field(diag, name);

                        stencil_space.setValue(cell, @field(block_result, name), f + (r - a) / d);
                    }
                }
            }
        }

        // *************************
        // Application *************
        // *************************

        pub fn apply(
            self: *const Self,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            sys: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            self.applyBase(oper, result, sys, context);

            for (0..self.mesh.active_levels) |level| {
                self.applyLevel(level, oper, result, sys, context);
            }
        }

        pub fn applyBase(
            self: *const Self,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            sys: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            if (comptime !(operator.isMeshOperator(N, O)(@TypeOf(oper)))) {
                @compileError("Func must satisfy isMeshOperator trait.");
            }

            const base_offset = 0;
            const base_total = self.mesh.base.cell_total;

            const base_result = system.systemStructSlice(result, base_offset, base_total);
            const base_sys = system.systemStructSlice(sys, base_offset, base_total);
            const base_context = system.systemStructSlice(context, base_offset, base_total);

            const stencil_space: StencilSpace = self.mesh.baseStencilSpace();

            var cell_indices = stencil_space.cellSpace().cells();

            while (cell_indices.next()) |cell| {
                const engine = EngineType(N, O, @TypeOf(oper)){
                    .inner = .{
                        .space = stencil_space,
                        .cell = cell,
                    },
                    .sys = base_sys,
                    .context = base_context,
                };

                const val: system.SystemValue(@TypeOf(oper).System) = oper.apply(engine);

                inline for (comptime system.systemFieldNames(@TypeOf(oper).System)) |name| {
                    stencil_space.cellSpace().setValue(
                        cell,
                        @field(base_result, name),
                        @field(val, name),
                    );
                }
            }
        }

        pub fn applyLevel(
            self: *const Self,
            level: usize,
            oper: anytype,
            result: system.SystemSlice(@TypeOf(oper).System),
            sys: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            if (comptime !(operator.isMeshOperator(N, O)(@TypeOf(oper)))) {
                @compileError("Func must satisfy isMeshOperator trait.");
            }

            const target: *const Level = self.mesh.getLevel(level);

            const level_offset = target.cell_offset;
            const level_total = target.cell_total;

            const level_result = system.systemStructSlice(result, level_offset, level_total);
            const level_sys = system.systemStructSlice(sys, level_offset, level_total);
            const level_context = system.systemStructSlice(context, level_offset, level_total);

            for (target.blocks.items(.cell_offset), target.blocks.items(.cell_total), 0..) |offset, total, id| {
                const block_result = system.systemStructSlice(level_result, offset, total);
                const block_sys = system.systemStructSlice(level_sys, offset, total);
                const block_context = system.systemStructSlice(level_context, offset, total);

                const stencil_space: StencilSpace = self.mesh.levelStencilSpace(level, id);

                var cell_indices = stencil_space.cellSpace().cells();

                while (cell_indices.next()) |cell| {
                    const engine = EngineType(N, O, @TypeOf(oper)){
                        .inner = .{
                            .space = stencil_space,
                            .cell = cell,
                        },
                        .context = block_context,
                        .sys = block_sys,
                    };

                    const val: system.SystemValue(@TypeOf(oper).System) = oper.apply(engine);

                    inline for (comptime system.systemFieldNames(@TypeOf(oper).System)) |name| {
                        stencil_space.cellSpace().setValue(
                            cell,
                            @field(block_result, name),
                            @field(val, name),
                        );
                    }
                }
            }
        }

        // *************************
        // Projection **************
        // *************************

        pub fn project(
            self: *const Self,
            func: anytype,
            result: system.SystemSlice(@TypeOf(func).Output),
            input: system.SystemSliceConst(@TypeOf(func).Input),
        ) void {
            self.projectBase(func, result, input);

            for (0..self.mesh.active_levels) |level| {
                self.projectLevel(level, func, result, input);
            }
        }

        pub fn projectBase(
            self: *const Self,
            func: anytype,
            result: system.SystemSlice(@TypeOf(func).Output),
            input: system.SystemSliceConst(@TypeOf(func).Input),
        ) void {
            if (comptime !(operator.isMeshFunction(N, O)(@TypeOf(func)))) {
                @compileError("Func must satisfy isMeshFunction trait.");
            }

            const base_offset = 0;
            const base_total = self.mesh.base.cell_total;

            const base_result = system.systemStructSlice(result, base_offset, base_total);
            const base_context = system.systemStructSlice(input, base_offset, base_total);

            const stencil_space: StencilSpace = self.mesh.baseStencilSpace();

            var cell_indices = stencil_space.cellSpace().cells();

            while (cell_indices.next()) |cell| {
                const engine = EngineType(N, O, @TypeOf(func)){
                    .inner = .{
                        .space = stencil_space,
                        .cell = cell,
                    },
                    .input = base_context,
                };

                const val: system.SystemValue(@TypeOf(func).Output) = func.value(engine);

                inline for (comptime system.systemFieldNames(@TypeOf(func).Output)) |name| {
                    stencil_space.cellSpace().setValue(
                        cell,
                        @field(base_result, name),
                        @field(val, name),
                    );
                }
            }
        }

        pub fn projectLevel(
            self: *const Self,
            level: usize,
            func: anytype,
            result: system.SystemSlice(@TypeOf(func).Output),
            input: system.SystemSliceConst(@TypeOf(func).Input),
        ) void {
            if (comptime !(operator.isMeshFunction(N, O)(@TypeOf(func)))) {
                @compileError("Func must satisfy isMeshFunction trait.");
            }

            const target: *const Level = self.mesh.getLevel(level);

            const level_offset = target.cell_offset;
            const level_total = target.cell_total;

            const level_result = system.systemStructSlice(result, level_offset, level_total);
            const level_context = system.systemStructSlice(input, level_offset, level_total);

            for (target.blocks.items(.cell_offset), target.blocks.items(.cell_total), 0..) |offset, total, id| {
                const block_result = system.systemStructSlice(level_result, offset, total);
                const block_context = system.systemStructSlice(level_context, offset, total);

                const stencil_space: StencilSpace = self.mesh.levelStencilSpace(level, id);

                var cell_indices = stencil_space.cellSpace().cells();

                while (cell_indices.next()) |cell| {
                    const engine = EngineType(N, O, @TypeOf(func)){
                        .inner = .{
                            .space = stencil_space,
                            .cell = cell,
                        },
                        .input = block_context,
                    };

                    const val: system.SystemValue(@TypeOf(func).Output) = func.value(engine);

                    inline for (comptime system.systemFieldNames(@TypeOf(func).Output)) |name| {
                        stencil_space.cellSpace().setValue(
                            cell,
                            @field(block_result, name),
                            @field(val, name),
                        );
                    }
                }
            }
        }

        // *************************
        // Output ******************
        // *************************

        fn GridData(comptime System: type, comptime NVertices: usize, comptime full: bool) type {
            const field_count = system.systemFieldCount(System);
            return struct {
                positions: ArrayListUnmanaged(f64) = .{},
                vertices: ArrayListUnmanaged(usize) = .{},
                fields: [field_count]ArrayListUnmanaged(f64) = [1]ArrayListUnmanaged(f64){.{}} ** field_count,
                input: System,

                fn deinit(data: *@This(), allocator: Allocator) void {
                    data.positions.deinit(allocator);
                    data.vertices.deinit(allocator);

                    for (0..data.fields.len) |i| {
                        data.fields[i].deinit(allocator);
                    }
                }

                fn build(
                    data: *@This(),
                    allocator: Allocator,
                    stencil: StencilSpace,
                    offset: usize,
                    total: usize,
                ) !void {
                    const cell_size = if (full) add(stencil.size, splat(2 * O)) else stencil.size;
                    const point_size = add(cell_size, splat(1));

                    const cell_space: IndexSpace = IndexSpace.fromSize(cell_size);
                    const point_space: IndexSpace = IndexSpace.fromSize(point_size);

                    const point_offset: usize = data.positions.items.len;

                    try data.positions.ensureUnusedCapacity(allocator, N * point_space.total());
                    try data.vertices.ensureUnusedCapacity(allocator, NVertices * cell_space.total());

                    // Fill positions and vertices
                    var points = point_space.cartesianIndices();

                    while (points.next()) |point| {
                        var vertex: [N]isize = undefined;

                        for (0..N) |i| {
                            vertex[i] = @as(isize, @intCast(point[i])) - @as(isize, @intCast(2 * O));
                        }

                        const pos = stencil.vertexPosition(vertex);
                        for (0..N) |i| {
                            data.positions.appendAssumeCapacity(pos[i]);
                        }
                    }

                    if (N == 1) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(add(cell, splat(1)));

                            data.vertices.appendAssumeCapacity(point_offset + v1);
                            data.vertices.appendAssumeCapacity(point_offset + v2);
                        }
                    } else if (N == 2) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(add(cell, [2]usize{ 0, 1 }));
                            const v3: usize = point_space.linearFromCartesian(add(cell, [2]usize{ 1, 1 }));
                            const v4: usize = point_space.linearFromCartesian(add(cell, [2]usize{ 1, 0 }));

                            data.vertices.appendAssumeCapacity(point_offset + v1);
                            data.vertices.appendAssumeCapacity(point_offset + v2);
                            data.vertices.appendAssumeCapacity(point_offset + v3);
                            data.vertices.appendAssumeCapacity(point_offset + v4);
                        }
                    } else if (N == 3) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 0, 1, 0 }));
                            const v3: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 1, 1, 0 }));
                            const v4: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 1, 0, 0 }));
                            const v5: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 0, 0, 1 }));
                            const v6: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 0, 1, 3 }));
                            const v7: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 1, 1, 3 }));
                            const v8: usize = point_space.linearFromCartesian(add(cell, [3]usize{ 1, 0, 3 }));

                            data.vertices.appendAssumeCapacity(point_offset + v1);
                            data.vertices.appendAssumeCapacity(point_offset + v2);
                            data.vertices.appendAssumeCapacity(point_offset + v3);
                            data.vertices.appendAssumeCapacity(point_offset + v4);
                            data.vertices.appendAssumeCapacity(point_offset + v5);
                            data.vertices.appendAssumeCapacity(point_offset + v6);
                            data.vertices.appendAssumeCapacity(point_offset + v7);
                            data.vertices.appendAssumeCapacity(point_offset + v8);
                        }
                    }

                    for (0..field_count) |id| {
                        try data.fields[id].ensureUnusedCapacity(allocator, cell_space.total());
                    }

                    const block_field = system.systemStructSlice(data.input, offset, total);

                    var cells = if (full) stencil.cellSpace().fullCells() else stencil.cellSpace().cells();

                    while (cells.next()) |cell| {
                        inline for (comptime system.systemFieldNames(System), 0..) |name, id| {
                            data.fields[id].appendAssumeCapacity(
                                stencil.value(
                                    cell,
                                    @field(block_field, name),
                                ),
                            );
                        }
                    }
                }
            };
        }

        pub fn writeVtk(self: *const Self, comptime full: bool, sys: anytype, out_stream: anytype) !void {
            if (comptime !(system.isSystemSliceConst(@TypeOf(sys)) or system.isSystemSlice(@TypeOf(sys)))) {
                @compileError("Sys must satisfy isSystemSliceConst trait.");
            }

            const vtkio = @import("../vtkio.zig");
            const VtuMeshOutput = vtkio.VtuMeshOutput;
            const VtkCellType = vtkio.VtkCellType;

            // Global Constants
            const cell_type: VtkCellType = switch (N) {
                1 => .line,
                2 => .quad,
                3 => .hexa,
                else => @compileError("Vtk Output not supported for N > 3"),
            };

            // Build data
            var data = GridData(@TypeOf(sys), cell_type.nvertices(), full){ .input = sys };
            defer data.deinit(self.gpa);

            // Build base
            try data.build(
                self.gpa,
                self.mesh.baseStencilSpace(),
                0,
                self.mesh.base.cell_total,
            );

            for (0..self.mesh.active_levels) |level| {
                const target: *const Level = self.mesh.getLevel(level);
                const level_offset = target.cell_offset;
                const block_offsets = target.blocks.items(.cell_offset);
                const block_totals = target.blocks.items(.cell_total);

                for (block_offsets, block_totals, 0..) |offset, total, id| {
                    const stencil: StencilSpace = self.mesh.levelStencilSpace(level, id);

                    try data.build(
                        self.gpa,
                        stencil,
                        level_offset + offset,
                        total,
                    );
                }
            }

            var grid: VtuMeshOutput = try VtuMeshOutput.init(self.gpa, .{
                .points = data.positions.items,
                .vertices = data.vertices.items,
                .cell_type = cell_type,
            });
            defer grid.deinit();

            for (system.systemFieldNames(@TypeOf(sys)), 0..) |name, id| {
                try grid.addCellField(name, data.fields[id].items, 1);
            }

            try grid.write(out_stream);
        }

        // **********************
        // Helpers **************
        // **********************

        pub fn ndofs(self: *const Self) usize {
            return self.mesh.cell_total;
        }
    };
}
