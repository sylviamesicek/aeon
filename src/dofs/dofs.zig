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
const meshes = @import("../mesh/mesh.zig");
const system = @import("../system.zig");

// submodules

const boundaries = @import("boundary.zig");
const multigrid = @import("multigrid.zig");
const operator = @import("operator.zig");

// Public Exports

pub const BoundaryCondition = boundaries.BoundaryCondition;
pub const SystemBoundaryCondition = boundaries.SystemBoundaryCondition;
pub const isSystemBoundary = boundaries.isSystemBoundary;

pub const Engine = operator.Engine;
pub const OperatorEngine = operator.OperatorEngine;
pub const FunctionEngine = operator.FunctionEngine;
pub const EngineType = operator.EngineType;
pub const isMeshFunction = operator.isMeshFunction;
pub const isMeshOperator = operator.isMeshOperator;
pub const isMeshProjection = operator.isMeshProjection;

pub const MultigridSolver = multigrid.MultigridSolver;

const SystemSlice = system.SystemSlice;
const SystemSliceConst = system.SystemSliceConst;

/// A map from a level and block to a the offset and total dofs.
pub fn DofMap(comptime N: usize, comptime O: usize) type {
    return struct {
        offsets: []const usize,
        levels: []const usize,

        const Self = @This();
        const Mesh = meshes.Mesh(N);
        const Index = index.Index(N);
        const CellSpace = basis.CellSpace(N, O);

        pub fn init(allocator: Allocator, mesh: *const Mesh) Self {
            const levels = try allocator.alloc(usize, mesh.active_levels + 1);
            errdefer allocator.free(levels);

            levels[0] = 0;

            var level_ptr: usize = 0;

            for (0..mesh.active_levels) |level| {
                const current = levels[level_ptr];
                level_ptr += 1;
                levels[level_ptr] = current + mesh.getLevel(level).blockTotal();
            }

            const offsets = try allocator.alloc(usize, levels[mesh.active_levels]);
            errdefer allocator.free(offsets);

            offsets[0] = 0;

            var offset_ptr: usize = 0;

            for (0..mesh.active_levels) |level| {
                for (mesh.getLevel(level).blocks.items(.bounds)) |bounds| {
                    const current = offsets[offset_ptr];
                    offset_ptr += 1;
                    offsets[offset_ptr] = current + CellSpace.fromSize(Index.scaled(bounds.size, mesh.tile_width)).total();
                }
            }

            return .{
                .levels = levels,
                .offsets = offsets,
            };
        }

        pub fn deinit(self: Self, allocator: Allocator) void {
            allocator.free(self.levels);
            allocator.free(self.offsets);
        }

        pub fn blockOffset(self: Self, level: usize, block: usize) usize {
            const idx = self.levels[level] + block;
            return self.offsets[idx];
        }

        pub fn blockTotal(self: Self, level: usize, block: usize) usize {
            const idx = self.levels[level] + block;
            return self.offsets[idx + 1] - self.offsets[idx];
        }

        pub fn total(self: Self) usize {
            return self.offsets[self.offsets.len - 1];
        }
    };
}

/// A namespace with many utils for managing DoFs, including filling boundaries,
/// writing output data, projecting functions, etc. All functions which take in
/// the mesh as an argument transfer/manipulate data between global vectors and
/// local windows. All functions which take in stencil spaces act on windows only.
pub fn DofUtils(comptime N: usize, comptime O: usize) type {
    return struct {
        const CellSpace = basis.CellSpace(N, O);
        const StencilSpace = basis.StencilSpace(N, O);
        const Index = index.Index(N);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexBox = geometry.Box(N, usize);
        const Mesh = meshes.Mesh(N);
        const Level = Mesh.Level;
        const BoundaryUtils = boundaries.BoundaryUtils(N, O);
        const Face = geometry.Face(N);
        const Region = geometry.Region(N);
        const Map = DofMap(N, O);

        /// Computes the number of cells along each axis for a given block.
        pub fn blockCellSize(mesh: *const Mesh, level: usize, block: usize) [N]usize {
            return Index.scaled(mesh.getLevel(level).blocks.items(.bounds)[block].size, mesh.tile_width);
        }

        /// Builds a stencil space for the given block.
        pub fn blockStencilSpace(mesh: *const Mesh, level: usize, block: usize) StencilSpace {
            return .{
                .physical_bounds = mesh.blockPhysicalBounds(level, block),
                .size = blockCellSize(mesh, level, block),
            };
        }

        /// Extracts the system boundary conditions from an operator.
        pub fn OperSystemBoundary(comptime T: type) type {
            return struct {
                oper: T,

                pub const System = T.System;

                pub fn boundary(self: @This(), pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                    return self.oper.boundarySys(pos, face);
                }
            };
        }

        /// Extracts the context boundary conditions from an operator.
        pub fn OperContextBoundary(comptime T: type) type {
            if (comptime !(operator.isMeshOperator(N, O)(T))) {
                @compileError("Oper must satisfy isMeshOperator traits.");
            }

            return struct {
                oper: T,

                pub const System = T.Context;

                pub fn boundary(self: @This(), pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                    return self.oper.boundaryCtx(pos, face);
                }
            };
        }

        /// Extracts the boundary conditions from a mesh function.
        pub fn FuncBoundary(comptime T: type) type {
            if (comptime !(operator.isMeshFunction(N, O)(T))) {
                @compileError("Oper must satisfy isMeshFunction traits.");
            }

            return struct {
                func: T,

                pub const System = T.Input;

                pub fn boundary(self: @This(), pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                    return self.oper.boundary(pos, face);
                }
            };
        }

        pub fn operSystemBoundary(oper: anytype) OperSystemBoundary(@TypeOf(oper)) {
            return .{
                .oper = oper,
            };
        }

        pub fn operContextBoundary(oper: anytype) OperContextBoundary(@TypeOf(oper)) {
            return .{
                .oper = oper,
            };
        }

        pub fn funcBoundary(func: anytype) FuncBoundary(@TypeOf(func)) {
            return .{
                .func = func,
            };
        }

        // ************************
        // Copy *******************
        // ************************

        /// Performs `copyDofsFromCells` for all levels and blocks.
        pub fn copyDofsFromCellsAll(
            comptime System: type,
            mesh: *const Mesh,
            dof_map: Map,
            dest: SystemSlice(System),
            src: SystemSliceConst(System),
        ) void {
            for (0..mesh.active_levels) |level| {
                for (0..mesh.getLevel(level).blockTotal()) |block| {
                    copyDofsFromCells(
                        System,
                        mesh,
                        dof_map,
                        level,
                        block,
                        dest,
                        src,
                    );
                }
            }
        }

        /// Copies the data from a vector of cells to a vector of dofs.
        pub fn copyDofsFromCells(
            comptime System: type,
            mesh: *const Mesh,
            dof_map: Map,
            level: usize,
            block: usize,
            dest: SystemSlice(System),
            src: SystemSliceConst(System),
        ) void {
            assert(dest.len == dof_map.total());
            assert(src == mesh.cell_total);

            const block_dest = dest.slice(dof_map.blockOffset(level, block), dof_map.blockTotal(level, block));
            const block_src = src.slice(mesh.blockCellOffset(level, block), mesh.blockCellTotal(level, block));

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, level, block));

            var cells = cell_space.cells();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                inline for (comptime std.enums.values(System)) |field| {
                    cell_space.setValue(cell, block_dest.field(field), block_src.field(field)[linear]);
                }
            }
        }

        /// Performs `copyCellsFromDofs` for all levels and blocks.
        pub fn copyCellsFromDofsAll(
            comptime System: type,
            mesh: *const Mesh,
            dof_map: Map,
            dest: SystemSlice(System),
            src: SystemSliceConst(System),
        ) void {
            for (0..mesh.active_levels) |level| {
                for (0..mesh.getLevel(level).blockTotal()) |block| {
                    copyCellsFromDofs(
                        System,
                        mesh,
                        dof_map,
                        level,
                        block,
                        dest,
                        src,
                    );
                }
            }
        }

        /// Copies the data from a vector of dofs to a vector of cells.
        pub fn copyCellsFromDofs(
            comptime System: type,
            mesh: *const Mesh,
            dof_map: Map,
            level: usize,
            block: usize,
            dest: SystemSlice(System),
            src: SystemSliceConst(System),
        ) void {
            assert(src.len == dof_map.total());
            assert(dest == mesh.cell_total);

            const block_src = src.slice(dof_map.blockOffset(level, block), dof_map.blockTotal(level, block));
            const block_dest = dest.slice(mesh.blockCellOffset(level, block), mesh.blockCellTotal(level, block));

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, level, block));

            var cells = cell_space.cells();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                inline for (comptime std.enums.values(System)) |field| {
                    block_dest.field(field)[linear] = cell_space.value(cell, block_src.field(field));
                }
            }
        }

        /// Copies data from one dof vector to another at the given block.
        pub fn copyDofs(
            comptime System: type,
            dof_map: Map,
            level: usize,
            block: usize,
            dest: SystemSlice(System),
            src: SystemSliceConst(System),
        ) void {
            assert(dest.len == dof_map.total());
            assert(src == dof_map.total());

            const cell_offset = dof_map.blockOffset(level, block);
            const cell_total = dof_map.blockTotal(level, block);

            inline for (comptime std.enums.values(System)) |field| {
                @memcpy(dest.field(field)[cell_offset .. cell_offset + cell_total], src.field(field)[cell_offset .. cell_offset + cell_total]);
            }
        }

        /// Copies data from one cell vector to another at the given block.
        pub fn copyCells(
            comptime System: type,
            mesh: *const Mesh,
            level: usize,
            block: usize,
            dest: SystemSlice(System),
            src: SystemSliceConst(System),
        ) void {
            assert(dest.len == mesh.cell_total);
            assert(src == mesh.cell_total);

            const cell_offset = mesh.blockCellOffset(level, block);
            const cell_total = mesh.blockCellTotal(level, block);

            inline for (comptime std.enums.values(System)) |field| {
                @memcpy(dest.field(field)[cell_offset .. cell_offset + cell_total], src.field(field)[cell_offset .. cell_offset + cell_total]);
            }
        }

        // ************************
        // Fill Ops ***************
        // ************************

        /// Performs fillBoundary for all levels and blocks.
        pub fn fillBoundaryAll(
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            boundary: anytype,
            sys: system.SystemSliceConst(@TypeOf(boundary).System),
        ) void {
            for (0..mesh.active_levels) |level| {
                for (0..mesh.getLevel(level).blockTotal()) |block| {
                    fillBoundary(
                        mesh,
                        block_map,
                        dof_map,
                        level,
                        block,
                        boundary,
                        sys,
                    );
                }
            }
        }

        /// Given a dof vector, fill all boundary dofs on that vector out to an extent O using the
        /// supplied boundary conditions.
        pub fn fillBoundary(
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            level: usize,
            block: usize,
            boundary: anytype,
            sys: system.SystemSliceConst(@TypeOf(boundary).System),
        ) void {
            fillBoundaryToExtent(O, mesh, block_map, dof_map, level, block, boundary, sys);
        }

        /// Perfom `fillBoundaryFull` for all levels and blocks.
        pub fn fillBoundaryFullAll(
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            boundary: anytype,
            sys: system.SystemSliceConst(@TypeOf(boundary).System),
        ) void {
            for (0..mesh.active_levels) |level| {
                for (0..mesh.getLevel(level).blockTotal()) |block| {
                    fillBoundaryFull(
                        mesh,
                        block_map,
                        dof_map,
                        level,
                        block,
                        boundary,
                        sys,
                    );
                }
            }
        }

        /// Given a dof vector, fill all boundary dofs on that vector out to an extent 2*O using the
        /// supplied boundary conditions.
        pub fn fillBoundaryFull(
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            level: usize,
            block: usize,
            boundary: anytype,
            sys: system.SystemSlice(@TypeOf(boundary).System),
        ) void {
            fillBoundaryToExtent(2 * O, mesh, block_map, dof_map, level, block, boundary, sys);
        }

        /// Internal helper function for filling boundary dofs to some extent E.
        fn fillBoundaryToExtent(
            comptime E: usize,
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            level: usize,
            block: usize,
            boundary: anytype,
            sys: system.SystemSliceConst(@TypeOf(boundary).System),
        ) void {
            const T = @TypeOf(boundary);

            if (comptime !isSystemBoundary(N)(T)) {
                @compileError("FillBaseBoundary requires boundary satisfy isSystemBoundary trait.");
            }

            assert(sys.len == dof_map.total());

            const target = mesh.getLevel(level);
            const bounds: IndexBox = target.blocks.items(.bounds)[block];

            const regions = comptime Region.orderedRegions();

            inline for (comptime regions[1..]) |region| {
                var exterior: bool = false;

                for (0..N) |i| {
                    if (region.sides[i] == .left and bounds.origin[i] == 0) {
                        exterior = true;
                    } else if (region.sides[i] == .right and bounds.origin[i] + bounds.size[i] == target.index_size[i]) {
                        exterior = true;
                    }
                }

                if (exterior) {
                    const stencil_space = blockStencilSpace(mesh, level, block);

                    const block_sys = sys.slice(
                        dof_map.blockOffset(level, block),
                        dof_map.blockTotal(level, block),
                    );

                    BoundaryUtils.fillBoundaryRegion(E, region, stencil_space, boundary, block_sys);
                } else {
                    fillInteriorBoundary(T.System, region, E, mesh, block_map, level, block, sys);
                }
            }
        }

        /// Fills non physical boundaries (ie boundaries within the numerical domain between two blocks).
        fn fillInteriorBoundary(
            comptime System: type,
            comptime region: Region,
            comptime E: usize,
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            level: usize,
            block: usize,
            sys: system.SystemSliceConst(System),
        ) void {
            const target: *const Level = mesh.getLevel(level);

            const blocks = target.blocks.slice();
            const patches = target.patches.slice();

            const block_bounds = blocks.items(.bounds)[block];
            const block_cell_space = CellSpace.fromSize(block_bounds.size);

            const patch = blocks.items(.patch)[block];
            const patch_bounds: IndexBox = patches.items(.bounds)[patch];
            const patch_space = IndexSpace.fromBox(patch_bounds);
            const patch_block_map: []const usize = mesh.patchTileSlice(level, patch, block_map);

            const relative_bounds: IndexBox = block_bounds.relativeTo(patch_bounds);

            const block_sys = sys.slice(
                dof_map.blockOffset(level, block),
                dof_map.blockTotal(level, block),
            );

            var tiles = region.innerFaceIndices(relative_bounds.size);

            while (tiles.next()) |tile| {
                var relative_tile: [N]usize = undefined;
                var buffer_tile: [N]usize = undefined;

                inline for (0..N) |i| {
                    relative_tile[i] = @intCast(@as(isize, @intCast(relative_bounds.origin[i])) + tile[i]);
                    buffer_tile[i] = @intCast(@as(isize, @intCast(relative_bounds.origin[i])) + tile[i] + region.extentDir()[i]);
                }

                const origin: [N]usize = Index.scaled(Index.toUnsigned(tile), mesh.tile_width);
                const neighbor: usize = patch_block_map[patch_space.linearFromCartesian(buffer_tile)];

                if (neighbor == maxInt(usize)) {
                    const coarse_buffer_tile: [N]usize = Index.coarsened(buffer_tile);

                    // If so cache various coarse variables
                    const coarse: *const Level = mesh.getLevel(level - 1);

                    const coarse_patch = coarse.parents.items[patch];
                    const coarse_patch_block_map: []const usize = mesh.patchTileSlice(level - 1, coarse_patch, block_map);
                    const coarse_patch_bounds = coarse.patches.items(.bounds)[coarse_patch];
                    const coarse_patch_space = IndexSpace.fromBox(coarse_patch_bounds);

                    const coarse_block = coarse_patch_block_map[coarse_patch_space.linearFromCartesian(coarse_buffer_tile)];
                    const coarse_block_cell_space: CellSpace = CellSpace.fromSize(coarse.blocks.items(.bounds)[coarse_block].size);

                    const coarse_block_sys = sys.slice(
                        dof_map.blockOffset(level - 1, coarse_block),
                        dof_map.blockTotal(level - 1, coarse_block),
                    );

                    var coarse_relative_block_bounds = coarse.blocks.items(.bounds)[coarse_block].relativeTo(coarse_patch_bounds);
                    coarse_relative_block_bounds.refine();

                    // Neighbor origin in subcell space
                    const coarse_neighbor_origin: [N]usize = Index.scaled(coarse_relative_block_bounds.localFromGlobal(relative_tile), mesh.tile_width);

                    var indices = region.cartesianIndices(E, Index.splat(mesh.tile_width));

                    while (indices.next()) |ind| {
                        // Cell in subcell space
                        const block_cell: [N]isize = CellSpace.offsetFromOrigin(origin, ind);
                        // Cell in neighbor in subcell space
                        const neighbor_cell: [N]isize = CellSpace.offsetFromOrigin(coarse_neighbor_origin, ind);

                        inline for (comptime std.enums.values(System)) |field| {
                            block_cell_space.setValue(
                                block_cell,
                                block_sys.field(field),
                                coarse_block_cell_space.prolong(
                                    neighbor_cell,
                                    coarse_block_sys.field(field),
                                ),
                            );
                        }
                    }
                } else {
                    // Copy from neighbor on same level
                    const neighbor_sys = sys.slice(
                        dof_map.blockOffset(level, neighbor),
                        dof_map.blockTotal(level, neighbor),
                    );

                    const neighbor_bounds: IndexBox = blocks.items(.bounds)[neighbor].relativeTo(patch_bounds);
                    const neighbor_cell_space: CellSpace = CellSpace.fromSize(neighbor_bounds.size);

                    const neighbor_origin: [N]usize = Index.scaled(neighbor_bounds.localFromGlobal(relative_tile), mesh.tile_width);

                    var indices = region.cartesianIndices(E, Index.splat(mesh.tile_width));

                    while (indices.next()) |idx| {
                        const block_cell: [N]isize = CellSpace.offsetFromOrigin(origin, idx);
                        const neighbor_cell: [N]isize = CellSpace.offsetFromOrigin(neighbor_origin, idx);

                        inline for (comptime std.enums.values(System)) |field| {
                            block_cell_space.setValue(
                                block_cell,
                                block_sys.field(field),
                                neighbor_cell_space.value(
                                    neighbor_cell,
                                    neighbor_sys.field(field),
                                ),
                            );
                        }
                    }
                }
            }
        }

        // *************************
        // Restrict / Prolong ******
        // *************************

        /// Given a global dof vector with correct boundary dofs at the given block, restrict the data to all underlying dofs.
        pub fn restrict(
            comptime System: type,
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            level: usize,
            block: usize,
            sys: SystemSlice(System),
        ) void {
            if (comptime !system.isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            assert(sys.len == dof_map.total());

            if (level == 0) {
                return;
            }

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, level, block));

            const target = mesh.getLevel(level);
            const coarse = mesh.getLevel(level - 1);

            const patch = coarse.parents.items[target.blocks.items(.patch)[block]];
            var patch_bounds = coarse.patches.items(.bounds)[patch];
            try patch_bounds.coarsen();

            var bounds = target.blocks.items(.bounds)[block];
            try bounds.coarsen();

            const patch_space = IndexSpace.fromBox(patch_bounds);
            const block_space = IndexSpace.fromBox(bounds);

            const block_sys = sys.slice(
                dof_map.blockOffset(level, block),
                dof_map.blockTotal(level, block),
            );

            const tile_offset = mesh.patchTileOffset(level - 1, patch);

            var tiles = block_space.cartesianIndices();

            while (tiles.next()) |tile| {
                const relative_tile = patch_bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const linear = patch_space.linearFromCartesian(relative_tile);
                const coarse_block = block_map[tile_offset + linear];

                const coarse_bounds = coarse.blocks.items(.bounds)[coarse_block];
                const coarse_cell_space = CellSpace.fromSize(blockCellSize(mesh, level - 1, coarse_block));

                const coarse_cell_offset = dof_map.blockOffset(level - 1, coarse_block);
                const coarse_cell_total = dof_map.blockTotal(level - 1, coarse_block);

                const coarse_sys = sys.slice(coarse_cell_offset, coarse_cell_total);

                const coarse_tile = coarse_bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const coarse_origin = Index.scaled(coarse_tile, mesh.tile_width);

                const origin = Index.scaled(tile, mesh.tile_width);

                var cells = IndexSpace.fromSize(Index.splat(mesh.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const supercell = Index.toSigned(Index.add(origin, cell));
                    const coarsecell = Index.toSigned(Index.add(coarse_origin, cell));

                    inline for (comptime std.enums.values(System)) |field| {
                        coarse_cell_space.setValue(coarsecell, coarse_sys.field(field), cell_space.restrict(supercell, block_sys.field(field)));
                    }
                }
            }
        }

        /// Given a set of dof vectors src, ctx, and b, where src and ctx are filled at all boundaries on level - 1 and b is filled at the
        /// given block. For each underlying dof set the value of the cell vector at this dof to be the restricted value of b minus the
        /// application of the operator at the underlying dof.
        pub fn restrictResidual(
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            level: usize,
            block: usize,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
            b: SystemSliceConst(@TypeOf(oper).System),
        ) void {
            const T = @TypeOf(oper);

            if (comptime !operator.isMeshOperator(N, O)(T)) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            assert(dest.len == mesh.cell_total);
            assert(src.len == dof_map.total());
            assert(b.len == dof_map.total());

            if (level == 0) {
                return;
            }

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, level, block));

            const target = mesh.getLevel(level);
            const coarse = mesh.getLevel(level - 1);

            const patch = coarse.parents.items[target.blocks.items(.patch)[block]];
            var patch_bounds = coarse.patches.items(.bounds)[patch];
            try patch_bounds.coarsen();

            var bounds = target.blocks.items(.bounds)[block];
            try bounds.coarsen();

            const patch_space = IndexSpace.fromBox(patch_bounds);
            const block_space = IndexSpace.fromBox(bounds);

            const block_b = b.slice(
                dof_map.blockOffset(level, block),
                dof_map.blockTotal(level, block),
            );

            const tile_offset = mesh.patchTileOffset(level - 1, patch);

            var tiles = block_space.cartesianIndices();

            while (tiles.next()) |tile| {
                const relative_tile = patch_bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const linear = patch_space.linearFromCartesian(relative_tile);
                const coarse_block = block_map[tile_offset + linear];

                const coarse_bounds = coarse.blocks.items(.bounds)[coarse_block];
                const coarse_stencil_space = blockStencilSpace(mesh, level - 1, coarse_block);
                const coarse_index_space = IndexSpace.fromSize(coarse_bounds.size);

                const coarse_cell_offset = dof_map.blockOffset(level - 1, coarse_block);
                const coarse_cell_total = dof_map.blockTotal(level - 1, coarse_block);

                const coarse_src = src.slice(coarse_cell_offset, coarse_cell_total);
                const coarse_ctx = ctx.slice(coarse_cell_offset, coarse_cell_total);

                const coarse_dest = dest.slice(
                    mesh.blockCellOffset(level - 1, coarse_block),
                    mesh.blockCellTotal(level - 1, coarse_block),
                );

                const coarse_tile = coarse_bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const coarse_origin = Index.scaled(coarse_tile, mesh.tile_width);

                const origin = Index.scaled(tile, mesh.tile_width);

                var cells = IndexSpace.fromSize(Index.splat(mesh.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const super_cell = Index.toSigned(Index.add(origin, cell));
                    const coarse_cell = Index.toSigned(Index.add(coarse_origin, cell));

                    const lin = coarse_index_space.linearFromCartesian(Index.toUnsigned(coarse_cell));

                    const engine = operator.EngineType(N, O, T){
                        .inner = .{
                            .space = coarse_stencil_space,
                            .cell = coarse_cell,
                        },
                        .ctx = coarse_ctx,
                        .sys = coarse_src,
                    };

                    const app = oper.apply(engine);

                    inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                        const b_val = cell_space.restrict(super_cell, block_b.field(field));
                        const a_val = @field(app, @tagName(field));
                        coarse_dest.field(field)[lin] = b_val - a_val;
                    }
                }
            }
        }

        /// Given a global dof vector with correct boundary dofs on the lower level, prolong the data to this block.
        pub fn prolong(
            comptime System: type,
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            level: usize,
            block: usize,
            sys: SystemSlice(System),
        ) void {
            if (comptime !system.isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            assert(sys.len == dof_map.total());

            if (level == 0) {
                return;
            }

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, level, block));

            const target = mesh.getLevel(level);
            const coarse = mesh.getLevel(level - 1);

            const patch = coarse.parents.items[target.blocks.items(.patch)[block]];
            const patch_bounds = coarse.patches.items(.bounds)[patch];
            const bounds = target.blocks.items(.bounds)[block];

            const patch_space = IndexSpace.fromBox(patch_bounds);
            const block_space = IndexSpace.fromBox(bounds);

            const block_sys = sys.slice(
                dof_map.blockOffset(level, block),
                dof_map.blockTotal(level, block),
            );

            const tile_offset = mesh.patchTileOffset(level - 1, patch);

            var tiles = block_space.cartesianIndices();

            while (tiles.next()) |tile| {
                const relative_tile = patch_bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const linear = patch_space.linearFromCartesian(relative_tile);
                const coarse_block = block_map[tile_offset + linear];

                var coarse_bounds = coarse.blocks.items(.bounds)[coarse_block];
                coarse_bounds.refine();

                const coarse_cell_space = CellSpace.fromSize(blockCellSize(mesh, level - 1, coarse_block));

                const coarse_cell_offset = dof_map.blockOffset(level - 1, coarse_block);
                const coarse_cell_total = dof_map.blockTotal(level - 1, coarse_block);

                const coarse_sys = sys.slice(coarse_cell_offset, coarse_cell_total);

                const coarse_tile = coarse_bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const coarse_origin = Index.scaled(coarse_tile, mesh.tile_width);

                const origin = Index.scaled(tile, mesh.tile_width);

                var cells = IndexSpace.fromSize(Index.splat(mesh.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const globalcell = Index.toSigned(Index.add(origin, cell));
                    const subcell = Index.toSigned(Index.add(coarse_origin, cell));

                    inline for (comptime std.enums.values(System)) |field| {
                        cell_space.setValue(
                            globalcell,
                            block_sys.field(field),
                            coarse_cell_space.prolong(subcell, coarse_sys.field(field)),
                        );
                    }
                }
            }
        }

        pub fn prolongCorrection(
            comptime System: type,
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            level: usize,
            block: usize,
            dest: SystemSlice(System),
            sys: SystemSliceConst(System),
            diff: SystemSliceConst(System),
        ) void {
            if (comptime !system.isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            assert(dest.len == mesh.cell_total);
            assert(sys.len == dof_map.total());
            assert(diff.len == dof_map.total());

            if (level == 0) {
                return;
            }

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, level, block));
            const index_space = IndexSpace.fromSize(cell_space.size);

            const target = mesh.getLevel(level);
            const coarse = mesh.getLevel(level - 1);

            const patch = coarse.parents.items[target.blocks.items(.patch)[block]];
            const patch_bounds = coarse.patches.items(.bounds)[patch];
            const bounds = target.blocks.items(.bounds)[block];

            const patch_space = IndexSpace.fromBox(patch_bounds);
            const block_space = IndexSpace.fromBox(bounds);

            const block_sys = sys.slice(
                dof_map.blockOffset(level, block),
                dof_map.blockTotal(level, block),
            );

            const block_dest = sys.slice(
                mesh.blockCellOffset(level, block),
                mesh.blockCellTotal(level, block),
            );

            const tile_offset = mesh.patchTileOffset(level - 1, patch);

            var tiles = block_space.cartesianIndices();

            while (tiles.next()) |tile| {
                const relative_tile = patch_bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const linear = patch_space.linearFromCartesian(relative_tile);
                const coarse_block = block_map[tile_offset + linear];

                var coarse_bounds = coarse.blocks.items(.bounds)[coarse_block];
                coarse_bounds.refine();

                const coarse_cell_space = CellSpace.fromSize(blockCellSize(mesh, level - 1, coarse_block));

                const coarse_cell_offset = dof_map.blockOffset(level - 1, coarse_block);
                const coarse_cell_total = dof_map.blockTotal(level - 1, coarse_block);

                const coarse_sys = sys.slice(coarse_cell_offset, coarse_cell_total);
                const coarse_diff = diff.slice(coarse_cell_offset, coarse_cell_total);

                const coarse_tile = coarse_bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const coarse_origin = Index.scaled(coarse_tile, mesh.tile_width);

                const origin = Index.scaled(tile, mesh.tile_width);

                var cells = IndexSpace.fromSize(Index.splat(mesh.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const globalcell = Index.toSigned(Index.add(origin, cell));
                    const subcell = Index.toSigned(Index.add(coarse_origin, cell));

                    const lin = index_space.linearFromCartesian(globalcell);

                    inline for (comptime std.enums.values(System)) |field| {
                        const u = coarse_cell_space.prolong(subcell, coarse_sys.field(field));
                        const v = coarse_cell_space.prolong(subcell, coarse_diff.field(field));

                        block_dest.field(field)[lin] = cell_space.value(globalcell, block_sys.field(field)) + u - v;
                    }
                }
            }
        }

        // *************************
        // Apply *******************
        // *************************

        /// Given two dof vectors src and ctx, where both boundaries have been filled at this block,
        /// set the value of a destination dof vector to be the application of the operator on src.
        pub fn apply(
            mesh: *const Mesh,
            dof_map: Map,
            level: usize,
            block: usize,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            assert(dest.len == dof_map.total());
            assert(src.len == dof_map.total());
            assert(ctx.len == dof_map.total());

            const stencil_space = blockStencilSpace(mesh, level, block);
            const cell_offset = dof_map.blockOffset(level, block);
            const cell_total = dof_map.blockTotal(level, block);
            const block_dest = dest.slice(cell_offset, cell_total);
            const block_src = src.slice(cell_offset, cell_total);
            const block_ctx = ctx.slice(cell_offset, cell_total);

            applyImpl(false, stencil_space, oper, block_dest, block_src, block_ctx);
        }

        /// Given two dof vectors src and ctx, where both boundaries have been filled fully at this block,
        /// set the value of a destination dof vector (including an extent O) to be the application of the operator on src.
        pub fn applyFull(
            mesh: *const Mesh,
            dof_map: Map,
            level: usize,
            block: usize,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            assert(dest.len == dof_map.total());
            assert(src.len == dof_map.total());
            assert(ctx.len == dof_map.total());

            const stencil_space = blockStencilSpace(mesh, level, block);
            const cell_offset = dof_map.blockOffset(level, block);
            const cell_total = dof_map.blockTotal(level, block);
            const block_dest = dest.slice(cell_offset, cell_total);
            const block_src = src.slice(cell_offset, cell_total);
            const block_ctx = ctx.slice(cell_offset, cell_total);

            applyImpl(true, stencil_space, oper, block_dest, block_src, block_ctx);
        }

        /// A helper function for applying an operator at a single block.
        fn applyImpl(
            comptime full: bool,
            stencil_space: StencilSpace,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            const T = @TypeOf(oper);

            var cells = if (full) stencil_space.cellSpace().fullCells() else stencil_space.cellSpace().cells();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                const engine = operator.EngineType(N, O, T){
                    .inner = .{
                        .space = stencil_space,
                        .cell = cell,
                    },
                    .ctx = ctx,
                    .sys = src,
                };

                const app = oper.apply(engine);

                inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                    dest.field(field)[linear] = @field(app, @tagName(field));
                }
            }
        }

        // *************************
        // Smoothing ***************
        // *************************

        // TODO make smooth dest a cell vector to be more consistent.

        pub fn smooth(
            mesh: *const Mesh,
            dof_map: Map,
            level: usize,
            block: usize,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
            rhs: SystemSliceConst(@TypeOf(oper).System),
        ) void {
            assert(dest.len == dof_map.total());
            assert(src.len == dof_map.total());
            assert(ctx.len == dof_map.total());
            assert(rhs.len == mesh.cell_total);

            const stencil_space = blockStencilSpace(mesh, level, block);
            const cell_offset = dof_map.blockOffset(level, block);
            const cell_total = dof_map.blockTotal(level, block);
            const block_dest = dest.slice(cell_offset, cell_total);
            const block_src = src.slice(cell_offset, cell_total);
            const block_ctx = ctx.slice(cell_offset, cell_total);
            const block_rhs = ctx.slice(
                mesh.blockCellOffset(level, block),
                mesh.blockCellTotal(level, block),
            );

            smoothImpl(stencil_space, oper, block_dest, block_src, block_ctx, block_rhs);
        }

        /// Runs one iteration of Jacobi's method on the given block, assuming all boundaries have been filled.
        fn smoothImpl(
            stencil_space: StencilSpace,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
            rhs: SystemSliceConst(@TypeOf(oper).System),
        ) void {
            const T = @TypeOf(oper);
            if (!(operator.isMeshOperator(N, O)(T))) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            var cells = stencil_space.cellSpace().cells();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                const engine = EngineType(N, O, T){
                    .inner = .{
                        .space = stencil_space,
                        .cell = cell,
                    },
                    .ctx = ctx,
                    .sys = src,
                };

                const app: system.SystemValue(T.System) = oper.apply(engine);
                const diag: system.SystemValue(T.System) = oper.applyDiagonal(engine);

                inline for (comptime std.enums.values(T.System)) |field| {
                    const f: f64 = stencil_space.value(cell, src.field(field));
                    const a: f64 = @field(app, @tagName(field));
                    const d: f64 = @field(diag, @tagName(field));
                    const r: f64 = rhs.field(field)[linear];

                    stencil_space.setValue(cell, dest.field(field), f + (r - a) / d);
                }
            }
        }

        // *************************
        // Projection **************
        // *************************

        /// Sets the values of a global cell solution vector using a projection function.
        pub fn projectCells(
            mesh: *const Mesh,
            projection: anytype,
            sys: system.SystemSlice(@TypeOf(projection).System),
        ) void {
            const T = @TypeOf(projection);

            if (comptime !isMeshProjection(N)(T)) {
                @compileError("ProjectBase expects projection to satisfy isMeshProjection.");
            }

            for (0..mesh.active_levels) |level| {
                for (0..mesh.getLevel(level).blockTotal()) |block| {
                    const stencil_space = blockStencilSpace(mesh, level, block);
                    const cell_space = stencil_space.cellSpace();

                    const block_dest = sys.slice(mesh.blockCellOffset(level, block), mesh.blockCellTotal(level, block));

                    var cells = cell_space.cells();
                    var linear: usize = 0;

                    while (cells.next()) |cell| : (linear += 1) {
                        const pos: [N]f64 = stencil_space.position(cell);
                        const value: system.SystemValue(T.System) = projection.project(pos);

                        inline for (comptime std.enums.values(T.System)) |field| {
                            block_dest.field(field)[linear] = @field(value, @tagName(field));
                        }
                    }
                }
            }
        }

        // *************************
        // Output ******************
        // *************************

        pub fn writeVtk(comptime System: type, allocator: Allocator, mesh: *const Mesh, sys: system.SystemSliceConst(System), out_stream: anytype) !void {
            if (comptime !system.isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            const field_count = comptime std.enums.values(System).len;

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

            var positions: ArrayListUnmanaged(f64) = .{};
            defer positions.deinit(allocator);

            var vertices: ArrayListUnmanaged(usize) = .{};
            defer vertices.deinit(allocator);

            var fields = [1]ArrayListUnmanaged(f64){.{}} ** field_count;

            defer {
                for (&fields) |*field| {
                    field.deinit(allocator);
                }
            }

            for (0..mesh.active_levels) |level| {
                const target: *const Level = mesh.getLevel(level);
                const level_offset = target.cell_offset;
                const block_offsets = target.blocks.items(.cell_offset);
                const block_totals = target.blocks.items(.cell_total);

                for (block_offsets, block_totals, 0..) |offset, total, block| {
                    const stencil: StencilSpace = blockStencilSpace(mesh, level, block);

                    const cell_size = stencil.size;
                    const point_size = Index.add(cell_size, Index.splat(1));

                    const cell_space: IndexSpace = IndexSpace.fromSize(cell_size);
                    const point_space: IndexSpace = IndexSpace.fromSize(point_size);

                    const point_offset: usize = positions.items.len;

                    try positions.ensureUnusedCapacity(allocator, N * point_space.total());
                    try vertices.ensureUnusedCapacity(allocator, cell_type.nvertices() * cell_space.total());

                    // Fill positions and vertices
                    var points = point_space.cartesianIndices();

                    while (points.next()) |point| {
                        const pos = stencil.vertexPosition(Index.toSigned(point));
                        for (0..N) |i| {
                            positions.appendAssumeCapacity(pos[i]);
                        }
                    }

                    if (N == 1) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(Index.add(cell, Index.splat(1)));

                            vertices.appendAssumeCapacity(point_offset + v1);
                            vertices.appendAssumeCapacity(point_offset + v2);
                        }
                    } else if (N == 2) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(Index.add(cell, [2]usize{ 0, 1 }));
                            const v3: usize = point_space.linearFromCartesian(Index.add(cell, [2]usize{ 1, 1 }));
                            const v4: usize = point_space.linearFromCartesian(Index.add(cell, [2]usize{ 1, 0 }));

                            vertices.appendAssumeCapacity(point_offset + v1);
                            vertices.appendAssumeCapacity(point_offset + v2);
                            vertices.appendAssumeCapacity(point_offset + v3);
                            vertices.appendAssumeCapacity(point_offset + v4);
                        }
                    } else if (N == 3) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 0, 1, 0 }));
                            const v3: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 1, 0 }));
                            const v4: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 0, 0 }));
                            const v5: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 0, 0, 1 }));
                            const v6: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 0, 1, 3 }));
                            const v7: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 1, 3 }));
                            const v8: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 0, 3 }));

                            vertices.appendAssumeCapacity(point_offset + v1);
                            vertices.appendAssumeCapacity(point_offset + v2);
                            vertices.appendAssumeCapacity(point_offset + v3);
                            vertices.appendAssumeCapacity(point_offset + v4);
                            vertices.appendAssumeCapacity(point_offset + v5);
                            vertices.appendAssumeCapacity(point_offset + v6);
                            vertices.appendAssumeCapacity(point_offset + v7);
                            vertices.appendAssumeCapacity(point_offset + v8);
                        }
                    }

                    for (&fields) |*field| {
                        try field.ensureUnusedCapacity(allocator, cell_space.total());
                    }

                    const block_sys = sys.slice(level_offset + offset, total);

                    for (0..total) |linear| {
                        inline for (comptime std.enums.values(System), 0..) |field, idx| {
                            fields[idx].appendAssumeCapacity(block_sys.field(field)[linear]);
                        }
                    }
                }
            }

            var grid: VtuMeshOutput = try VtuMeshOutput.init(allocator, .{
                .points = positions.items,
                .vertices = vertices.items,
                .cell_type = cell_type,
            });
            defer grid.deinit();

            inline for (comptime std.meta.fieldNames(System), 0..) |name, id| {
                try grid.addCellField(name, fields[id].items, 1);
            }

            try grid.write(out_stream);
        }
    };
}
