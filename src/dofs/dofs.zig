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

        const Self = @This();
        const Mesh = meshes.Mesh(N);
        const Index = index.Index(N);
        const CellSpace = basis.CellSpace(N, O);

        pub fn init(allocator: Allocator, mesh: *const Mesh) !Self {
            const offsets = try allocator.alloc(usize, mesh.blocks.len + 1);
            errdefer allocator.free(offsets);

            {
                offsets[0] = 0;

                var cur: usize = 0;

                for (mesh.blocks) |block| {
                    offsets[cur + 1] = offsets[cur] + CellSpace.fromSize(Index.scaled(block.bounds.size, mesh.tile_width)).total();
                    cur += 1;
                }
            }

            return .{
                .offsets = offsets,
            };
        }

        pub fn deinit(self: Self, allocator: Allocator) void {
            allocator.free(self.offsets);
        }

        pub fn offset(self: Self, block: usize) usize {
            return self.offsets[block];
        }

        pub fn total(self: Self, block: usize) usize {
            return self.offsets[block + 1] - self.offsets[block];
        }

        pub fn ndofs(self: Self) usize {
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
        const BoundaryUtils = boundaries.BoundaryUtils(N, O);
        const Face = geometry.Face(N);
        const Region = geometry.Region(N);
        const Map = DofMap(N, O);

        /// Computes the number of cells along each axis for a given block.
        pub fn blockCellSize(mesh: *const Mesh, block: usize) [N]usize {
            return Index.scaled(mesh.blocks[block].bounds.size, mesh.tile_width);
        }

        /// Builds a stencil space for the given block.
        pub fn blockStencilSpace(mesh: *const Mesh, block: usize) StencilSpace {
            return .{
                .physical_bounds = mesh.blockPhysicalBounds(block),
                .size = blockCellSize(mesh, block),
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

        /// Copies the data from a vector of cells to a vector of dofs.
        pub fn copyDofsFromCells(
            comptime System: type,
            mesh: *const Mesh,
            dof_map: Map,
            block: usize,
            dest: SystemSlice(System),
            src: SystemSliceConst(System),
        ) void {
            assert(dest.len == dof_map.ndofs());
            assert(src.len == mesh.cell_total);

            const block_dest = dest.slice(dof_map.offset(block), dof_map.total(block));
            const block_src = src.slice(mesh.blocks[block].cell_offset, mesh.blocks[block].cell_total);

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, block));

            var cells = cell_space.cells();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                inline for (comptime std.enums.values(System)) |field| {
                    cell_space.setValue(cell, block_dest.field(field), block_src.field(field)[linear]);
                }
            }
        }

        /// Copies the data from a vector of dofs to a vector of cells.
        pub fn copyCellsFromDofs(
            comptime System: type,
            mesh: *const Mesh,
            dof_map: Map,
            block: usize,
            dest: SystemSlice(System),
            src: SystemSliceConst(System),
        ) void {
            assert(src.len == dof_map.ndofs());
            assert(dest.len == mesh.cell_total);

            const block_src = src.slice(dof_map.offset(block), dof_map.total(block));
            const block_dest = dest.slice(mesh.blocks[block].cell_offset, mesh.blocks[block].cell_total);

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, block));

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
            block: usize,
            dest: SystemSlice(System),
            src: SystemSliceConst(System),
        ) void {
            assert(dest.len == dof_map.ndofs());
            assert(src.len == dof_map.ndofs());

            const cell_offset = dof_map.offset(block);
            const cell_total = dof_map.total(block);

            inline for (comptime std.enums.values(System)) |field| {
                @memcpy(dest.field(field)[cell_offset .. cell_offset + cell_total], src.field(field)[cell_offset .. cell_offset + cell_total]);
            }
        }

        /// Copies data from one cell vector to another at the given block.
        pub fn copyCells(
            comptime System: type,
            mesh: *const Mesh,
            block: usize,
            dest: SystemSlice(System),
            src: SystemSliceConst(System),
        ) void {
            assert(dest.len == mesh.cell_total);
            assert(src == mesh.cell_total);

            const cell_offset = mesh.blocks[block].cell_offset;
            const cell_total = mesh.blocks[block].cell_total;

            inline for (comptime std.enums.values(System)) |field| {
                @memcpy(dest.field(field)[cell_offset .. cell_offset + cell_total], src.field(field)[cell_offset .. cell_offset + cell_total]);
            }
        }

        // ************************
        // Fill Ops ***************
        // ************************

        /// Given a dof vector, fill all boundary dofs on that vector out to an extent O using the
        /// supplied boundary conditions.
        pub fn fillBoundary(
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            block: usize,
            boundary: anytype,
            sys: system.SystemSlice(@TypeOf(boundary).System),
        ) void {
            fillBoundaryToExtent(O, mesh, block_map, dof_map, block, boundary, sys);
        }

        /// Given a dof vector, fill all boundary dofs on that vector out to an extent 2*O using the
        /// supplied boundary conditions.
        pub fn fillBoundaryFull(
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            block: usize,
            boundary: anytype,
            sys: system.SystemSlice(@TypeOf(boundary).System),
        ) void {
            fillBoundaryToExtent(2 * O, mesh, block_map, dof_map, block, boundary, sys);
        }

        /// Internal helper function for filling boundary dofs to some extent E.
        fn fillBoundaryToExtent(
            comptime E: usize,
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            block: usize,
            boundary: anytype,
            sys: system.SystemSlice(@TypeOf(boundary).System),
        ) void {
            const T = @TypeOf(boundary);

            if (comptime !isSystemBoundary(N)(T)) {
                @compileError("FillBaseBoundary requires boundary satisfy isSystemBoundary trait.");
            }

            assert(sys.len == dof_map.ndofs());

            const bounds: IndexBox = mesh.blocks[block].bounds;
            const index_size = mesh.levels[mesh.patches[mesh.blocks[block].patch].level].index_size;

            const regions = comptime Region.orderedRegions();

            inline for (comptime regions[1..]) |region| {
                var exterior: bool = false;

                for (0..N) |i| {
                    if (region.sides[i] == .left and bounds.origin[i] == 0) {
                        exterior = true;
                    } else if (region.sides[i] == .right and bounds.origin[i] + bounds.size[i] == index_size[i]) {
                        exterior = true;
                    }
                }

                if (exterior) {
                    const stencil_space = blockStencilSpace(mesh, block);

                    const block_sys = sys.slice(
                        dof_map.offset(block),
                        dof_map.total(block),
                    );

                    BoundaryUtils.fillBoundaryRegion(E, region, stencil_space, boundary, block_sys);
                } else {
                    fillInteriorBoundary(T.System, region, E, mesh, block_map, dof_map, block, sys);
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
            block_id: usize,
            sys: system.SystemSlice(System),
        ) void {
            const block = mesh.blocks[block_id];
            const block_cell_space = CellSpace.fromSize(blockCellSize(mesh, block_id));

            const patch_id = mesh.blocks[block_id].patch;
            const patch = mesh.patches[patch_id];
            const patch_space = IndexSpace.fromBox(patch.bounds);
            const patch_block_map: []const usize = block_map[patch.tile_offset .. patch.tile_offset + patch.tile_total];

            const relative_bounds: IndexBox = block.bounds.relativeTo(patch.bounds);

            const block_sys = sys.slice(
                dof_map.offset(block_id),
                dof_map.total(block_id),
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
                const neighbor_id: usize = patch_block_map[patch_space.linearFromCartesian(buffer_tile)];

                if (neighbor_id == maxInt(usize)) {
                    const coarse_buffer_tile: [N]usize = Index.coarsened(buffer_tile);

                    // If so cache various coarse variables
                    const coarse_patch = mesh.patches[neighbor_id];
                    const coarse_patch_block_map: []const usize = block_map[coarse_patch.tile_offset .. coarse_patch.tile_offset + coarse_patch.tile_total];
                    const coarse_patch_space = IndexSpace.fromBox(coarse_patch.bounds);

                    const coarse_block_id = coarse_patch_block_map[coarse_patch_space.linearFromCartesian(coarse_buffer_tile)];
                    const coarse_block = mesh.blocks[coarse_block_id];
                    const coarse_block_cell_space: CellSpace = CellSpace.fromSize(coarse_block.bounds.size);

                    const coarse_block_sys = sys.slice(
                        dof_map.offset(coarse_block_id),
                        dof_map.total(coarse_block_id),
                    );

                    const coarse_relative_bounds = coarse_block.bounds.relativeTo(coarse_patch.bounds).refined();

                    // Neighbor origin in subcell space
                    const coarse_neighbor_origin: [N]usize = Index.scaled(coarse_relative_bounds.localFromGlobal(relative_tile), mesh.tile_width);

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
                        dof_map.offset(neighbor_id),
                        dof_map.total(neighbor_id),
                    );

                    const neighbor = mesh.blocks[neighbor_id];
                    const neighbor_cell_space: CellSpace = CellSpace.fromSize(neighbor.bounds.size);

                    const neighbor_origin: [N]usize = Index.scaled(neighbor.bounds.localFromGlobal(relative_tile), mesh.tile_width);

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
            block_id: usize,
            sys: SystemSlice(System),
        ) void {
            if (comptime !system.isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            assert(sys.len == dof_map.ndofs());

            const block_sys = sys.slice(
                dof_map.offset(block_id),
                dof_map.total(block_id),
            );

            const block = mesh.blocks[block_id];
            const patch = mesh.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, block_id));

            const coarse_patch = mesh.patches[patch.parent.?];

            const patch_bounds = coarse_patch.bounds;
            const bounds = block.bounds.coarsened();

            const patch_space = IndexSpace.fromBox(patch_bounds);
            const block_space = IndexSpace.fromBox(bounds);

            var tiles = block_space.cartesianIndices();

            while (tiles.next()) |tile| {
                const relative_tile = patch.bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const linear = patch_space.linearFromCartesian(relative_tile);
                const coarse_block_id = block_map[coarse_patch.tile_offset + linear];

                const coarse_block = mesh.blocks[coarse_block_id];

                const coarse_cell_space = CellSpace.fromSize(blockCellSize(mesh, coarse_block_id));

                const coarse_dofs_offset = dof_map.offset(coarse_block_id);
                const coarse_dofs_total = dof_map.total(coarse_block_id);

                const coarse_tile = coarse_block.bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const coarse_origin = Index.scaled(coarse_tile, mesh.tile_width);

                const origin = Index.scaled(tile, mesh.tile_width);

                const coarse_sys = sys.slice(coarse_dofs_offset, coarse_dofs_total);

                var cells = IndexSpace.fromSize(Index.splat(mesh.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const super_cell = Index.toSigned(Index.add(origin, cell));
                    const coarse_cell = Index.toSigned(Index.add(coarse_origin, cell));

                    inline for (comptime std.enums.values(System)) |field| {
                        coarse_cell_space.setValue(coarse_cell, coarse_sys.field(field), cell_space.restrict(super_cell, block_sys.field(field)));
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
            block_id: usize,
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
            assert(src.len == dof_map.ndofs());
            assert(ctx.len == dof_map.ndofs());
            assert(b.len == dof_map.ndofs());

            const block_b = b.slice(
                dof_map.offset(block_id),
                dof_map.total(block_id),
            );

            const block = mesh.blocks[block_id];
            const patch = mesh.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, block_id));

            const coarse_patch = mesh.patches[patch.parent.?];

            const patch_bounds = coarse_patch.bounds;
            const bounds = block.bounds.coarsened();

            const patch_space = IndexSpace.fromBox(patch_bounds);
            const block_space = IndexSpace.fromBox(bounds);

            var tiles = block_space.cartesianIndices();

            while (tiles.next()) |tile| {
                const relative_tile = patch.bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const linear = patch_space.linearFromCartesian(relative_tile);
                const coarse_block_id = block_map[coarse_patch.tile_offset + linear];

                const coarse_block = mesh.blocks[coarse_block_id];

                const coarse_stencil_space = blockStencilSpace(mesh, coarse_block_id);
                const coarse_index_space = IndexSpace.fromSize(coarse_stencil_space.size);

                const coarse_dofs_offset = dof_map.offset(coarse_block_id);
                const coarse_dofs_total = dof_map.total(coarse_block_id);
                const coarse_cell_offset = coarse_block.cell_offset;
                const coarse_cell_total = coarse_block.cell_total;

                const coarse_tile = coarse_block.bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const coarse_origin = Index.scaled(coarse_tile, mesh.tile_width);

                const origin = Index.scaled(tile, mesh.tile_width);

                const coarse_src = src.slice(coarse_dofs_offset, coarse_dofs_total);
                const coarse_ctx = ctx.slice(coarse_dofs_offset, coarse_dofs_total);

                const coarse_dest = dest.slice(
                    coarse_cell_offset,
                    coarse_cell_total,
                );

                var cells = IndexSpace.fromSize(Index.splat(mesh.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const supercell = Index.toSigned(Index.add(origin, cell));
                    const coarsecell = Index.toSigned(Index.add(coarse_origin, cell));

                    const lin = coarse_index_space.linearFromCartesian(Index.toUnsigned(coarsecell));

                    const engine = operator.EngineType(N, O, T){
                        .inner = .{
                            .space = coarse_stencil_space,
                            .cell = coarsecell,
                        },
                        .ctx = coarse_ctx,
                        .sys = coarse_src,
                    };

                    const app = oper.apply(engine);

                    inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                        const b_val = cell_space.restrict(supercell, block_b.field(field));
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
            block_id: usize,
            sys: SystemSlice(System),
        ) void {
            if (comptime !system.isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            assert(sys.len == dof_map.total());

            const block = mesh.blocks[block_id];
            const patch = mesh.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, block_id));

            const coarse_patch = mesh.patches[patch.parent.?];

            const patch_space = IndexSpace.fromBox(patch.bounds);
            const block_space = IndexSpace.fromBox(block.bounds);

            const block_sys = sys.slice(
                dof_map.offset(block_id),
                dof_map.total(block_id),
            );

            var tiles = block_space.cartesianIndices();

            while (tiles.next()) |tile| {
                const relative_tile = patch.bounds.localFromGlobal(block.bounds.globalFromLocal(tile));
                const linear = patch_space.linearFromCartesian(relative_tile);
                const coarse_block_id = block_map[coarse_patch.tile_offset + linear];

                const coarse_block = mesh.blocks[coarse_block_id];

                const coarse_cell_space = CellSpace.fromSize(blockCellSize(mesh, coarse_block_id));

                const coarse_cell_offset = dof_map.offset(coarse_block_id);
                const coarse_cell_total = dof_map.total(coarse_block_id);

                const coarse_sys = sys.slice(coarse_cell_offset, coarse_cell_total);

                const coarse_tile = coarse_block.bounds.refined().localFromGlobal(block.bounds.globalFromLocal(tile));
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

        /// Prolongs a correction to the given block. Sys and diff must be dof vectors filled on l - 1. Dest is a cell vector.
        pub fn prolongCorrection(
            comptime System: type,
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: Map,
            block_id: usize,
            dest: SystemSlice(System),
            sys: SystemSliceConst(System),
            diff: SystemSliceConst(System),
        ) void {
            if (comptime !system.isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            assert(dest.len == mesh.cell_total);
            assert(sys.len == dof_map.ndofs());
            assert(diff.len == dof_map.ndofs());

            const block = mesh.blocks[block_id];
            const patch = mesh.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const cell_space = CellSpace.fromSize(blockCellSize(mesh, block_id));

            const coarse_patch = mesh.patches[patch.parent.?];

            const patch_space = IndexSpace.fromBox(patch.bounds);
            const block_space = IndexSpace.fromBox(block.bounds);

            const block_sys = sys.slice(
                dof_map.offset(block_id),
                dof_map.total(block_id),
            );

            const block_dest = dest.slice(
                block.cell_offset,
                block.cell_total,
            );

            var tiles = block_space.cartesianIndices();

            while (tiles.next()) |tile| {
                const relative_tile = patch.bounds.localFromGlobal(block.bounds.globalFromLocal(tile));
                const linear = patch_space.linearFromCartesian(relative_tile);
                const coarse_block_id = block_map[coarse_patch.tile_offset + linear];

                const coarse_block = mesh.blocks[coarse_block_id];

                const coarse_cell_space = CellSpace.fromSize(blockCellSize(mesh, coarse_block_id));

                const coarse_cell_offset = dof_map.offset(coarse_block_id);
                const coarse_cell_total = dof_map.total(coarse_block_id);

                const coarse_sys = sys.slice(coarse_cell_offset, coarse_cell_total);
                const coarse_diff = diff.slice(coarse_cell_offset, coarse_cell_total);

                const coarse_tile = coarse_block.bounds.refined().localFromGlobal(block.bounds.globalFromLocal(tile));
                const coarse_origin = Index.scaled(coarse_tile, mesh.tile_width);

                const origin = Index.scaled(tile, mesh.tile_width);

                var cells = IndexSpace.fromSize(Index.splat(mesh.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const global_cell = Index.toSigned(Index.add(origin, cell));
                    const sub_cell = Index.toSigned(Index.add(coarse_origin, cell));

                    const lin = block_space.linearFromCartesian(Index.toUnsigned(global_cell));

                    inline for (comptime std.enums.values(System)) |field| {
                        const u = coarse_cell_space.prolong(sub_cell, coarse_sys.field(field));
                        const v = coarse_cell_space.prolong(sub_cell, coarse_diff.field(field));

                        block_dest.field(field)[lin] = cell_space.value(global_cell, block_sys.field(field)) + u - v;
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
            block_id: usize,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            assert(dest.len == dof_map.ndofs());
            assert(src.len == dof_map.ndofs());
            assert(ctx.len == dof_map.ndofs());

            const stencil_space = blockStencilSpace(mesh, block_id);
            const dof_offset = dof_map.offset(block_id);
            const dof_total = dof_map.total(block_id);
            const block_dest = dest.slice(dof_offset, dof_total);
            const block_src = src.slice(dof_offset, dof_total);
            const block_ctx = ctx.slice(dof_offset, dof_total);

            applyImpl(false, stencil_space, oper, block_dest, block_src, block_ctx);
        }

        /// Given two dof vectors src and ctx, where both boundaries have been filled fully at this block,
        /// set the value of a destination dof vector (including an extent O) to be the application of the operator on src.
        pub fn applyFull(
            mesh: *const Mesh,
            dof_map: Map,
            block_id: usize,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            assert(dest.len == dof_map.ndofs());
            assert(src.len == dof_map.ndofs());
            assert(ctx.len == dof_map.ndofs());

            const stencil_space = blockStencilSpace(mesh, block_id);
            const dof_offset = dof_map.offset(block_id);
            const dof_total = dof_map.total(block_id);
            const block_dest = dest.slice(dof_offset, dof_total);
            const block_src = src.slice(dof_offset, dof_total);
            const block_ctx = ctx.slice(dof_offset, dof_total);

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

            var cells = if (full) stencil_space.cellSpace().cellsToExtent(O) else stencil_space.cellSpace().cells();
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
            block_id: usize,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
            rhs: SystemSliceConst(@TypeOf(oper).System),
        ) void {
            assert(dest.len == dof_map.ndofs());
            assert(src.len == dof_map.ndofs());
            assert(ctx.len == dof_map.ndofs());
            assert(rhs.len == mesh.cell_total);

            const block = mesh.blocks[block_id];

            const stencil_space = blockStencilSpace(mesh, block_id);
            const dof_offset = dof_map.offset(block_id);
            const dof_total = dof_map.total(block_id);
            const block_dest = dest.slice(dof_offset, dof_total);
            const block_src = src.slice(dof_offset, dof_total);
            const block_ctx = ctx.slice(dof_offset, dof_total);
            const block_rhs = rhs.slice(
                block.cell_offset,
                block.cell_total,
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
            if (comptime !operator.isMeshOperator(N, O)(T)) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            const cell_space = stencil_space.cellSpace();

            var cells = cell_space.cells();
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
                    const f: f64 = cell_space.value(cell, src.field(field));
                    const a: f64 = @field(app, @tagName(field));
                    const d: f64 = @field(diag, @tagName(field));
                    const r: f64 = rhs.field(field)[linear];

                    cell_space.setValue(cell, dest.field(field), f + (r - a) / d);
                }
            }
        }

        // *************************
        // Residual ****************
        // *************************

        pub fn residualNorm(
            mesh: *const Mesh,
            dof_map: Map,
            oper: anytype,
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
            rhs: SystemSliceConst(@TypeOf(oper).System),
        ) system.SystemValue(@TypeOf(oper).System) {
            var result: system.SystemValue(@TypeOf(oper).System) = undefined;

            inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                @field(result, @tagName(field)) = 0.0;
            }

            for (0..mesh.blocks.len) |block| {
                const norm = residualNormSq(mesh, dof_map, block, oper, src, ctx, rhs);

                inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                    @field(result, @tagName(field)) += @field(norm, @tagName(field));
                }
            }

            inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                @field(result, @tagName(field)) = @sqrt(@field(result, @tagName(field)));
            }

            return result;
        }

        fn residualNormSq(
            mesh: *const Mesh,
            dof_map: Map,
            block_id: usize,
            oper: anytype,
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
            rhs: SystemSliceConst(@TypeOf(oper).System),
        ) system.SystemValue(@TypeOf(oper).System) {
            assert(rhs.len == mesh.cell_total);
            assert(src.len == dof_map.ndofs());
            assert(ctx.len == dof_map.ndofs());

            var result: system.SystemValue(@TypeOf(oper).System) = undefined;

            inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                @field(result, @tagName(field)) = 0.0;
            }

            const block = mesh.blocks[block_id];

            const stencil_space = blockStencilSpace(mesh, block_id);
            const cell_offset = dof_map.offset(block_id);
            const cell_total = dof_map.total(block_id);
            const block_src = src.slice(cell_offset, cell_total);
            const block_ctx = ctx.slice(cell_offset, cell_total);
            const block_rhs = rhs.slice(block.cell_offset, block.cell_total);

            var cells = stencil_space.cellSpace().cells();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                const engine = operator.EngineType(N, O, @TypeOf(oper)){
                    .inner = .{
                        .space = stencil_space,
                        .cell = cell,
                    },
                    .ctx = block_ctx,
                    .sys = block_src,
                };

                const app = oper.apply(engine);

                inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                    const a = @field(app, @tagName(field));
                    const b = block_rhs.field(field)[linear];
                    @field(result, @tagName(field)) += (b - a) * (b - a);
                }
            }

            return result;
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

            for (0..mesh.blocks.len) |block_id| {
                const block = mesh.blocks[block_id];

                const stencil_space = blockStencilSpace(mesh, block_id);
                const cell_space = stencil_space.cellSpace();

                const block_dest = sys.slice(block.cell_offset, block.cell_total);

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

            // Temporary
            // TODO fix to print all exposed blocks

            const top_level = mesh.levels[mesh.levels.len - 1];

            for (top_level.block_offset..top_level.block_offset + top_level.block_total) |block_id| {
                const block = mesh.blocks[block_id];

                const stencil: StencilSpace = blockStencilSpace(mesh, block_id);

                const cell_size = stencil.size;
                const point_size = Index.add(cell_size, Index.splat(1));

                const cell_space: IndexSpace = IndexSpace.fromSize(cell_size);
                const point_space: IndexSpace = IndexSpace.fromSize(point_size);

                const point_offset: usize = positions.items.len / N;

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

                const block_sys = sys.slice(block.cell_offset, block.cell_total);

                for (0..block.cell_total) |linear| {
                    inline for (comptime std.enums.values(System), 0..) |field, idx| {
                        fields[idx].appendAssumeCapacity(block_sys.field(field)[linear]);
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
