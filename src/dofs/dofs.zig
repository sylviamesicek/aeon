//! A module for managing and handling degrees of freedom defined on a mesh,
//! transforming cell bases representations of functions to dof based ones,
//! filling ghost dofs, restricting, prolonging, smoothing, and applying operators.

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
const NodeSpace = basis.NodeSpace;
const StencilSpace = basis.StencilSpace;

const geometry = @import("../geometry/geometry.zig");
const Face = geometry.Face;

const meshes = @import("../mesh/mesh.zig");

const system = @import("../system.zig");
const SystemSlice = system.SystemSlice;
const SystemSliceConst = system.SystemSliceConst;
const SystemValue = system.SystemValue;
const EmptySystem = system.EmptySystem;
const isSystem = system.isSystem;

// submodules

const boundaries = @import("boundary.zig");
const engines = @import("engine.zig");

// *********************************
// Public API **********************
// *********************************

pub const BoundaryCondition = boundaries.BoundaryCondition;
pub const BoundaryUtils = boundaries.BoundaryUtils;
pub const SystemBoundaryCondition = boundaries.SystemBoundaryCondition;
pub const isSystemBoundary = boundaries.isSystemBoundary;

pub const Engine = engines.Engine;
pub const EngineSetting = engines.EngineSetting;

/// A map from each block to a corresponding dof offset and total. This
/// is similar to the `cell_offset`/`cell_total` field of a `Block(N)`, but
/// accounts for up to 4*O additional ghost dofs along each axis.
pub fn DofMap(comptime N: usize, comptime O: usize) type {
    return struct {
        offsets: []const usize,

        const Self = @This();
        const Mesh = meshes.Mesh(N);
        const Index = geometry.Index(N);

        pub fn init(allocator: Allocator, mesh: *const Mesh) !Self {
            const offsets = try allocator.alloc(usize, mesh.blocks.len + 1);
            errdefer allocator.free(offsets);

            {
                offsets[0] = 0;

                var cur: usize = 0;

                for (mesh.blocks) |block| {
                    offsets[cur + 1] = offsets[cur] + NodeSpace(N, O).fromSize(Index.scaled(block.bounds.size, mesh.tile_width)).total();
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
        const Map = DofMap(N, O);

        const Index = geometry.Index(N);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexBox = geometry.Box(N, usize);
        const Region = geometry.Region(N);
        const Mesh = meshes.Mesh(N);

        const RO = 0;
        const PO = O;

        /// Computes the number of cells along each axis for a given block.
        pub fn blockCellSize(mesh: *const Mesh, block: usize) [N]usize {
            return Index.scaled(mesh.blocks[block].bounds.size, mesh.tile_width);
        }

        /// Builds a stencil space for the given block.
        pub fn blockStencilSpace(mesh: *const Mesh, block: usize) StencilSpace(N, O) {
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

                pub fn boundary(self: @This(), pos: [N]f64, face: Face(N)) SystemBoundaryCondition(System) {
                    return self.oper.boundarySys(pos, face);
                }
            };
        }

        /// Extracts the context boundary conditions from an operator.
        pub fn OperContextBoundary(comptime T: type) type {
            if (comptime !(isSystemOperator(N, O)(T))) {
                @compileError("Oper must satisfy isMeshOperator traits.");
            }

            return struct {
                oper: T,

                pub const System = T.Context;

                pub fn boundary(self: @This(), pos: [N]f64, face: Face(N)) SystemBoundaryCondition(System) {
                    return self.oper.boundaryCtx(pos, face);
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
            dest.assertLen(dof_map.ndofs());
            src.assertLen(mesh.cell_total);

            const block_dest = dest.slice(dof_map.offset(block), dof_map.total(block));
            const block_src = src.slice(mesh.blocks[block].cell_offset, mesh.blocks[block].cell_total);

            const node_space = NodeSpace(N, O).fromSize(blockCellSize(mesh, block));

            var nodes = node_space.nodes(0);
            var linear: usize = 0;

            while (nodes.next()) |node| : (linear += 1) {
                inline for (comptime std.enums.values(System)) |field| {
                    node_space.setValue(node, block_dest.field(field), block_src.field(field)[linear]);
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
            dest.assertLen(mesh.cell_total);
            src.assertLen(dof_map.ndofs());

            const block_src = src.slice(dof_map.offset(block), dof_map.total(block));
            const block_dest = dest.slice(mesh.blocks[block].cell_offset, mesh.blocks[block].cell_total);

            const node_space = NodeSpace(N, O).fromSize(blockCellSize(mesh, block));

            var nodes = node_space.nodes(0);
            var linear: usize = 0;

            while (nodes.next()) |node| : (linear += 1) {
                inline for (comptime std.enums.values(System)) |field| {
                    block_dest.field(field)[linear] = node_space.value(node, block_src.field(field));
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
            dest.assertLen(dof_map.ndofs());
            src.assertLen(dof_map.ndofs());

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
            dest.assertLen(mesh.cell_total);
            src.assertLen(mesh.cell_total);

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
            dof_map: Map,
            block: usize,
            boundary: anytype,
            sys: system.SystemSlice(@TypeOf(boundary).System),
        ) void {
            fillBoundaryToExtent(O, mesh, dof_map, block, boundary, sys);
        }

        /// Internal helper function for filling boundary dofs to some extent E.
        fn fillBoundaryToExtent(
            comptime E: usize,
            mesh: *const Mesh,
            dof_map: Map,
            block: usize,
            boundary: anytype,
            sys: system.SystemSlice(@TypeOf(boundary).System),
        ) void {
            const T = @TypeOf(boundary);

            if (comptime !isSystemBoundary(N)(T)) {
                @compileError("FillBaseBoundary requires boundary satisfy isSystemBoundary trait.");
            }

            sys.assertLen(dof_map.ndofs());

            const bounds: IndexBox = mesh.blocks[block].bounds;
            const index_size = mesh.levels[mesh.patches[mesh.blocks[block].patch].level].index_size;

            const regions = comptime Region.orderedRegions();

            inline for (comptime regions[1..]) |region| {
                var exterior: bool = true;

                for (0..N) |i| {
                    if (region.sides[i] == .left and bounds.origin[i] != 0) {
                        exterior = false;
                    } else if (region.sides[i] == .right and bounds.origin[i] + bounds.size[i] != index_size[i]) {
                        exterior = false;
                    }
                }

                if (exterior) {
                    const stencil_space = blockStencilSpace(mesh, block);

                    const block_sys = sys.slice(
                        dof_map.offset(block),
                        dof_map.total(block),
                    );

                    BoundaryUtils(N, O).fillBoundaryRegion(E, region, stencil_space, boundary, block_sys);
                } else {
                    fillInteriorBoundary(T.System, region, E, mesh, dof_map, block, sys);
                }
            }
        }

        /// Fills non physical boundaries (ie boundaries within the numerical domain between two blocks).
        fn fillInteriorBoundary(
            comptime System: type,
            comptime region: Region,
            comptime E: usize,
            mesh: *const Mesh,
            dof_map: Map,
            block_id: usize,
            sys: system.SystemSlice(System),
        ) void {
            const block_map = mesh.block_map;

            const block = mesh.blocks[block_id];
            const block_cell_space = NodeSpace(N, O).fromSize(blockCellSize(mesh, block_id));

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
                    const coarse_block_cell_space = NodeSpace(N, O).fromSize(coarse_block.bounds.size);

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
                        const block_cell: [N]isize = NodeSpace(N, O).offsetFromOrigin(origin, ind);
                        // Cell in neighbor in subcell space
                        const neighbor_cell: [N]isize = NodeSpace(N, O).offsetFromOrigin(coarse_neighbor_origin, ind);

                        inline for (comptime std.enums.values(System)) |field| {
                            block_cell_space.setValue(
                                block_cell,
                                block_sys.field(field),
                                coarse_block_cell_space.prolong(
                                    PO,
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
                    const neighbor_cell_space = NodeSpace(N, O).fromSize(neighbor.bounds.size);

                    const neighbor_origin: [N]usize = Index.scaled(neighbor.bounds.localFromGlobal(relative_tile), mesh.tile_width);

                    var indices = region.cartesianIndices(E, Index.splat(mesh.tile_width));

                    while (indices.next()) |idx| {
                        const block_cell: [N]isize = NodeSpace(N, O).offsetFromOrigin(origin, idx);
                        const neighbor_cell: [N]isize = NodeSpace(N, O).offsetFromOrigin(neighbor_origin, idx);

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
            dof_map: Map,
            block_id: usize,
            sys: SystemSlice(System),
        ) void {
            if (comptime !system.isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            sys.assertLen(dof_map.ndofs());

            const block_sys = sys.slice(
                dof_map.offset(block_id),
                dof_map.total(block_id),
            );

            const block = mesh.blocks[block_id];
            const patch = mesh.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const cell_space = NodeSpace(N, O).fromSize(blockCellSize(mesh, block_id));

            const coarse_patch = mesh.patches[patch.parent.?];

            const bounds = block.bounds.coarsened();

            const patch_space = IndexSpace.fromBox(coarse_patch.bounds);

            var tiles = IndexSpace.fromBox(bounds).cartesianIndices();

            while (tiles.next()) |tile| {
                const relative_tile = coarse_patch.bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const linear = patch_space.linearFromCartesian(relative_tile);
                const coarse_block_id = mesh.block_map[coarse_patch.tile_offset + linear];
                const coarse_block = mesh.blocks[coarse_block_id];

                const coarse_cell_space = NodeSpace(N, O).fromSize(blockCellSize(mesh, coarse_block_id));

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
                        const v = cell_space.restrict(RO, super_cell, block_sys.field(field));
                        coarse_cell_space.setValue(coarse_cell, coarse_sys.field(field), v);
                    }
                }
            }
        }

        /// Given a set of dof vectors src, ctx, and b, where src and ctx are filled at all boundaries on level - 1 and b is filled at the
        /// given block. For each underlying dof set the value of the cell vector at this dof to be the restricted value of b minus the
        /// application of the operator at the underlying dof.
        pub fn restrictRhs(
            mesh: *const Mesh,
            dof_map: Map,
            block_id: usize,
            oper: anytype,
            rhs: SystemSlice(@TypeOf(oper).System),
            res: SystemSliceConst(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            const T = @TypeOf(oper);

            if (comptime !isSystemOperator(N, O)(T)) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            rhs.assertLen(mesh.cell_total);
            res.assertLen(mesh.cell_total);
            src.assertLen(dof_map.ndofs());
            ctx.assertLen(dof_map.ndofs());

            const block = mesh.blocks[block_id];
            const patch = mesh.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const block_res = res.toConst().slice(
                block.cell_offset,
                block.cell_total,
            );

            const cell_space = NodeSpace(N, 0).fromSize(blockCellSize(mesh, block_id));

            const coarse_patch = mesh.patches[patch.parent.?];

            const bounds = block.bounds.coarsened();

            const patch_space = IndexSpace.fromBox(coarse_patch.bounds);

            var tiles = IndexSpace.fromBox(bounds).cartesianIndices();

            while (tiles.next()) |tile| {
                const relative_tile = coarse_patch.bounds.localFromGlobal(bounds.globalFromLocal(tile));
                const linear = patch_space.linearFromCartesian(relative_tile);
                const coarse_block_id = mesh.block_map[coarse_patch.tile_offset + linear];

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

                const coarse_rhs = rhs.slice(
                    coarse_cell_offset,
                    coarse_cell_total,
                );

                var cells = IndexSpace.fromSize(Index.splat(mesh.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const supercell = Index.toSigned(Index.add(origin, cell));
                    const coarsecell = Index.toSigned(Index.add(coarse_origin, cell));

                    const lin = coarse_index_space.linearFromCartesian(Index.toUnsigned(coarsecell));

                    const coarse_engine = Engine(N, O, .normal, T.System, T.Context).new(
                        coarse_stencil_space,
                        coarsecell,
                        coarse_src,
                        coarse_ctx,
                    );

                    const app = oper.apply(.normal, coarse_engine);

                    inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                        const res_val = cell_space.restrict(RO, supercell, block_res.field(field));
                        const a_val = @field(app, @tagName(field));
                        coarse_rhs.field(field)[lin] = res_val + a_val;
                    }
                }
            }
        }

        /// Given a global dof vector with correct boundary dofs on the lower level, prolong the data to this block.
        pub fn prolong(
            comptime System: type,
            mesh: *const Mesh,
            dof_map: Map,
            block_id: usize,
            sys: SystemSlice(System),
        ) void {
            if (comptime !system.isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            sys.assertLen(dof_map.ndofs());

            const block = mesh.blocks[block_id];
            const patch = mesh.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const cell_space = NodeSpace(N, O).fromSize(blockCellSize(mesh, block_id));

            const block_sys = sys.slice(
                dof_map.offset(block_id),
                dof_map.total(block_id),
            );

            var tiles = IndexSpace.fromBox(block.bounds).cartesianIndices();

            while (tiles.next()) |tile| {
                // Find underlying block
                const coarse_block_id = underlyingBlock(mesh, block_id, tile);

                const coarse_block = mesh.blocks[coarse_block_id];

                const coarse_cell_space = NodeSpace(N, O).fromSize(blockCellSize(mesh, coarse_block_id));

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
                            coarse_cell_space.prolong(PO, subcell, coarse_sys.field(field)),
                        );
                    }
                }
            }
        }

        /// Prolongs a correction to the given block. Sys and diff must be dof vectors filled on l - 1. Dest is a cell vector.
        pub fn prolongCorrection(
            comptime System: type,
            mesh: *const Mesh,
            dof_map: Map,
            block_id: usize,
            sys: SystemSlice(System),
            sys_old: SystemSliceConst(System),
        ) void {
            if (comptime !system.isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            sys.assertLen(dof_map.ndofs());
            sys_old.assertLen(dof_map.ndofs());

            const block = mesh.blocks[block_id];
            const patch = mesh.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const block_sys = sys.slice(
                dof_map.offset(block_id),
                dof_map.total(block_id),
            );

            const cell_space = NodeSpace(N, O).fromSize(blockCellSize(mesh, block_id));

            var tiles = IndexSpace.fromBox(block.bounds).cartesianIndices();

            while (tiles.next()) |tile| {
                const coarse_block_id = underlyingBlock(mesh, block_id, tile);

                const coarse_block = mesh.blocks[coarse_block_id];

                const coarse_cell_space = NodeSpace(N, O).fromSize(blockCellSize(mesh, coarse_block_id));

                const coarse_dof_offset = dof_map.offset(coarse_block_id);
                const coarse_dof_total = dof_map.total(coarse_block_id);

                const coarse_sys = sys.slice(coarse_dof_offset, coarse_dof_total);
                const coarse_old_sys = sys_old.slice(coarse_dof_offset, coarse_dof_total);

                const coarse_tile = coarse_block.bounds.refined().localFromGlobal(block.bounds.globalFromLocal(tile));
                const coarse_origin = Index.scaled(coarse_tile, mesh.tile_width);

                const origin = Index.scaled(tile, mesh.tile_width);

                var cells = IndexSpace.fromSize(Index.splat(mesh.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const global_cell = Index.toSigned(Index.add(origin, cell));
                    const sub_cell = Index.toSigned(Index.add(coarse_origin, cell));

                    inline for (comptime std.enums.values(System)) |field| {
                        const u = coarse_cell_space.prolong(PO, sub_cell, coarse_sys.field(field));
                        const v = coarse_cell_space.prolong(PO, sub_cell, coarse_old_sys.field(field));

                        const sys_val = cell_space.value(global_cell, block_sys.field(field));
                        cell_space.setValue(global_cell, block_sys.field(field), sys_val + u - v);
                    }
                }
            }
        }

        // *************************
        // Residual ****************
        // *************************

        pub fn residual(
            mesh: *const Mesh,
            dof_map: Map,
            block_id: usize,
            oper: anytype,
            res: SystemSlice(@TypeOf(oper).System),
            rhs: SystemSliceConst(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            const T = @TypeOf(oper);

            res.assertLen(mesh.cell_total);
            rhs.assertLen(mesh.cell_total);
            src.assertLen(dof_map.ndofs());
            ctx.assertLen(dof_map.ndofs());

            const block = mesh.blocks[block_id];

            const stencil_space = blockStencilSpace(mesh, block_id);

            const dof_offset = dof_map.offset(block_id);
            const dof_total = dof_map.total(block_id);

            const block_res = res.slice(block.cell_offset, block.cell_total);
            const block_rhs = rhs.slice(block.cell_offset, block.cell_total);
            const block_src = src.slice(dof_offset, dof_total);
            const block_ctx = ctx.slice(dof_offset, dof_total);

            var nodes = stencil_space.nodeSpace().nodes(0);
            var linear: usize = 0;

            while (nodes.next()) |node| : (linear += 1) {
                const engine = Engine(N, O, .normal, T.System, T.Context).new(
                    stencil_space,
                    node,
                    block_src,
                    block_ctx,
                );

                const app = oper.apply(.normal, engine);

                inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                    const a_val = @field(app, @tagName(field));
                    block_res.field(field)[linear] = block_rhs.field(field)[linear] - a_val;
                }
            }
        }

        // *************************
        // Apply *******************
        // *************************

        /// Applies an operator to a source dof vector and context, storing the result in the given dest cell vector.
        pub fn apply(
            mesh: *const Mesh,
            dof_map: Map,
            block_id: usize,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            const T = @TypeOf(oper);

            dest.assertLen(mesh.cell_total);
            src.assertLen(dof_map.ndofs());
            ctx.assertLen(dof_map.ndofs());

            const block = mesh.blocks[block_id];

            const stencil_space = blockStencilSpace(mesh, block_id);
            const dof_offset = dof_map.offset(block_id);
            const dof_total = dof_map.total(block_id);
            const block_dest = dest.slice(block.cell_offset, block.cell_total);
            const block_src = src.slice(dof_offset, dof_total);
            const block_ctx = ctx.slice(dof_offset, dof_total);

            var nodes = stencil_space.nodeSpace().nodes(0);
            var linear: usize = 0;

            while (nodes.next()) |node| : (linear += 1) {
                const engine = Engine(N, O, .normal, T.System, T.Context).new(
                    stencil_space,
                    node,
                    block_src,
                    block_ctx,
                );

                const app = oper.apply(.normal, engine);

                inline for (comptime std.enums.values(@TypeOf(oper).System)) |field| {
                    block_dest.field(field)[linear] = @field(app, @tagName(field));
                }
            }
        }

        // *************************
        // Smoothing ***************
        // *************************

        pub fn smooth(
            mesh: *const Mesh,
            dof_map: Map,
            block_id: usize,
            oper: anytype,
            dest: SystemSlice(@TypeOf(oper).System),
            rhs: SystemSliceConst(@TypeOf(oper).System),
            src: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            const T = @TypeOf(oper);

            dest.assertLen(mesh.cell_total);
            rhs.assertLen(mesh.cell_total);
            src.assertLen(dof_map.ndofs());
            ctx.assertLen(dof_map.ndofs());

            const block = mesh.blocks[block_id];

            const dof_offset = dof_map.offset(block_id);
            const dof_total = dof_map.total(block_id);

            const block_dest = dest.slice(block.cell_offset, block.cell_total);
            const block_rhs = rhs.slice(block.cell_offset, block.cell_total);
            const block_src = src.slice(dof_offset, dof_total);
            const block_ctx = ctx.slice(dof_offset, dof_total);

            const stencil_space = blockStencilSpace(mesh, block_id);
            const dof_space = stencil_space.nodeSpace();

            var nodes = dof_space.nodes(0);
            var linear: usize = 0;

            while (nodes.next()) |node| : (linear += 1) {
                const engine = Engine(N, O, .normal, T.System, T.Context).new(
                    stencil_space,
                    node,
                    block_src,
                    block_ctx,
                );

                const app = oper.apply(.normal, engine);

                const diag_engine = Engine(N, O, .diagonal, T.System, T.Context).new(
                    stencil_space,
                    node,
                    block_src,
                    block_ctx,
                );

                const diag = oper.apply(.diagonal, diag_engine);

                inline for (comptime std.enums.values(T.System)) |field| {
                    const a: f64 = @field(app, @tagName(field));
                    const d: f64 = @field(diag, @tagName(field));
                    const r = block_rhs.field(field)[linear];
                    const f = dof_space.value(node, block_src.field(field));

                    block_dest.field(field)[linear] = f + (r - a) / d;
                }
            }
        }

        // ******************************
        // Helpers **********************
        // ******************************

        /// Finds the coarse block underlying the given tile in the given refined block. Returns `maxInt(usize)` if
        /// no such coarse block exists.
        fn underlyingBlock(mesh: *const Mesh, block_id: usize, tile: [N]usize) usize {
            const block = mesh.blocks[block_id];
            const patch = mesh.patches[block.patch];
            const coarse_patch = mesh.patches[
                patch.parent orelse {
                    return std.math.maxInt(usize);
                }
            ];

            const coarse_tile_space = IndexSpace.fromBox(coarse_patch.bounds);
            const coarse_tile = Index.coarsened(coarse_patch.bounds.refined().localFromGlobal(block.bounds.globalFromLocal(tile)));

            const linear = coarse_tile_space.linearFromCartesian(coarse_tile);
            return mesh.block_map[coarse_patch.tile_offset + linear];
        }
    };
}

/// A namespace for routines which run over the whole mesh.
pub fn DofUtilsTotal(comptime N: usize, comptime O: usize) type {
    return struct {
        const CellSpace = basis.NodeSpace(N, O);
        const StencilSpace = basis.StencilSpace(N, O);
        const Index = geometry.Index(N);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexBox = geometry.Box(N, usize);
        const Mesh = meshes.Mesh(N);
        const Region = geometry.Region(N);
        const Map = DofMap(N, O);
        const Utils = DofUtils(N, O);

        // *************************
        // Projection **************
        // *************************

        /// Sets the values of a global cell solution vector using a projection function.
        pub fn project(
            mesh: *const Mesh,
            dof_map: Map,
            projection: anytype,
            dest: SystemSlice(@TypeOf(projection).System),
            ctx: SystemSliceConst(@TypeOf(projection).Context),
        ) void {
            const T = @TypeOf(projection);

            if (comptime !isSystemProjection(N, O)(T)) {
                @compileError("project() expects projection to satisfy isMeshProjection.");
            }

            dest.assertLen(mesh.cell_total);
            ctx.assertLen(dof_map.ndofs());

            for (0..mesh.blocks.len) |block_id| {
                const block = mesh.blocks[block_id];

                const block_dest = dest.slice(block.cell_offset, block.cell_total);
                const block_ctx = ctx.slice(dof_map.offset(block_id), dof_map.total(block_id));

                const stencil_space = Utils.blockStencilSpace(mesh, block_id);

                var cells = stencil_space.nodeSpace().nodes(0);
                var linear: usize = 0;

                while (cells.next()) |cell| : (linear += 1) {
                    const engine = ProjectionEngine(N, O, T.Context).new(
                        stencil_space,
                        cell,
                        EmptySystem.sliceConst(),
                        block_ctx,
                    );

                    const value: SystemValue(T.System) = projection.project(engine);

                    inline for (comptime std.enums.values(T.System)) |field| {
                        block_dest.field(field)[linear] = @field(value, @tagName(field));
                    }
                }
            }
        }
    };
}

pub fn ProjectionEngine(comptime N: usize, comptime O: usize, comptime Context: type) type {
    return Engine(N, O, .normal, system.EmptySystem, Context);
}

/// A trait which checks if a type is an operator. Such a type follows the following set of declarations.
/// ```
/// const Operator = struct {
///     pub const Context = enum {
///         field1,
///         field2,
///         // ...
///     };
///
///     pub const System = enum {
///         result,
///     };
///
///     pub fn apply(self: Operator, engine: OperatorEngine(2, 2, Context, System)) SystemValue(System) {
///         // ...
///     }
///
///     pub fn applyDiagonal(self: Operator, engine: OperatorEngine(2, 2, Context, System)) SystemValue(System) {
///         // ...
///     }
/// };
/// ```
pub fn isSystemOperator(comptime N: usize, comptime O: usize) fn (type) bool {
    _ = O;
    const hasFn = std.meta.trait.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (comptime !(@hasDecl(T, "System") and @TypeOf(T.System) == type and system.isSystem(T.System))) {
                return false;
            }

            if (comptime !(@hasDecl(T, "Context") and @TypeOf(T.Context) == type and system.isSystem(T.Context))) {
                return false;
            }

            if (comptime !(hasFn("apply")(T))) {
                return false;
            }

            if (comptime !(hasFn("boundarySys")(T) and @TypeOf(T.boundarySys) == fn (T, [N]f64, Face(N)) SystemBoundaryCondition(T.System))) {
                return false;
            }

            if (comptime !(hasFn("boundaryCtx")(T) and @TypeOf(T.boundaryCtx) == fn (T, [N]f64, Face(N)) SystemBoundaryCondition(T.Context))) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}

pub fn hasSystemOperatorCallback(comptime N: usize, comptime O: usize) bool {
    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (!isSystemOperator(N, O)(T)) {
                return false;
            }

            const hasFn = std.meta.trait.hasFn;

            if (!(hasFn("callback")(T) and @TypeOf(T.callback) == fn (*const T, usize, f64, SystemSliceConst(T.System)) void)) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}

pub fn isSystemProjection(comptime N: usize, comptime O: usize) fn (type) bool {
    const hasFn = std.meta.trait.hasFn;

    const Closure = struct {
        fn trait(comptime T: type) bool {
            if (comptime !(@hasDecl(T, "System") and @TypeOf(T.System) == type and system.isSystem(T.System))) {
                return false;
            }

            if (comptime !(@hasDecl(T, "Context") and @TypeOf(T.Context) == type and system.isSystem(T.Context))) {
                return false;
            }

            if (comptime !(hasFn("project")(T) and @TypeOf(T.project) == fn (T, ProjectionEngine(N, O, T.Context)) SystemValue(T.System))) {
                return false;
            }

            return true;
        }
    };

    return Closure.trait;
}

test {
    _ = boundaries;
}
