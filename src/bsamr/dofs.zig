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
const mesh = @import("../mesh/mesh.zig");
const common = @import("../common/common.zig");

const CellMap = mesh.CellMap;
const TileMap = mesh.TileMap;

const BoundaryKind = common.BoundaryKind;
const NodeMap = mesh.CellMap;
const Robin = common.Robin;

// Other submodules

const mesh_ = @import("mesh.zig");

/// A convience structure that caches a number of offset maps, cell vectors and tile vectors,
/// with functions for filling boundary conditions, interpolating values, and applying operators.
pub fn DofManager(comptime N: usize, comptime M: usize) type {
    return struct {
        gpa: Allocator,
        cell_map: CellMap,
        tile_map: TileMap,
        node_map: NodeMap,
        tile_to_block: ArrayListUnmanaged(usize),

        const Self = @This();

        const FaceIndex = geometry.FaceIndex(N);
        const IndexBox = geometry.IndexBox(N);
        const IndexMixin = geometry.IndexMixin(N);
        const IndexSpace = geometry.IndexSpace(N);
        const RealBox = geometry.RealBox(N);
        const Region = geometry.Region(N);

        const add = IndexMixin.add;
        const addWithSign = IndexMixin.addWithSign;
        const coarsened = IndexMixin.coarsened;
        const offsetFromOrigin = IndexMixin.offsetFromOrigin;
        const scaled = IndexMixin.scaled;
        const splat = IndexMixin.splat;
        const toUnsigned = IndexMixin.toUnsigned;
        const toSigned = IndexMixin.toSigned;

        const Engine = mesh.Engine(N, M);

        const Mesh = mesh_.Mesh(N);

        const NodeSpace = common.NodeSpace(N, M);
        const NodeSpaceZeroth = common.NodeSpace(N, 0);
        const isBoundary = common.isBoundary(N);

        pub fn init(allocator: Allocator) Self {
            return .{
                .gpa = allocator,
                .cell_map = CellMap.init(allocator),
                .tile_map = TileMap.init(allocator),
                .node_map = NodeMap.init(allocator),
                .tile_to_block = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            self.cell_map.deinit();
            self.tile_map.deinit();
            self.node_map.deinit();
            self.tile_to_block.deinit(self.gpa);
        }

        pub fn build(self: *Self, grid: *const Mesh) !void {
            try grid.buildCellMap(&self.cell_map);
            try grid.buildTileMap(&self.tile_map);
            try grid.buildNodeMap(M, &self.node_map);

            try self.tile_to_block.resize(self.gpa, self.tile_map.numTiles());

            @memset(self.tile_to_block.items, maxInt(usize));

            for (0..grid.patches.len) |patch_id| {
                const tile_to_block: []usize = self.tile_map.slice(patch_id, self.tile_to_block.items);
                const patch = grid.patches[patch_id];

                for (patch.block_offset..patch.block_total + patch.block_offset) |block_id| {
                    const block = grid.blocks[block_id];
                    IndexSpace.fromBox(patch.bounds).fillWindow(block.bounds.relativeTo(patch.bounds), usize, tile_to_block, block_id);
                }
            }
        }

        pub fn tileToBlock(self: *const Self, tile: usize) usize {
            return self.tile_to_block.items[tile];
        }

        pub fn tileToBlockOrNull(self: *const Self, tile: usize) ?usize {
            const res = self.tile_to_block.items[tile];
            if (res == maxInt(usize)) {
                return null;
            } else {
                return res;
            }
        }

        // *******************************
        // Transfer **********************
        // *******************************

        pub fn transfer(
            self: *const Self,
            grid: *const Mesh,
            boundary: anytype,
            dest: []f64,
            src: []const f64,
        ) void {
            assert(dest.len == self.numNodes());
            assert(src.len == self.numCells());

            self.copyNodesFromCells(grid, dest, src);
            self.fillBoundary(grid, boundary, dest);
        }

        // *******************************
        // Boundaries ********************
        // *******************************

        fn BoundaryWrapper(comptime T: type) type {
            if (comptime !isBoundary(T)) {
                @compileError("T must satisfy isBoundary trait.");
            }
            return struct {
                inner: T,
                bounds: RealBox,

                pub fn kind(face: FaceIndex) BoundaryKind {
                    return T.kind(face);
                }

                pub fn robin(self: @This(), pos: [N]f64, face: FaceIndex) Robin {
                    const global = self.bounds.transformPos(pos);
                    var res: Robin = self.inner.robin(global, face);
                    res.flux /= self.bounds.size[face.axis];
                    return res;
                }
            };
        }

        pub fn fillBoundary(
            self: *const Self,
            grid: *const Mesh,
            boundary: anytype,
            dest: []f64,
        ) void {
            for (0..grid.blocks.len) |block_id| {
                self.fillBlockBoundary(grid, block_id, boundary, dest);
            }
        }

        pub fn fillLevelBoundary(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            boundary: anytype,
            dest: []f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.fillBlockBoundary(grid, block_id, boundary, dest);
            }
        }

        /// Fills the boundary values of a node vector.
        pub fn fillBlockBoundary(
            self: *const Self,
            grid: *const Mesh,
            block_id: usize,
            boundary: anytype,
            dest: []f64,
        ) void {
            assert(self.numNodes() == dest.len);

            const level = grid.blockLevel(block_id);
            const bounds = grid.blocks[block_id].bounds;
            const tile_size: [N]usize = grid.levels[level].tile_size;

            const regions = comptime Region.enumerateOrdered();

            inline for (comptime regions[1..]) |region| {
                var exterior: bool = true;

                for (0..N) |i| {
                    if (region.sides[i] == .left and bounds.origin[i] != 0) {
                        exterior = false;
                    } else if (region.sides[i] == .right and bounds.origin[i] + bounds.size[i] != tile_size[i]) {
                        exterior = false;
                    }
                }

                if (exterior) {
                    const wrapped = BoundaryWrapper(@TypeOf(boundary)){
                        .inner = boundary,
                        .bounds = grid.blockPhysicalBounds(block_id),
                    };
                    _ = wrapped; // autofix

                    const space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));
                    _ = space; // autofix
                    // BoundaryUtils.fillBoundaryRegion(region, space, wrapped, self.node_map.slice(block_id, dest));
                } else {
                    self.fillBlockIntBoundary(region, grid, block_id, dest);
                }
            }
        }

        fn fillBlockIntBoundary(
            self: *const Self,
            comptime region: Region,
            grid: *const Mesh,
            block_id: usize,
            field: []f64,
        ) void {
            const extent_dir: [N]isize = comptime region.extentDir();

            const block = grid.blocks[block_id];
            const block_node_space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));
            const block_field: []f64 = self.node_map.slice(block_id, field);

            const patch = grid.patches[block.patch];
            const patch_space = IndexSpace.fromBox(patch.bounds);
            const patch_block_map: []const usize = self.tile_map.slice(block.patch, self.tile_to_block.items);

            const relative_bounds = block.bounds.relativeTo(patch.bounds);

            const buffer_exists: bool = blk: {
                var res: bool = true;
                inline for (0..N) |i| {
                    if (comptime extent_dir[i] == -1) {
                        res = res and (relative_bounds.origin[i] > 0);
                    } else if (comptime extent_dir[i] == 1) {
                        res = res and (relative_bounds.origin[i] + relative_bounds.size[i] < patch.bounds.size[i]);
                    }
                }
                break :blk res;
            };

            var tiles = region.innerFaceCells(relative_bounds.size);

            while (tiles.next()) |t| {
                // Tile in patch space
                const tile: [N]usize = relative_bounds.globalFromLocal(t);
                // Origin in block cell space
                const origin: [N]usize = scaled(t, grid.tile_width);

                const neighbor_id: usize = blk: {
                    if (!buffer_exists) {
                        break :blk maxInt(usize);
                    }

                    const buffer_tile: [N]usize = addWithSign(tile, extent_dir);

                    break :blk patch_block_map[patch_space.linearFromCartesian(buffer_tile)];
                };

                if (neighbor_id == maxInt(usize)) {
                    // Prolong from underlying block
                    const coarse_patch_id = patch.parent.?;
                    const coarse_patch = grid.patches[coarse_patch_id];
                    const coarse_patch_block_map: []const usize = self.tile_map.slice(coarse_patch_id, self.tile_to_block.items);
                    const coarse_patch_space = IndexSpace.fromBox(coarse_patch.bounds);

                    // Tile in coarse subtile patch space
                    const coarse_tile = coarse_patch.bounds.refined().localFromGlobal(patch.bounds.globalFromLocal(tile));

                    const coarse_block_id = coarse_patch_block_map[coarse_patch_space.linearFromCartesian(coarsened(coarse_tile))];
                    const coarse_block_node_space = NodeSpace.fromCellSize(grid.blockCellSize(coarse_block_id));
                    const coarse_block_field: []const f64 = self.node_map.slice(coarse_block_id, field);

                    // Cell origin in coarse subcell patch space
                    const coarse_neighbor_origin: [N]usize = scaled(coarse_tile, grid.tile_width);

                    var indices = region.nodes(M, splat(grid.tile_width));

                    while (indices.next()) |idx| {
                        const block_node: [N]isize = offsetFromOrigin(origin, idx);
                        const neighbor_node: [N]isize = offsetFromOrigin(coarse_neighbor_origin, idx);

                        const v = coarse_block_node_space.prolong(toUnsigned(neighbor_node), coarse_block_field);
                        block_node_space.setNodeValue(block_node, block_field, v);
                    }
                } else {
                    // Copy from neighboring block on same patch
                    const neighbor = grid.blocks[neighbor_id];
                    const neighbor_relative_bounds = neighbor.bounds.relativeTo(patch.bounds);
                    const neighbor_t = neighbor_relative_bounds.localFromGlobal(tile);
                    const neighbor_node_space = NodeSpace.fromCellSize(grid.blockCellSize(neighbor_id));
                    const neighbor_origin = scaled(neighbor_t, grid.tile_width);
                    const neighbor_field: []const f64 = self.node_map.slice(neighbor_id, field);

                    var indices = region.nodes(M, splat(grid.tile_width));

                    while (indices.next()) |idx| {
                        const block_node: [N]isize = offsetFromOrigin(origin, idx);
                        const neighbor_node: [N]isize = offsetFromOrigin(neighbor_origin, idx);

                        const v = neighbor_node_space.nodeValue(neighbor_node, neighbor_field);
                        block_node_space.setNodeValue(block_node, block_field, v);
                    }
                }
            }
        }

        // *************************
        // Copy ********************
        // *************************

        pub fn copyNodesFromCells(
            self: *const Self,
            grid: *const Mesh,
            dest: []f64,
            src: []const f64,
        ) void {
            for (0..grid.blocks.len) |block_id| {
                self.copyBlockNodesFromCells(grid, block_id, dest, src);
            }
        }

        pub fn copyLevelNodesFromCells(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            dest: []f64,
            src: []const f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.copyBlockNodesFromCells(grid, block_id, dest, src);
            }
        }

        // Fills interior nodes of the given block using cell data.
        pub fn copyBlockNodesFromCells(
            self: *const Self,
            grid: *const Mesh,
            block_id: usize,
            dest: []f64,
            src: []const f64,
        ) void {
            assert(dest.len == self.numNodes());
            assert(src.len == self.numCells());

            const block_dest: []f64 = self.node_map.slice(block_id, dest);
            const block_src: []const f64 = self.cell_map.slice(block_id, src);

            const node_space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));

            var cells = node_space.cellSpace().cartesianIndices();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                node_space.setValue(cell, block_dest, block_src[linear]);
            }
        }

        pub fn copyCellsFromNodes(
            self: *const Self,
            grid: *const Mesh,
            dest: []f64,
            src: []const f64,
        ) void {
            for (0..grid.blocks.len) |block_id| {
                self.copyBlockCellsFromNodes(grid, block_id, dest, src);
            }
        }

        pub fn copyLevelCellsFromNodes(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            dest: []f64,
            src: []const f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.copyBlockCellsFromNodes(grid, block_id, dest, src);
            }
        }

        pub fn copyBlockCellsFromNodes(
            self: *const Self,
            grid: *const Mesh,
            block_id: usize,
            dest: []f64,
            src: []const f64,
        ) void {
            assert(dest.len == self.numCells());
            assert(src.len == self.numNodes());

            const block_dest: []f64 = self.cell_map.slice(block_id, dest);
            const block_src: []const f64 = self.node_map.slice(block_id, src);

            const node_space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));

            var cells = node_space.cellSpace().cartesianIndices();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                block_dest[linear] = node_space.value(cell, block_src);
            }
        }

        pub fn copyLevelCells(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            dest: []f64,
            src: []const f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.copyBlockCells(block_id, dest, src);
            }
        }

        pub fn copyBlockCells(
            self: *const Self,
            block_id: usize,
            dest: []f64,
            src: []const f64,
        ) void {
            assert(dest.len == self.numCells());
            assert(src.len == self.numCells());

            const block_dest: []f64 = self.cell_map.slice(block_id, dest);
            const block_src: []const f64 = self.cell_map.slice(block_id, src);

            @memcpy(block_dest, block_src);
        }

        pub fn copyLevelNodes(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            dest: []f64,
            src: []const f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.copyBlockNodes(block_id, dest, src);
            }
        }

        pub fn copyBlockNodes(
            self: *const Self,
            block_id: usize,
            dest: []f64,
            src: []const f64,
        ) void {
            assert(dest.len == self.numNodes());
            assert(src.len == self.numNodes());

            const block_dest: []f64 = self.node_map.slice(block_id, dest);
            const block_src: []const f64 = self.node_map.slice(block_id, src);

            @memcpy(block_dest, block_src);
        }

        // pub fn swapBlockNodes(
        //     self: *const Self,
        //     block_id: usize,
        //     a: []f64,
        //     b: []f64,
        // ) void {
        //     assert(a.len == self.numNodes());
        //     assert(b.len == self.numNodes());

        //     const block_a: []f64 = self.node_map.slice(block_id, a);
        //     const block_b: []f64 = self.node_map.slice(block_id, b);

        //     for (0..block_a.len) |idx| {
        //         const tmp = block_a[idx];
        //         block_a[idx] = block_b[idx];
        //         block_b[idx] = tmp;
        //     }
        // }

        // *************************
        // Restrict / Prolong ******
        // *************************

        pub fn restrictLevel(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            dest: []f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.restrictBlock(grid, block_id, dest);
            }
        }

        /// Given a node vector with correct boundary node at the given block, restrict the data to all underlying dofs.
        pub fn restrictBlock(
            self: *const Self,
            grid: *const Mesh,
            block_id: usize,
            field: []f64,
        ) void {
            assert(self.node_map.numNodes() == field.len);

            const block_field: []f64 = self.node_map.slice(block_id, field);

            const block = grid.blocks[block_id];
            const patch = grid.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const block_bounds = block.bounds.coarsened();
            const node_space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));

            var tiles = IndexSpace.fromBox(block_bounds).cartesianIndices();

            while (tiles.next()) |t| {
                const coarse_block_id = self.underlyingBlock(grid, block_id, t);
                const coarse_block = grid.blocks[coarse_block_id];
                const coarse_node_space = NodeSpace.fromCellSize(grid.blockCellSize(coarse_block_id));
                const coarse_block_field: []f64 = self.node_map.slice(coarse_block_id, field);

                const tile = coarse_block.bounds.localFromGlobal(block_bounds.globalFromLocal(t));

                const origin = scaled(t, grid.tile_width);
                const coarse_origin = scaled(tile, grid.tile_width);

                var cells = IndexSpace.fromSize(splat(grid.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const super_cell = add(origin, cell);
                    const coarse_cell = add(coarse_origin, cell);

                    const v = node_space.restrict(super_cell, block_field);
                    coarse_node_space.setValue(coarse_cell, coarse_block_field, v);
                }
            }
        }

        pub fn prolongLevel(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            dest: []f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.prolongBlock(grid, block_id, dest);
            }
        }

        /// Given a global node vector with correct boundary nodes on the lower level, prolong the data from all underlying dofs.
        pub fn prolongBlock(
            self: *const Self,
            grid: *const Mesh,
            block_id: usize,
            field: []f64,
        ) void {
            assert(self.node_map.numNodes() == field.len);

            const block_field: []f64 = self.node_map.slice(block_id, field);

            const block = grid.blocks[block_id];
            const patch = grid.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const node_space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));

            var tiles = IndexSpace.fromBox(block.bounds).cartesianIndices();

            while (tiles.next()) |t| {
                const coarse_block_id = self.underlyingBlock(grid, block_id, t);
                const coarse_block = grid.blocks[coarse_block_id];
                const coarse_node_space = NodeSpace.fromCellSize(grid.blockCellSize(coarse_block_id));
                const coarse_block_field: []f64 = self.node_map.slice(coarse_block_id, field);

                const coarse_tile = coarse_block.bounds.refined().localFromGlobal(block.bounds.globalFromLocal(t));
                const coarse_origin = scaled(coarse_tile, grid.tile_width);

                const origin = scaled(t, grid.tile_width);

                var cells = IndexSpace.fromSize(splat(grid.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const target_cell = add(origin, cell);
                    const coarse_cell = add(coarse_origin, cell);

                    const v = coarse_node_space.prolong(coarse_cell, coarse_block_field);
                    node_space.setValue(target_cell, block_field, v);
                }
            }
        }

        pub fn restrictCells(
            self: *const Self,
            grid: *const Mesh,
            field: []f64,
        ) void {
            for (0..grid.blocks.len) |rev_block_id| {
                const block_id = grid.blocks.len - 1 - rev_block_id;
                self.restrictBlockCells(grid, block_id, field);
            }
        }

        pub fn restrictLevelCells(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            dest: []f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.restrictBlockCells(grid, block_id, dest);
            }
        }

        /// Given a cell vector restrict the data to all underlying cells using linear interpolation.
        pub fn restrictBlockCells(
            self: *const Self,
            grid: *const Mesh,
            block_id: usize,
            field: []f64,
        ) void {
            assert(self.cell_map.numCells() == field.len);

            const block_field: []f64 = self.cell_map.slice(block_id, field);

            const block = grid.blocks[block_id];
            const patch = grid.patches[block.patch];

            if (patch.parent == null) {
                return;
            }

            const block_bounds = block.bounds.coarsened();
            const node_space = NodeSpaceZeroth.fromCellSize(grid.blockCellSize(block_id));

            var tiles = IndexSpace.fromBox(block_bounds).cartesianIndices();

            while (tiles.next()) |t| {
                const coarse_block_id = self.underlyingBlock(grid, block_id, t);
                const coarse_block = grid.blocks[coarse_block_id];
                const coarse_node_space = NodeSpaceZeroth.fromCellSize(grid.blockCellSize(coarse_block_id));
                const coarse_block_field: []f64 = self.cell_map.slice(coarse_block_id, field);

                const tile = coarse_block.bounds.localFromGlobal(block_bounds.globalFromLocal(t));

                const origin = scaled(t, grid.tile_width);
                const coarse_origin = scaled(tile, grid.tile_width);

                var cells = IndexSpace.fromSize(splat(grid.tile_width)).cartesianIndices();

                while (cells.next()) |cell| {
                    const super_cell = add(origin, cell);
                    const coarse_cell = add(coarse_origin, cell);

                    const v = node_space.restrict(super_cell, block_field);
                    coarse_node_space.setValue(coarse_cell, coarse_block_field, v);
                }
            }
        }

        // ******************************
        // Operator *********************
        // ******************************

        pub fn project(
            self: *const Self,
            grid: *const Mesh,
            projection: anytype,
            dest: []f64,
        ) void {
            for (0..grid.blocks.len) |block_id| {
                self.projectBlock(grid, block_id, projection, dest);
            }
        }

        pub fn projectLevel(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            projection: anytype,
            dest: []f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.projectBlock(grid, block_id, projection, dest);
            }
        }

        pub fn projectBlock(
            self: *const Self,
            grid: *const Mesh,
            block_id: usize,
            projection: anytype,
            dest: []f64,
        ) void {
            const Proj = @TypeOf(projection);

            if (comptime !mesh.isProjection(N, M)(Proj)) {
                @compileError("Projection must satisfy isProjection trait.");
            }

            assert(dest.len == self.numCells());

            const block_dest: []f64 = self.cell_map.slice(block_id, dest);

            const start = self.node_map.offset(block_id);
            const end = self.node_map.offset(block_id + 1);

            const space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));

            const bounds: RealBox = grid.blockPhysicalBounds(block_id);

            var cells = space.cellSpace().cartesianIndices();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                const engine: Engine = .{
                    .space = space,
                    .bounds = bounds,
                    .cell = cell,
                    .start = start,
                    .end = end,
                };

                block_dest[linear] = projection.project(engine);
            }
        }

        pub fn apply(
            self: *const Self,
            grid: *const Mesh,
            operator: anytype,
            dest: []f64,
            src: []const f64,
        ) void {
            for (0..grid.blocks.len) |block_id| {
                self.applyBlock(grid, block_id, operator, dest, src);
            }
        }

        pub fn applyLevel(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            operator: anytype,
            dest: []f64,
            src: []const f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.applyBlock(grid, block_id, operator, dest, src);
            }
        }

        pub fn applyBlock(
            self: *const Self,
            grid: *const Mesh,
            block_id: usize,
            operator: anytype,
            dest: []f64,
            src: []const f64,
        ) void {
            const Op = @TypeOf(operator);

            if (comptime !mesh.isOperator(N, M)(Op)) {
                @compileError("Operator must satisfy isOperator trait.");
            }

            assert(dest.len == self.numCells());
            assert(src.len == self.numNodes());

            const block_dest: []f64 = self.cell_map.slice(block_id, dest);

            const start = self.node_map.offset(block_id);
            const end = self.node_map.offset(block_id + 1);

            const space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));

            const bounds: RealBox = grid.blockPhysicalBounds(block_id);

            var cells = space.cellSpace().cartesianIndices();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                const engine: Engine = .{
                    .space = space,
                    .bounds = bounds,
                    .cell = cell,
                    .start = start,
                    .end = end,
                };

                block_dest[linear] = operator.apply(engine, src);
            }
        }

        pub fn residual(
            self: *const Self,
            grid: *const Mesh,
            operator: anytype,
            dest: []f64,
            src: []const f64,
            rhs: []const f64,
        ) void {
            for (0..grid.blocks.len) |block_id| {
                self.residualBlock(grid, block_id, operator, dest, src, rhs);
            }
        }

        pub fn residualLevel(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            operator: anytype,
            dest: []f64,
            src: []const f64,
            rhs: []const f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.residualBlock(grid, block_id, operator, dest, src, rhs);
            }
        }

        pub fn residualBlock(
            self: *const Self,
            grid: *const Mesh,
            block_id: usize,
            operator: anytype,
            dest: []f64,
            src: []const f64,
            rhs: []const f64,
        ) void {
            const Op = @TypeOf(operator);

            if (comptime !mesh.isOperator(N, M)(Op)) {
                @compileError("Operator must satisfy isOperator trait.");
            }

            assert(dest.len == self.numCells());
            assert(src.len == self.numNodes());
            assert(rhs.len == self.numCells());

            const block_dest: []f64 = self.cell_map.slice(block_id, dest);
            const block_rhs: []const f64 = self.cell_map.slice(block_id, rhs);

            const space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));

            const start = self.node_map.offset(block_id);
            const end = self.node_map.offset(block_id + 1);

            const bounds: RealBox = grid.blockPhysicalBounds(block_id);

            var cells = space.cellSpace().cartesianIndices();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                const engine: Engine = .{
                    .space = space,
                    .bounds = bounds,
                    .cell = cell,
                    .start = start,
                    .end = end,
                };

                block_dest[linear] = block_rhs[linear] - operator.apply(engine, src);
            }
        }

        pub fn smooth(
            self: *const Self,
            grid: *const Mesh,
            operator: anytype,
            dest: []f64,
            src: []const f64,
            rhs: []const f64,
        ) void {
            for (0..grid.blocks.len) |block_id| {
                self.smoothBlock(grid, block_id, operator, dest, src, rhs);
            }
        }

        pub fn smoothLevel(
            self: *const Self,
            grid: *const Mesh,
            level_id: usize,
            operator: anytype,
            dest: []f64,
            src: []const f64,
            rhs: []const f64,
        ) void {
            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                self.smoothBlock(grid, block_id, operator, dest, src, rhs);
            }
        }

        pub fn smoothBlock(
            self: *const Self,
            grid: *const Mesh,
            block_id: usize,
            operator: anytype,
            dest: []f64,
            src: []const f64,
            rhs: []const f64,
        ) void {
            assert(dest.len == self.numCells());
            assert(src.len == self.numNodes());
            assert(rhs.len == self.numCells());

            const block_dest: []f64 = self.cell_map.slice(block_id, dest);
            const block_src: []const f64 = self.node_map.slice(block_id, src);
            const block_rhs: []const f64 = self.cell_map.slice(block_id, rhs);

            const space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));

            const start = self.node_map.offset(block_id);
            const end = self.node_map.offset(block_id + 1);

            const bounds: RealBox = grid.blockPhysicalBounds(block_id);

            var cells = space.cellSpace().cartesianIndices();
            var linear: usize = 0;

            while (cells.next()) |cell| : (linear += 1) {
                const engine: Engine = .{
                    .space = space,
                    .bounds = bounds,
                    .cell = cell,
                    .start = start,
                    .end = end,
                };

                const val = space.value(cell, block_src);
                const app = operator.apply(engine, src);
                const appDiag = operator.applyDiag(engine);

                block_dest[linear] = val + 2.0 / 3.0 * (block_rhs[linear] - app) / appDiag;
            }
        }

        // ******************************
        // Helpers **********************
        // ******************************

        /// Finds the coarse block underlying the given tile in the given refined block. Returns `maxInt(usize)` if
        /// no such coarse block exists.
        pub fn underlyingBlock(self: *const Self, grid: *const Mesh, block_id: usize, tile: [N]usize) usize {
            const block = grid.blocks[block_id];
            const patch = grid.patches[block.patch];
            const coarse_patch_id = patch.parent orelse {
                return std.math.maxInt(usize);
            };
            const coarse_patch = grid.patches[coarse_patch_id];

            const coarse_tile_space = IndexSpace.fromBox(coarse_patch.bounds);
            const coarse_tile = coarsened(coarse_patch.bounds.refined().localFromGlobal(block.bounds.globalFromLocal(tile)));

            const linear = coarse_tile_space.linearFromCartesian(coarse_tile);
            const tile_offset = self.tile_map.offset(coarse_patch_id);
            return self.tile_to_block.items[tile_offset + linear];
        }

        // *******************************
        // Aliases ***********************
        // *******************************

        pub fn numCells(self: *const Self) usize {
            return self.cell_map.numCells();
        }

        pub fn numNodes(self: *const Self) usize {
            return self.node_map.numNodes();
        }

        pub fn numTiles(self: *const Self) usize {
            return self.tile_map.numTiles();
        }
    };
}
