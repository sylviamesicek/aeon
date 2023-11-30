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
const nodes = @import("../nodes/nodes.zig");

const CellMap = mesh.CellMap;
const TileMap = mesh.TileMap;

const BoundaryKind = nodes.BoundaryKind;
const NodeMap = nodes.NodeMap;
const Robin = nodes.Robin;

// Other submodules

const mesh_ = @import("mesh.zig");

/// A convience structure that caches a number of offset maps, cell vectors and tile vectors,
/// with functions for filling boundary conditions, restricting, and prolonging.
pub fn DofManager(comptime N: usize, comptime M: usize) type {
    return struct {
        gpa: Allocator,
        cell_map: CellMap,
        tile_map: TileMap,
        node_map: NodeMap,
        tile_to_block: ArrayListUnmanaged(usize),

        const Self = @This();

        const FaceIndex = geometry.FaceIndex(M);
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

        const Mesh = mesh_.Mesh(N);

        const NodeSpace = nodes.NodeSpace(N, M);
        const NodeSpaceZeroth = nodes.NodeSpace(N, 0);
        const isBoundary = nodes.isBoundary(N);

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

        pub fn build(self: *Self, grid: Mesh) !void {
            try grid.buildCellMap(&self.cell_map);
            try grid.buildTileMap(&self.tile_map);
            try grid.buildNodeMap(M, &self.node_map);

            try self.tile_to_block.resize(self.tile_map.numTiles());

            @memset(self.tile_to_block.items, maxInt(usize));

            for (0..grid.patches.len) |patch_id| {
                const tile_to_block: []usize = self.tile_map.slice(patch_id, self.tile_to_block.items);
                const patch = grid.patches[patch_id];

                for (patch.block_offset..patch.block_total + patch.block_offset) |block_id| {
                    const block = self.blocks[block_id];
                    IndexSpace.fromBox(patch.bounds).fillWindow(block.bounds.relativeTo(patch.bounds), usize, tile_to_block, block_id);
                }
            }
        }

        pub fn tileToBlock(self: Self, tile: usize) usize {
            return self.tile_to_block.items[tile];
        }

        pub fn tileToBlockOrNull(self: Self, tile: usize) ?usize {
            const res = self.tile_to_block.items[tile];
            if (res == maxInt(usize)) {
                return null;
            } else {
                return res;
            }
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

                pub fn kind(axis: usize) BoundaryKind {
                    return T.kind(axis);
                }

                pub fn robin(self: @This(), pos: [N]f64, face: FaceIndex) Robin {
                    const global = self.bounds.transformPos(pos);
                    var res: Robin = self.inner.robin(global, face);
                    res.flux /= self.bounds.size[face.axis];
                    return res;
                }
            };
        }

        /// Fills the boundary values of a node vector.
        pub fn fillBoundary(
            self: Self,
            grid: *const Mesh,
            block: usize,
            boundary: anytype,
            field: []f64,
        ) void {
            assert(self.numNodes() == field.len);

            const level = grid.blockLevel(block);
            const bounds = grid.blocks[block].bounds;
            const tile_size = grid.levels[level].tile_size;

            const regions = comptime Region.orderedRegions();

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
                        .bounds = grid.blockPhysicalBounds(block),
                    };

                    const node_space = NodeSpace.fromCellSize(bounds.size);
                    node_space.fillBoundaryRegion(region, wrapped, self.node_map.slice(block, field));
                } else {
                    self.fillInteriorBoundary(region, grid, block, field);
                }
            }
        }

        fn fillInteriorBoundary(
            self: Self,
            comptime region: Region,
            grid: *const Mesh,
            block_id: usize,
            field: []f64,
        ) void {
            const extent_dir: [N]isize = region.extentDir();

            const block = grid.blocks[block_id];
            const block_node_space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));
            const block_field: []usize = self.node_map.slice(block_id, field);

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
                    const coarse_block_field: []const usize = self.node_map.slice(coarse_block_id, field);

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
                    const neighbor_field: []const usize = self.node_map.slice(neighbor_id, field);

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
        // Restrict / Prolong ******
        // *************************

        /// Given a node vector with correct boundary node at the given block, restrict the data to all underlying dofs.
        pub fn restrict(
            self: Self,
            grid: *const Mesh,
            block_id: usize,
            field: []f64,
        ) void {
            assert(self.node_map.numNodes() == field.len);

            const block_field: []usize = self.node_map.slice(block_id, field);

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

        /// Given a global node vector with correct boundary nodes on the lower level, prolong the data from all underlying dofs.
        pub fn prolong(
            self: Self,
            grid: *const Mesh,
            block_id: usize,
            field: []f64,
        ) void {
            assert(self.node_map.numNodes() == field.len);

            const block_field: []usize = self.node_map.slice(block_id, field);

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

        /// Given a cell vector restrict the data to all underlying cells using linear interpolation.
        pub fn restrictCells(
            self: Self,
            grid: *const Mesh,
            block_id: usize,
            field: []f64,
        ) void {
            assert(self.cell_map.numCells() == field.len);

            const block_field: []usize = self.cell_map.slice(block_id, field);

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
        // Helpers **********************
        // ******************************

        /// Finds the coarse block underlying the given tile in the given refined block. Returns `maxInt(usize)` if
        /// no such coarse block exists.
        pub fn underlyingBlock(self: Self, grid: *const Mesh, block_id: usize, tile: [N]usize) usize {
            const block = grid.blocks[block_id];
            const patch = grid.patches[block.patch];
            const coarse_patch = mesh.grid[
                patch.parent orelse {
                    return std.math.maxInt(usize);
                }
            ];

            const coarse_tile_space = IndexSpace.fromBox(coarse_patch.bounds);
            const coarse_tile = coarsened(coarse_patch.bounds.refined().localFromGlobal(block.bounds.globalFromLocal(tile)));

            const linear = coarse_tile_space.linearFromCartesian(coarse_tile);
            return self.block_map[coarse_patch.tile_offset + linear];
        }

        // *******************************
        // Aliases ***********************
        // *******************************

        pub fn numCells(self: Self) usize {
            return self.cell_map.numCells();
        }

        pub fn numNodes(self: Self) usize {
            return self.node_map.numNodes();
        }

        pub fn numTiles(self: Self) usize {
            return self.tile_map.numTiles();
        }
    };
}
