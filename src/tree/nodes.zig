const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;
const panic = std.debug.panic;

const tree = @import("tree.zig");
const null_index = tree.null_index;
const boundary_index = tree.boundary_index;

const permute = @import("permute.zig");

const common = @import("../common/common.zig");
const geometry = @import("../geometry/geometry.zig");
const utils = @import("../utils.zig");

const Range = utils.Range;
const RangeMap = utils.RangeMap;

/// Stores the additional structured needed to associate nodes with a mesh. Uniform children with common parents
/// are grouped into blocks to prevent unnessary ghost node duplication. Similar to a mesh, this node manager
/// describes the structure of a discretization.
/// All routines for transfering between node vectors, prolongation, smoothing, and filling ghost nodes are handled
/// by seperate `NodeWorker` classes which handle parallelism and dispatching work to other devices
/// (like the GPU or other processes).
pub fn NodeManager(comptime N: usize) type {
    return struct {
        gpa: Allocator,
        /// The set of blocks that make up the mesh.
        blocks: ArrayListUmanaged(Block),
        /// Additional info stored per mesh cell.
        cells: MultiArrayList(Cell),

        /// Map from blocks to cell ranges.
        block_to_cells: RangeMap,
        /// Map from blocks to node ranges
        block_to_nodes: RangeMap,
        /// Map from levels to block ranges.
        level_to_blocks: RangeMap,

        /// Number of nodes per axis of each cell.
        cell_size: [N]usize,
        /// A cell permutation structure,
        cell_permute: CellPermutation,

        const TreeMesh = tree.TreeMesh(N);
        const CellPermutation = permute.CellPermutation(N);

        const AxisMask = geometry.AxisMask(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const RealBox = geometry.RealBox(N);
        const Region = geometry.Region(N);
        const numSplits = AxisMask.count;

        // Sub types

        /// Represents blocks of uniform and contiguous cells which all share a common anscestor.
        pub const Block = struct {
            /// Physical bounds of block.
            bounds: RealBox = RealBox.unit,
            /// Number of cells along each axis of block.
            size: [N]usize = [1]usize{1} ** N,
            /// Boolean array indicating whether or not a given face is on the physical boundary.
            boundary: [FaceIndex.count]bool,
            /// Refinement level of block.
            refinement: usize,
        };

        /// Additional information stored for each cell of the mesh.
        pub const Cell = struct {
            /// Block that this cell belongs to.
            block: usize = 0,
            /// Position within block of this cell.
            index: [N]usize = [1]usize{0} ** N,
            /// Cache of neighboring cells.
            neighbors: [Region.count]usize,
        };

        /// Creates a new `NodeManager`, precaching information like cell_size and max_refinement.
        pub fn init(allocator: Allocator, cell_size: [N]usize, max_refinement: usize) !@This() {
            const cell_permute = try CellPermutation.init(allocator, max_refinement);
            errdefer cell_permute.deinit(allocator);

            for (0..N) |axis| {
                assert(cell_size[axis] % 2 == 0);
            }

            return .{
                .gpa = allocator,
                .blocks = .{},
                .cells = .{},

                .block_to_cells = .{},
                .block_to_nodes = .{},
                .level_to_blocks = .{},

                .cell_size = cell_size,
                .cell_permute = cell_permute,
            };
        }

        /// Frees a `NodeManager`.
        pub fn deinit(self: *@This()) void {
            self.blocks.deinit(self.gpa);
            self.cells.deinit(self.gpa);

            self.block_to_cells.deinit(self.gpa);
            self.block_to_nodes.deinit(self.gpa);
            self.level_to_blocks.deinit(self.gpa);

            self.cell_permute.deinit(self.gpa);
        }

        /// Builds a new `NodeManager` from a mesh, using the given allocator for scratch allocations.
        pub fn build(self: *@This(), allocator: Allocator, mesh: *const TreeMesh) !void {
            // ****************************
            // Computes Blocks + Offsets
            self.buildBlocks(allocator, mesh);

            // *****************************
            // Compute neighbors
            self.buildCells(mesh);

            // ********************************
            // Compute block to nodes map
            self.buildBlockToNodes();
        }

        /// Runs a recursive algorithm that traverses the tree from root to leaves, computing valid blocks while doing so.
        /// This algorithm works level by level (queueing up blocks for use on the next level). And thus preserves the invariant
        /// that all blocks on the same level are contiguous in the block array.
        fn buildBlocks(self: *@This(), allocator: Allocator, mesh: *const TreeMesh) void {
            // Cache
            const cells = mesh.cells.slice();
            const levels = mesh.numLevels();

            // Reset level offsets

            try self.level_to_blocks.resize(allocator, levels + 1);
            self.level_to_blocks.set(0, 0);
            self.level_to_blocks.set(1, 1);

            // Reset blocks
            self.blocks.clearRetainingCapacity();
            try self.blocks.append(self.gpa, .{ .refinement = 0 });

            self.block_to_cells.clear();
            try self.block_to_cells.append(self.gpa, 0);

            // Iterate levels
            const BlockInfo = struct {
                boundary: [FaceIndex.count]bool,
                refinement: usize,
                offset: usize,
                total: usize,
            };

            // Allocate scratch data
            var stack: ArrayListUmanaged(BlockInfo) = .{};
            defer stack.deinit(allocator);

            for (1..levels) |level| {
                const coarse = self.level_to_blocks.range(level - 1);
                // Transfer all blocks on l - 1 to stack (in reverse)
                for (0..coarse.end - coarse.start) |rev_idx| {
                    const idx = coarse.end - 1 - rev_idx;
                    const block = self.blocks.items[idx];
                    const offset = self.block_to_cells.offset(idx);
                    const total = cellTotalFromRefinement(block.refinement);

                    try stack.append(allocator, .{
                        .boundary = [1]bool{true} ** FaceIndex.count,
                        .refinement = block.refinement,
                        .offset = offset,
                        .total = total,
                    });
                }

                // Iterate until the stack is empty
                while (stack.popOrNull()) |block| {
                    var leaf_count: usize = 0;

                    for (block.offset..block.offset + block.total) |cell| {
                        if (cells.items(.children)[cell] == null_index) {
                            leaf_count += 1;
                        }
                    }

                    if (leaf_count == block.total) {
                        // Skip block if it is a leaf
                        continue;
                    } else if (leaf_count == 0 and block.refinement >= self.cell_permute.maxRefinement()) {
                        // Accept

                        // Kind of hacky, this depends on the fact that all grandchildren are stored contigiously.
                        // Document this invariant.

                        const grandchildren = cells.items(.children)[block.offset];

                        try self.blocks.append(self.gpa, .{
                            .boundary = block.boundary,
                            .refinement = block.refinement + 1,
                        });
                        try self.block_to_cells.append(self.gpa, grandchildren);
                    }

                    // Some are leaves, some are not. Perform subdivision.
                    const refinement_sub = block.refinement - 1;
                    const total_sub = cellTotalFromRefinement(refinement_sub);

                    var offset_sub = block.offset;

                    for (AxisMask.enumerate()) |split| {
                        var boundary = block.boundary;

                        for (split.innerFaces()) |face| {
                            boundary[face.toLinear()] = false;
                        }

                        try stack.append(allocator, .{
                            .boundary = boundary,
                            .refinement = refinement_sub,
                            .node_offset = offset_sub,
                            .node_total = total_sub,
                        });

                        offset_sub += total_sub;
                    }
                }

                self.level_to_blocks.set(level + 1, self.blocks.items.len);
            }

            // Append total number of cells
            try self.block_to_cells.append(self.gpa, cells.len);

            // Compute bounds and size
            for (self.blocks.items, 0..) |*block, block_id| {
                var base: usize = self.block_to_cells.offset(block_id);

                for (0..block.refinement) |_| {
                    base = cells.items(.parent)[base];
                }

                block.bounds = cells.items(.bounds)[base];
                block.size = cellSizeFromRefinement(block.refinement);
            }
        }

        /// Computes the block to node offset map.
        fn buildBlockToNodes(self: *@This()) void {
            try self.block_to_nodes.resize(self.gpa, self.blocks.items.len + 1);

            var offset: usize = 0;
            self.block_to_nodes.set(0, offset);

            for (self.blocks.items, 0..) |block, block_id| {
                var total: usize = 1;

                for (0..N) |axis| {
                    total *= self.cell_size[axis] * block.size[axis];
                }

                offset += total;

                self.block_to_nodes.set(block_id + 1, offset);
            }
        }

        /// Builds extra cell information like neighbors.
        fn buildCells(self: *@This(), mesh: *const TreeMesh) void {
            // Cache pointers
            const cells = mesh.cells.slice();

            // Reset and reserve
            try self.cells.resize(self.gpa, cells.len);
            // Set root
            self.cells.items[0].neighbors = [1]usize{boundary_index} ** Region.count;

            // Loop through every non root cell
            for (1..cells.len) |cell| {
                // Parent
                const parent = cells.items(.parent)[cell];

                // Find split index
                const split_lin = cell - cells.items(.children)[parent];
                const split = AxisMask.fromLinear(@intCast(split_lin));

                // Store all neighbors of this cell
                var neighbors: [Region.count]usize = undefined;

                for (Region.enumerate()) |region| {
                    // Central region is simply null
                    if (region == Region.central()) {
                        neighbors[region.linear()] = null_index;
                        continue;
                    }

                    // Start with current cell
                    var neighbor: usize = cell;
                    // Did we cross a coarse fine interface
                    var interface: bool = false;

                    for (0..N) |axis| {
                        assert(neighbor != boundary_index);

                        // If current neighbor is null continue.
                        if (neighbor == null_index or region.sides[axis] == .middle) {
                            continue;
                        }

                        // If we have crossed an interface, do not traverse this axis unless we are truly
                        // searching for a new node.
                        if (interface and (region.sides[axis] == .right) != split.isSet(axis)) {
                            continue;
                        }

                        // Find face to traverse
                        const face = FaceIndex{
                            .side = region.sides[axis] == .right,
                            .axis = axis,
                        };

                        // Get neighbor, searching parent neighbors if necessary
                        const traverse = cells.items(.neighbors)[neighbor][face.toLinear()];

                        if (traverse == null_index) {
                            // We had to traverse a fine-coarse boundary
                            const neighbor_parent = cells.items(.parent)[neighbor];
                            interface = true;
                            // Update neighbor
                            neighbor = cells.items(.neighbors)[neighbor_parent][face.toLinear()];
                        } else if (traverse == boundary_index) {
                            interface = false;
                            // On boundary, ignore.
                            neighbor = null_index;
                        } else {
                            // Update neighbor
                            neighbor = traverse;
                        }
                    }

                    neighbors[region.linear()] = neighbor;
                }

                // Set neighbors
                self.cells.items[cell].neighbors = neighbors;
            }

            // Set block and index of cells
            for (0..self.numBlocks()) |block_id| {
                const block = self.blockFromId(block_id);

                var cell_indices = IndexSpace.fromSize(block.size).cartesianIndices();
                while (cell_indices.next()) |index| {
                    const cell = self.cellFromBlock(block_id, index);
                    self.cells.items[cell].block = block_id;
                    self.cells.items[cell].index = index;
                }
            }
        }

        /// Total number of cells in a block given refinement.
        fn cellTotalFromRefinement(refinement: usize) usize {
            var result: usize = 1;

            for (0..refinement) |_| {
                result *= numSplits;
            }

            return result;
        }

        /// Total number of cells along each axis of a block given refinement.
        fn cellSizeFromRefinement(refinement: usize) [N]usize {
            var result: usize = 1;

            for (0..refinement) |_| {
                result *= 2;
            }

            return [1]usize{result} ** N;
        }

        pub fn buildNodeMap(self: *const @This(), comptime M: usize, map: *RangeMap) !void {
            try map.resize(map.allocator, self.blocks.items.len + 1);

            var offset: usize = 0;
            map.set(0, 0);

            for (self.blocks.items, 0..) |block, block_id| {
                var total: usize = 1;

                for (0..N) |axis| {
                    total *= self.cell_size[axis] * block.size[axis] + 2 * M;
                }

                offset += total;

                map.set(block_id + 1, offset);
            }
        }

        /// Returns the number of nodes required to store compact node vectors on the mesh.
        pub fn numNodes(self: *const @This()) usize {
            return self.block_to_nodes.items[self.block_to_nodes.items.len - 1];
        }

        /// Number of blocks in the mesh.
        pub fn numBlocks(self: *const @This()) usize {
            return self.blocks.items.len;
        }

        /// Retrieves a block descriptor from a `block_id`.
        pub fn blockFromId(self: *const @This(), block_id: usize) Block(N) {
            return self.blocks.items[block_id];
        }

        pub fn cellFromBlock(self: *const @This(), block_id: usize, index: []usize) usize {
            const block = self.blocks.items[block_id];
            const offset = self.block_to_cells[block_id];
            const linear = IndexSpace.fromSize(block.size).linearFromCartesian(index);
            return offset + self.cell_permute.permutation(block.refinement)[linear];
        }
    };
}

pub fn NodeWorker(comptime N: usize, comptime M: usize) type {
    return struct {
        gpa: Allocator,
        map: RangeMap,

        mesh: *const TreeMesh,
        manager: *const NodeManager(N),

        const Block = NodeManager(N).Block;
        const Cell = NodeManager(N).Cell;

        const TreeMesh = tree.TreeMesh(N);
        const CellPermutation = permute.CellPermutation(N);

        const AxisMask = geometry.AxisMask(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexMixin = geometry.IndexMixin(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Region = geometry.Region(N);
        const add = IndexMixin.add;
        const addSigned = IndexMixin.addSigned;
        const coarsened = IndexMixin.coarsened;
        const mul = IndexMixin.mul;
        const refined = IndexMixin.refined;
        const scaled = IndexMixin.scaled;
        const toSigned = IndexMixin.toSigned;
        const toUnsigned = IndexMixin.toUnsigned;

        const BoundaryEngine = common.BoundaryEngine(N, M);
        const Engine = common.Engine(N, M);
        const NodeSpace = common.NodeSpace(N, M);

        pub fn init(allocator: Allocator, mesh: *const TreeMesh, manager: *const NodeManager(N)) !@This() {
            var map = RangeMap.init(allocator);
            errdefer map.deinit();

            try manager.buildNodeMap(M, &map);

            return .{
                .gpa = allocator,
                .map = map,

                .mesh = mesh,
                .manager = manager,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.map.deinit(self.gpa);
        }

        pub fn numNodes(self: @This()) usize {
            return self.map.total();
        }

        // **************************
        // Ghost Nodes **************
        // **************************

        /// Fills the ghost nodes of given level.
        pub fn fillGhostNodes(self: @This(), level: usize, boundary: anytype, field: []f64) void {
            assert(field.len == self.map.total());

            // Short circuit in the case of level = 0
            if (level == 0) {
                self.fillBaseGhostNodes(boundary, field);
            } else {
                // Otherwise check each block on the level
                const blocks = self.manager.level_to_blocks.range(level);
                for (blocks.start..blocks.end) |block_id| {
                    self.fillBlockGhostNodes(block_id, boundary, field);
                }
            }
        }

        fn fillBaseGhostNodes(self: @This(), boundary: anytype, field: []usize) void {
            const block = self.manager.blockFromId(0);
            const block_field = field[self.map.offset(0)..self.map.offset(1)];
            const node_space = self.nodeSpaceFromBlock(block);
            const boundary_engine = BoundaryEngine{ .space = node_space };

            boundary_engine.fill(boundary, block_field);
        }

        /// Fills the boundary of a given block, whether by exptrapolation or prolongation.
        fn fillBlockGhostNodes(self: @This(), block_id: usize, boundary: anytype, field: []usize) void {
            assert(block_id != 0);

            const block = self.manager.blockFromId(block_id);
            const block_field = field[self.map.offset(block_id)..self.map.offset(block_id + 1)];
            const node_space = self.nodeSpaceFromBlock(block);
            const boundary_engine = BoundaryEngine{ .space = node_space };

            const regions = comptime Region.enumerateOrdered();

            // Loop over the regions
            inline for (comptime regions[1..]) |region| {
                const smask = comptime region.toMask();

                var bmask = AxisMask.initEmpty();

                for (0..N) |axis| {
                    if (region.sides[axis] == .center) {
                        continue;
                    }

                    const face: FaceIndex = .{
                        .axis = axis,
                        .side = region.sides[axis] == .right,
                    };

                    bmask.setValue(axis, block.boundary[face.toLinear()]);
                }

                // If no axes lie on a physical boundary, prolong
                if (bmask.isEmpty()) {
                    // Prolong boundary
                    self.fillBlockGhostNodesInterior(region, block_id, field);
                    // Skip following checks
                    continue;
                }

                // Otherwise enumerate the possible axis combinations
                inline for (AxisMask.enumerate()[1..]) |mask| {
                    // Intersect with mask to make sure we aren't generating unnessary code.
                    const imask = comptime mask.intersectWith(smask);

                    if (bmask == imask) {
                        boundary_engine.fillRegion(region, imask, boundary, block_field);
                    }
                }
            }
        }

        /// Fills ghost nodes on the interior of the numerical domain.
        fn fillBlockGhostNodesInterior(self: @This(), region: Region, block_id: usize, field: []f64) void {
            assert(block_id != 0);

            const block = self.manager.blockFromId(block_id);
            const block_node_space = self.nodeSpaceFromBlock(block);
            const block_field: []f64 = self.map.slice(block_id, field);

            // Cache pointers
            const cells = self.mesh.cells.slice();
            const cell_meta = self.manager.cells.slice();
            const cell_size = self.manager.cell_size;

            var cell_indices = region.innerFaceCells(block.size);

            while (cell_indices.next()) |index| {
                // Uses permutation to find actual cell
                const cell = self.manager.cellFromBlock(block_id, index);
                // Origin in node space
                const origin: [N]usize = mul(index, cell_size);
                // Get neighbor in this direction
                var neighbor = cell_meta.items(.neighbors)[cell][region.linear()];

                if (neighbor == null_index) {
                    // Prolong from coarser level.
                    const parent = cells.items(.parent)[cell];
                    // Get split mask.
                    const split_lin = cell - cells.items(.children)[parent];
                    const split = AxisMask.fromLinear(split_lin);

                    const neighbor_region = region.maskedBySplit(split);

                    neighbor = cell_meta.items(.neighbors)[parent][neighbor_region.linear()];

                    const neighbor_block_id = cell_meta.items(.block)[neighbor];
                    const neighbor_block = self.manager.blockFromId(neighbor_block_id);
                    const neighbor_index = cell_meta.items(.index)[neighbor];
                    const neighbor_field = self.map.slice(neighbor_block_id, field);
                    const neighbor_node_space = self.nodeSpaceFromBlock(neighbor_block);

                    var neighbor_origin = toSigned(refined(mul(neighbor_index, cell_size)));

                    for (0..N) |axis| {
                        if (split.isSet(axis)) {
                            neighbor_origin[axis] += cell_size[axis];
                        }

                        switch (neighbor_region.sides[axis]) {
                            .left => neighbor_origin[axis] += 2 * cell_size[axis],
                            .right => neighbor_origin[axis] -= 2 * cell_size[axis],
                            else => {},
                        }
                    }

                    // Iterate over node offsets
                    var offsets = region.nodes(M, cell_size);

                    while (offsets.next()) |offset| {
                        const node: [N]isize = addSigned(origin, offset);
                        const neighbor_node: [N]isize = addSigned(neighbor_origin, offset);

                        const v = neighbor_node_space.prolong(toUnsigned(neighbor_node), neighbor_field);
                        block_node_space.setNodeValue(node, block_field, v);
                    }
                } else {
                    // Neighbor is on same level.
                    const neighbor_block_id = cell_meta.items(.block)[neighbor];
                    const neighbor_index = cell_meta.items(.index)[neighbor];
                    const neighbor_block = self.manager.blockFromId(neighbor_block_id);
                    const neighbor_field = self.map.slice(neighbor_block_id, field);
                    const neighbor_node_space = self.nodeSpaceFromBlock(neighbor_block);

                    var neighbor_origin = toSigned(mul(neighbor_index, cell_size));

                    for (0..N) |axis| {
                        switch (region.sides[axis]) {
                            .left => neighbor_origin[axis] += cell_size[axis],
                            .right => neighbor_origin[axis] -= cell_size[axis],
                            else => {},
                        }
                    }

                    var offsets = region.nodes(M, cell_size);

                    while (offsets.next()) |offset| {
                        const node = addSigned(origin, offset);
                        const neighbor_node = addSigned(neighbor_origin, offset);

                        const v = neighbor_node_space.nodeValue(neighbor_node, neighbor_field);
                        block_node_space.setNodeValue(node, block_field, v);
                    }
                }
            }
        }

        // *******************************
        // Prolongation/Restriction ******
        // *******************************

        pub fn prolong(self: @This(), level: usize, field: []f64) void {
            assert(field.len == self.map.total());

            if (level == 0) {
                return;
            }

            const blocks = self.manager.level_to_blocks.range(level);
            for (blocks.start..blocks.end) |block_id| {
                self.prolongBlock(block_id, field);
            }
        }

        fn prolongBlock(self: @This(), block_id: usize, field: []f64) void {
            assert(block_id != 0);

            const block = self.manager.blockFromId(block_id);
            const block_field = self.map.slice(block_id, field);
            const block_node_space = self.nodeSpaceFromBlock(block);

            // Cache pointers
            const cells = self.mesh.cells.slice();
            const cell_meta = self.manager.cells.slice();
            const cell_size = self.manager.cell_size;

            var cell_indices = IndexSpace.fromSize(block.size).cartesianIndices();

            while (cell_indices.next()) |index| {
                // Uses permutation to find actual cell
                const cell = self.manager.cellFromBlock(block_id, index);
                // Get parent of current cell
                const parent = cells.items(.parent)[cell];
                // Get split mask.
                const split_lin = cell - cells.items(.children)[parent];
                const split = AxisMask.fromLinear(split_lin);
                // Get super block
                const parent_block_id: usize = cell_meta.items(.block)[parent];
                const parent_index: [N]usize = cell_meta.items(.index)[parent];
                const parent_block = self.manager.blockFromId(parent_block_id);
                const parent_block_field: []const usize = self.map.slice(parent_block_id, field);
                const parent_block_node_space = self.nodeSpaceFromBlock(parent_block);
                var parent_origin = refined(mul(parent_index, cell_size));
                // Offset origin to take into account split
                for (0..N) |axis| {
                    if (split.isSet(axis)) {
                        parent_origin[axis] += cell_size[axis];
                    }
                }
                // Get origin
                const origin = mul(index, cell_size);

                var offsets = IndexSpace.fromSize(cell_size).cartesianIndices();
                while (offsets.next()) |offset| {
                    const node = add(origin, offset);
                    const subnode = add(parent_origin, offset);

                    const val = parent_block_node_space.prolong(subnode, parent_block_field);
                    block_node_space.setValue(node, block_field, val);
                }
            }
        }

        pub fn restrict(self: @This(), level: usize, field: []f64) void {
            assert(field.len == self.map.total());

            if (level == 0) {
                return;
            }

            const blocks = self.manager.level_to_blocks.range(level);
            for (blocks.start..blocks.end) |block_id| {
                self.restrictBlock(block_id, field);
            }
        }

        fn restrictBlock(self: @This(), block_id: usize, field: []f64) void {
            assert(block_id != 0);

            const block = self.manager.blockFromId(block_id);
            const block_field = self.map.slice(block_id, field);
            const block_node_space = self.nodeSpaceFromBlock(block);

            // Cache pointers
            const cells = self.mesh.cells.slice();
            const cell_meta = self.manager.cells.slice();
            const cell_size = self.manager.cell_size;

            var cell_indices = IndexSpace.fromSize(block.size).cartesianIndices();

            while (cell_indices.next()) |index| {
                // Uses permutation to find actual cell
                const cell = self.manager.cellFromBlock(block_id, index);
                // Get parent of current cell
                const parent = cells.items(.parent)[cell];
                // Get split mask.
                const split_lin = cell - cells.items(.children)[parent];
                const split = AxisMask.fromLinear(split_lin);
                // Get super block
                const parent_block_id: usize = cell_meta.items(.block)[parent];
                const parent_index: [N]usize = cell_meta.items(.index)[parent];
                const parent_block = self.manager.blockFromId(parent_block_id);
                const parent_block_field: []const usize = self.map.slice(parent_block_id, field);
                const parent_block_node_space = self.nodeSpaceFromBlock(parent_block);

                var parent_origin = mul(parent_index, cell_size);
                // Offset origin to take into account split
                for (0..N) |axis| {
                    if (split.isSet(axis)) {
                        parent_origin[axis] += cell_size[axis] / 2;
                    }
                }
                // Get origin
                const origin = coarsened(mul(index, cell_size));

                var offsets = IndexSpace.fromSize(cell_size).cartesianIndices();
                while (offsets.next()) |offset| {
                    const node = add(parent_origin, offset);
                    const supernode = add(origin, offset);

                    const val = block_node_space.restrict(supernode, block_field);
                    parent_block_node_space.setValue(node, parent_block_field, val);
                }
            }
        }

        // ********************************
        // Projection/Application *********
        // ********************************

        /// Using the given projection to set the values of the field on this level.
        pub fn project(self: @This(), level: usize, projection: anytype, field: []f64) void {
            const Proj = @TypeOf(projection);

            if (comptime !common.isProjection(N, M)(Proj)) {
                @compileError("Projection must satisfy isProjection trait");
            }

            assert(field == self.map.total());

            const blocks = self.manager.level_to_blocks.range(level);
            for (blocks.start..blocks.end) |block_id| {
                self.projectBlock(block_id, projection, field);
            }
        }

        /// Set values of the field on the given block using the projection.
        fn projectBlock(self: @This(), block_id: usize, projection: anytype, field: []f64) void {
            const block_field: []f64 = self.map.slice(block_id, field);
            const block_range = self.map.range(block_id);

            const node_space = self.nodeSpaceFromBlock(self.manager.blockFromId(block_id));

            var cells = node_space.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const engine = Engine{
                    .space = node_space,
                    .cell = cell,
                    .start = block_range.start,
                    .end = block_range.end,
                };

                node_space.setValue(cell, block_field, projection.project(engine));
            }
        }

        /// Applies the operator to the source function on this level, storing the result on field.
        pub fn apply(self: @This(), level: usize, operator: anytype, src: []const f64, field: []f64) void {
            const Oper = @TypeOf(operator);

            if (comptime !common.isOperator(N, M)(Oper)) {
                @compileError("Operator must satisfy isOperator trait");
            }

            assert(field == self.map.total());
            assert(src == self.map.total());

            const blocks = self.manager.level_to_blocks.range(level);
            for (blocks.start..blocks.end) |block_id| {
                self.applyBlock(block_id, operator, src, field);
            }
        }

        /// Applies the operator to the source function on this block, storing the result on field.
        fn applyBlock(self: @This(), block_id: usize, operator: anytype, src: []const f64, field: []f64) void {
            const block_field: []f64 = self.map.slice(block_id, field);
            const block_range = self.map.range(block_id);

            const node_space = self.nodeSpaceFromBlock(self.manager.blockFromId(block_id));

            var cells = node_space.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const engine = Engine{
                    .space = node_space,
                    .cell = cell,
                    .start = block_range.start,
                    .end = block_range.end,
                };

                node_space.setValue(cell, block_field, operator.apply(engine, src));
            }
        }

        // ***************************************
        // Smoothing *****************************
        // ***************************************

        /// Performs jacobi smoothing on this level.
        pub fn smooth(self: @This(), level: usize, operator: anytype, src: []const f64, rhs: []const f64, field: []f64) void {
            const Oper = @TypeOf(operator);

            if (comptime !common.isOperator(N, M)(Oper)) {
                @compileError("Operator must satisfy isOperator trait");
            }

            assert(field == self.map.total());
            assert(src == self.map.total());

            const blocks = self.manager.level_to_blocks.range(level);
            for (blocks.start..blocks.end) |block_id| {
                self.smoothBlock(block_id, operator, src, rhs, field);
            }
        }

        /// Performs jacobi smoothing on this block.
        fn smoothBlock(self: @This(), block_id: usize, operator: anytype, src: []const f64, rhs: []const f64, field: []f64) void {
            const block_field: []f64 = self.map.slice(block_id, field);
            const block_rhs: []const f64 = self.map.slice(block_id, rhs);
            const block_src: []const f64 = self.map.slice(block_id, src);
            const block_range = self.map.range(block_id);

            const node_space = self.nodeSpaceFromBlock(self.manager.blockFromId(block_id));

            var cells = node_space.cellSpace().cartesianIndices();

            while (cells.next()) |cell| {
                const engine = Engine{
                    .space = node_space,
                    .cell = cell,
                    .start = block_range.start,
                    .end = block_range.end,
                };

                const sval = node_space.value(cell, block_src);
                const rval = node_space.value(cell, block_rhs);
                const app = operator.apply(engine, src);
                const appDiag = operator.applyDiag(engine);

                node_space.setValue(cell, block_field, sval + 2.0 / 3.0 * (rval - app) / appDiag);
            }
        }

        /// Helper function for determining the node space of a block
        fn nodeSpaceFromBlock(self: @This(), block: Block) NodeSpace {
            var size: [N]usize = undefined;

            for (0..N) |axis| {
                size[axis] = block.size[axis] * self.manager.cell_size[axis];
            }

            return .{
                .bounds = block.bounds,
                .size = size,
            };
        }
    };
}
