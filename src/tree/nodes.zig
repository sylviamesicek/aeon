const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUmanaged = std.ArrayListUnmanaged;
const assert = std.debug.assert;
const panic = std.debug.panic;

const tree = @import("tree.zig");
const null_index = tree.null_index;
const boundary_index = tree.boundary_index;

const geometry = @import("../geometry/geometry.zig");

pub fn Block(comptime N: usize) type {
    return struct {
        /// Physical bounds of block.
        bounds: RealBox = RealBox.unit,
        /// Number of cells along each axis of block.
        size: [N]usize = [1]usize{1} ** N,
        /// Refinement level of block
        refinement: usize,

        const RealBox = geometry.RealBox(N);
    };
}

/// A map from blocks (defined in a mesh agnostic way) into a buffered node vector.
/// This is filled by the appropriate mesh routine.
pub const NodeMap = struct {
    offsets: ArrayList(usize),

    pub fn init(allocator: Allocator) NodeMap {
        const offsets = ArrayList(usize).init(allocator);
        return .{ .offsets = offsets };
    }

    /// Frees a `CellMap`.
    pub fn deinit(self: *NodeMap) void {
        self.offsets.deinit();
    }

    /// Returns the cells offset for a given block.
    pub fn offset(self: NodeMap, block: usize) usize {
        return self.offsets.items[block];
    }

    /// Returns the total number of cells in a given block.
    pub fn total(self: NodeMap, block: usize) usize {
        return self.offsets.items[block + 1] - self.offsets.items[block];
    }

    /// Returns of slice of cells for a given block.
    pub fn slice(self: NodeMap, block: usize, data: anytype) @TypeOf(data) {
        return data[self.offsets.items[block]..self.offsets.items[block + 1]];
    }

    /// Returns the number of nodes in the total map.
    pub fn numNodes(self: NodeMap) usize {
        return self.offsets.items[self.offsets.items.len - 1];
    }
};

pub fn NodeManager(comptime N: usize) type {
    return struct {
        gpa: Allocator,
        /// The set of blocks that make up the mesh.
        blocks: ArrayListUmanaged(Block(N)),
        /// Map
        block_to_cells: ArrayListUmanaged(usize),
        /// Packed node map
        block_to_nodes: ArrayListUmanaged(usize),
        /// A map from levels to block offsets.
        level_to_blocks: ArrayListUmanaged(usize),
        /// A full map of neighbors of cells including corners.
        cell_neighbors: ArrayListUmanaged([numRegions]usize),
        /// Number of nodes per axis of cell
        cell_size: [N]usize = [1]usize{16} ** N,

        const TreeMesh = tree.TreeMesh(N);
        const FaceIndex = geometry.FaceIndex(N);
        const Region = geometry.Region(N);
        const SplitIndex = geometry.SplitIndex(N);
        const numRegions = Region.count;
        const numSplitIndices = SplitIndex.count;

        pub fn init(allocator: Allocator) @This() {
            return .{
                .gpa = allocator,
                .blocks = .{},
                .block_to_cells = .{},
                .block_to_nodes = .{},
                .level_to_blocks = .{},
                .cell_neighbors = .{},
            };
        }

        pub fn deinit(self: *@This()) void {
            self.blocks.deinit(self.gpa);
            self.block_to_nodes.deinit(self.gpa);
            self.level_to_blocks.deinit(self.gpa);
            self.cell_neighbors.deinit(self.gpa);
        }

        pub fn build(self: *@This(), allocator: Allocator, mesh: *const TreeMesh) !void {
            // ****************************
            // Computes Blocks + Offsets
            self.computeBlocks(allocator, mesh);

            // *****************************
            // Compute neighbors
            self.computeCellNeighbors(mesh);

            // ********************************
            // Compute block to nodes map
            self.computeBlockToNodes();
        }

        fn computeBlocks(self: *@This(), allocator: Allocator, mesh: *const TreeMesh) void {
            // Cache
            const cells = mesh.cells.slice();
            const levels = mesh.numLevels();

            // Reset level offsets
            self.level_to_blocks.shrinkRetainingCapacity(0);
            try self.level_to_blocks.ensureTotalCapacity(self.gpa, levels + 1);

            self.level_to_blocks.appendAssumeCapacity(0);
            self.level_to_blocks.appendAssumeCapacity(1);

            // Reset blocks
            self.blocks.shrinkRetainingCapacity(0);
            self.block_to_cells.shrinkRetainingCapacity(0);

            try self.blocks.append(self.gpa, .{ .refinement = 0 });
            try self.block_to_cells.append(self.gpa, 0);

            // Iterate levels
            const BlockInfo = struct {
                refinement: usize,
                offset: usize,
                total: usize,
            };

            // Allocate scratch data
            var stack: ArrayListUmanaged(BlockInfo) = .{};
            defer stack.deinit(allocator);

            for (1..levels) |level| {
                // Transfer all blocks on l - 1 to stack (in reverse)
                for (self.level_to_blocks.items[level - 1]..self.level_to_blocks.items[level]) |rev_idx| {
                    const idx = self.level_to_blocks.items[level] - 1 - rev_idx;
                    const block = self.blocks.items[idx];
                    const offset = self.block_to_cells.items[idx];
                    const total = cellTotalFromRefinement(block.refinement);

                    try stack.append(allocator, .{
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
                    } else if (leaf_count == 0) {
                        // Accept

                        // Kind of hacky, this depends on the fact that all grandchildren are stored contigiously.
                        // Document this invariant.

                        const granchildren = cells.items(.children)[block.offset];

                        try self.blocks.append(self.gpa, .{ .refinement = block.refinement + 1 });
                        try self.block_to_cells.append(self.gpa, granchildren);
                    }

                    // Some are leaves, some are not. Perform subdivision.
                    const refinement_sub = block.refinement - 1;
                    const total_sub = cellTotalFromRefinement(refinement_sub);

                    var offset_sub = block.offset;

                    for (SplitIndex.enumerate()) |_| {
                        try stack.append(allocator, .{
                            .refinement = refinement_sub,
                            .node_offset = offset_sub,
                            .node_total = total_sub,
                        });

                        offset_sub += total_sub;
                    }
                }

                try self.level_to_blocks.append(self.gpa, self.blocks.len);
            }

            // Append total number of cells
            try self.block_to_cells.append(self.gpa, cells.len);
            // Reset permutations
            try self.block_cell_perm.resize(self.gpa, cells.len);

            // Compute bounds and size
            for (self.blocks.items, self.block_to_cells.items) |*block, cell| {
                var base: usize = cell;

                for (0..block.refinement) |_| {
                    base = cells.items(.parent)[base];
                }

                block.bounds = cells.items(.bounds)[base];
                block.size = cellSizeFromRefinement(block.refinement);
            }
        }

        fn computeCellNeighbors(self: *@This(), mesh: *const TreeMesh) void {
            // Cache pointers
            const cells = mesh.cells.slice();

            // Reset and reserve
            self.cell_neighbors.shrinkRetainingCapacity(0);
            try self.cell_neighbors.ensureTotalCapacity(self.gpa, cells.len);
            // Add root
            self.cell_neighbors.appendAssumeCapacity([1]usize{boundary_index} ** numRegions);

            // Loop through every non root node
            for (1..cells.len) |node| {
                // Parent
                const parent = cells.items(.parent)[node];

                // Find split index
                const sidx = node - cells.items(.children)[parent];
                const split = SplitIndex.fromLinear(@intCast(sidx));
                const split_sides = split.toCartesian();

                // Store all neighbors of this cell
                var neighbors: [numRegions]usize = undefined;

                for (Region.enumerate()) |region| {
                    // Central region is simply null
                    if (region == Region.central()) {
                        neighbors[region.linear()] = null_index;
                        continue;
                    }

                    var neighbor: usize = node;
                    var neighbor_coarse: bool = false;

                    for (0..N) |axis| {
                        if (neighbor == null_index or neighbor == boundary_index or region.sides[axis] == .middle) {
                            continue;
                        }

                        if (neighbor_coarse and (region.sides[axis] == .right) != split_sides[axis]) {
                            continue;
                        }

                        // Find face to traverse
                        const face = FaceIndex{
                            .side = region.sides[axis] == .right,
                            .axis = axis,
                        };

                        // Get neighbor, searching parent neighbors if necessary
                        var traverse = cells.items(.neighbors)[neighbor][face.toLinear()];

                        if (traverse == null_index) {
                            // We had to traverse a fine-coarse boundary
                            const neighbor_parent = cells.items(.parent)[neighbor];
                            neighbor_coarse = true;
                            // Update traverse
                            traverse = cells.items(.neighbors)[neighbor_parent][face.toLinear()];
                        } else if (neighbor == boundary_index) {
                            // No update needed if face is on boundary
                            neighbor_coarse = false;
                            continue;
                        }

                        // Update neighbor
                        neighbor = traverse;
                    }

                    neighbors[region.linear()] = neighbor;
                }

                self.cell_neighbors.appendAssumeCapacity(neighbors);
            }
        }

        fn computeBlockToNodes(self: *@This()) void {
            self.block_to_nodes.shrinkRetainingCapacity(0);
            try self.block_to_nodes.ensureTotalCapacity(self.gpa, self.blocks.items.len + 1);

            var offset: usize = 0;
            self.block_to_nodes.appendAssumeCapacity(offset);

            for (self.blocks.items) |block| {
                var total: usize = 1;

                for (0..N) |axis| {
                    total *= self.cell_size[axis] * block.size[axis];
                }

                offset += total;

                self.block_to_nodes.appendAssumeCapacity(offset);
            }
        }

        fn cellTotalFromRefinement(refinement: usize) usize {
            var result: usize = 1;

            for (0..refinement) |_| {
                result *= numSplitIndices;
            }

            return result;
        }

        fn cellSizeFromRefinement(refinement: usize) [N]usize {
            var result: usize = 1;

            for (0..refinement) |_| {
                result *= 2;
            }

            return [1]usize{result} ** N;
        }

        /// Returns the number of nodes required to store compact node vectors on the mesh.
        pub fn numNodes(self: *const @This()) usize {
            return self.block_to_nodes.items[self.block_to_nodes.items.len - 1];
        }

        pub fn buildNodeMap(self: *const @This(), comptime M: usize, map: *NodeMap) !void {
            map.offsets.shrinkRetainingCapacity(0);
            try map.offsets.ensureTotalCapacity(self.blocks.items.len + 1);

            var offset: usize = 0;
            map.offsets.appendAssumeCapacity(offset);

            for (self.blocks.items) |block| {
                var total: usize = 1;

                for (0..N) |axis| {
                    total *= self.cell_size[axis] * block.size[axis] + 2 * M;
                }

                offset += total;

                map.offsets.appendAssumeCapacity(offset);
            }
        }
    };
}

pub fn CellPermutation(comptime N: usize) type {
    return struct {
        buffer: []const usize,
        offsets: []const usize,

        const IndexMixin = geometry.IndexMixin(N);
        const IndexSpace = geometry.IndexSpace(N);
        const SplitIndex = geometry.SplitIndex(N);

        pub fn init(allocator: Allocator, max_refinement: usize) !@This() {
            // Compute offsets
            var offsets: []usize = try allocator.alloc(usize, max_refinement + 1);
            errdefer allocator.free(offsets);

            var offset: usize = 0;
            var total: usize = 1;

            offsets[0] = offset;

            for (0..max_refinement) |i| {
                offset += total;
                offsets[i] = offset;
                total *= SplitIndex.count;
            }

            // Compute buffer
            var buffer: []usize = try allocator.alloc(usize, offsets[max_refinement]);
            errdefer allocator.free(buffer);

            buffer[0] = 0;

            var size: usize = 1;

            for (0..(max_refinement - 1)) |i| {
                const src: []const usize = buffer[offset[i]..offset[i + 1]];
                const dest: []usize = buffer[offset[i + 1]..offset[i + 2]];

                const sspace = IndexSpace.fromSize(IndexMixin.splat(size));
                const dspace = IndexSpace.fromSize(IndexMixin.splat(size * 2));

                var indices = sspace.cartesianIndices();

                while (indices.next()) |sindex| {
                    const dindex = IndexMixin.refined(sindex);

                    const slinear = sspace.linearFromCartesian(sindex);
                    const dlinear = dspace.linearFromCartesian(dindex);

                    for (SplitIndex.enumerate()) |split| {
                        dest[dlinear + split.toLinear()] = src[slinear] + split.toLinear();
                    }
                }

                size *= 2;
            }

            return .{
                .buffer = buffer,
                .offsets = offsets,
            };
        }

        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.buffer);
            allocator.free(self.offsets);
        }

        pub fn permuation(self: @This(), refinement: usize) []const usize {
            return self.buffer[self.offsets[refinement]..self.offsets[refinement + 1]];
        }
    };
}
