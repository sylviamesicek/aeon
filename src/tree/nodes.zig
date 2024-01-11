const std = @import("std");
const Allocator = std.mem.Allocator;
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
        bounds: RealBox,
        /// Number of cells along each axis of block.
        size: [N]usize,
        /// Refinement level of block
        refinement: usize,

        const RealBox = geometry.RealBox(N);
    };
}

pub fn NodeManager(comptime N: usize) type {
    return struct {
        gpa: Allocator,
        /// The set of blocks that make up the mesh.
        blocks: ArrayListUmanaged(Block(N)),
        block_to_nodes: ArrayListUmanaged(usize),
        /// A map from levels to block offsets.
        level_to_blocks: ArrayListUmanaged(usize),
        /// A full map of neighbors of cells including corners.
        cell_neighbors: ArrayListUmanaged([numRegions]usize),
        // Number of nodes per axis of cell
        cell_size: [N]usize,

        const TreeMesh = tree.TreeMesh(N);
        const FaceIndex = geometry.FaceIndex(N);
        const Region = geometry.Region(N);
        const SplitIndex = geometry.SplitIndex(N);
        const numRegions = geometry.numRegions(N);
        const numSplitIndices = geometry.numSplitIndices(N);

        pub fn init(allocator: Allocator) @This() {
            return .{
                .gpa = allocator,
                .blocks = .{},
                .level_to_blocks = .{},
                .cell_neighbors = .{},
            };
        }

        pub fn deinit(self: *@This()) void {
            self.blocks.deinit(self.gpa);
            self.level_to_blocks.deinit(self.gpa);
            self.cell_neighbors.deinit(self.gpa);
        }

        pub fn build(self: *@This(), allocator: Allocator, mesh: *const TreeMesh) !void {
            // Reset offsets
            self.level_to_blocks.shrinkRetainingCapacity(0);
            try self.level_to_blocks.ensureTotalCapacity(self.gpa, mesh.level_to_cells.items.len);

            try self.level_to_blocks.append(self.gpa, 0);
            try self.level_to_blocks.append(self.gpa, 1);

            // Reset blocks
            self.blocks.shrinkRetainingCapacity(0);

            try self.blocks.append(self.gpa, .{
                .refinement = 0,
                .node_offset = 0,
                .node_total = 1,
            });

            // ****************************
            // Computes Blocks + Offsets

            // Cache pointers
            const cells = mesh.cells.slice();

            // Allocate scratch data
            var stack: ArrayListUmanaged(Block(N)) = .{};
            defer stack.deinit(allocator);

            // Iterate levels
            const levels = mesh.numLevels();

            for (1..levels) |level| {
                // Transfer all blocks on l - 1 to stack (in reverse)
                for (self.level_to_blocks.items[level - 1]..self.level_to_blocks.items[level]) |rev_idx| {
                    const idx = self.level_to_blocks.items[level] - 1 - rev_idx;
                    try stack.append(allocator, self.blocks.items[idx]);
                }

                // Iterate until the stack is empty
                while (stack.popOrNull()) |block| {
                    var leaf_count: usize = 0;

                    for (block.node_offset..block.node_offset + block.node_total) |node| {
                        if (cells.items(.children)[node] == null_index) {
                            leaf_count += 1;
                        }
                    }

                    if (leaf_count == block.node_total) {
                        // Skip block
                        continue;
                    } else if (leaf_count == 0) {
                        // Accept

                        // Kind of hacky, this depends on the fact that all grandchildren are stored contigiously.
                        // Document this invariant.

                        const node_offset = cells.items(.children)[block.node_offset];

                        try self.blocks.append(self.gpa, .{
                            .refinement = block.refinement + 1,
                            .node_offset = node_offset,
                            .node_total = block.node_total * numSplitIndices,
                        });
                    }

                    // Some are leaves, some are not. Perform subdivision.
                    const refinement_sub = block.refinement - 1;
                    const total_sub = blk: {
                        var result: usize = 1;

                        for (0..refinement_sub) |_| {
                            result *= 2;
                        }

                        break :blk result;
                    };

                    var offset_sub = block.node_offset;

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

            // *****************************
            // Compute neighbors

            // Reset and reserve
            self.cell_neighbors.shrinkRetainingCapacity(0);
            try self.cell_neighbors.ensureTotalCapacity(self.gpa, cells.len);
            // Add root
            self.cell_neighbors.appendAssumeCapacity([1]usize{boundary_index} ** numRegions);

            // Loop through every other node
            for (1..cells.len) |node| {
                const parent = cells.items(.parent)[node];

                const sidx = node - cells.items(.children)[parent];
                const split = SplitIndex.fromLinear(@intCast(sidx));
                const split_sides = split.toCartesian();

                var neighbors: [numRegions]usize = undefined;

                for (Region.enumerate()) |region| {
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
                        var traverse = cells.items(.neighbors)[node][face.toLinear()];

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
    };
}
