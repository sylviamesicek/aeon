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
///
/// All routines for transfering between node vectors, prolongation, smoothing, and filling ghost nodes are handled
/// by seperate `NodeWorker` classes which handle parallelism and dispatching work to other devices
/// (like the GPU or other processes).
pub fn NodeManager(comptime N: usize, comptime M: usize) type {
    return struct {
        gpa: Allocator,
        /// The set of blocks that make up the mesh.
        blocks: ArrayListUmanaged(Block),
        /// Additional info stored per mesh cell.
        cells: MultiArrayList(Cell),

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

        // Subtypes

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
            /// Cell offset
            cells: usize,
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

            assert(max_refinement > 1);

            return .{
                .gpa = allocator,
                .blocks = .{},
                .cells = .{},

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

            self.block_to_nodes.deinit(self.gpa);
            self.level_to_blocks.deinit(self.gpa);

            self.cell_permute.deinit(self.gpa);
        }

        /// Builds a new `NodeManager` from a mesh, using the given allocator for scratch allocations.
        pub fn build(self: *@This(), allocator: Allocator, mesh: *const TreeMesh) !void {
            // ****************************
            // Computes Blocks + Offsets
            try self.buildBlocks(allocator, mesh);

            // *****************************
            // Compute neighbors
            try self.buildCells(mesh);

            // ********************************
            // Compute block to nodes map
            try self.buildBlockToNodes();
        }

        /// Runs a recursive algorithm that traverses the tree from root to leaves, computing valid blocks while doing so.
        /// This algorithm works level by level (queueing up blocks for use on the next level). And thus preserves the invariant
        /// that all blocks on the same level are contiguous in the block array.
        fn buildBlocks(self: *@This(), allocator: Allocator, mesh: *const TreeMesh) !void {
            // Cache
            const cells = mesh.cells.slice();
            const levels = mesh.numLevels();

            // Reset level offsets
            try self.level_to_blocks.resize(allocator, levels + 1);

            // // Reset blocks
            // self.blocks.clearRetainingCapacity();

            // try self.blocks.append(self.gpa, .{
            //     .refinement = 0,
            //     .boundary = [1]bool{true} ** FaceIndex.count,
            //     .cells = 0,
            // });

            // Iterate levels
            const BlockMeta = struct {
                /// Base cell
                base: usize,
                /// Physical Boundaries
                boundary: [FaceIndex.count]bool,
                /// Refinement level
                refinement: usize,
            };

            // Allocate block meta data
            var blocks: ArrayListUmanaged(BlockMeta) = .{};
            defer blocks.deinit(allocator);
            // Stack for each level
            var stack: ArrayListUmanaged(BlockMeta) = .{};
            defer stack.deinit(allocator);

            try blocks.append(allocator, .{
                .base = 0,
                .boundary = [1]bool{true} ** FaceIndex.count,
                .refinement = 0,
            });

            // Set base level offset
            self.level_to_blocks.set(0, 0);
            self.level_to_blocks.set(1, 1);

            for (1..levels) |level| {
                const coarse = self.level_to_blocks.range(level - 1);

                // Reset stack
                stack.clearRetainingCapacity();

                // Transfer all blocks on l - 1 to stack in reverse order
                for (0..coarse.end - coarse.start) |rev_idx| {
                    const idx = coarse.end - 1 - rev_idx;

                    try stack.append(allocator, blocks.items[idx]);
                }

                // Iterate until the stack is empty
                while (stack.popOrNull()) |block| {
                    // Find cell offset and cell total.
                    var cell_offset: usize = block.base;

                    for (0..block.refinement) |_| {
                        cell_offset = cells.items(.children)[cell_offset];
                    }

                    const cell_total = cellTotalFromRefinement(block.refinement);

                    // Count number of leaves
                    var leaf_count: usize = 0;

                    for (cell_offset..cell_offset + cell_total) |cell| {
                        if (cells.items(.children)[cell] == null_index) {
                            leaf_count += 1;
                        }
                    }

                    if (leaf_count == cell_total) {
                        // Skip block if it is a leaf
                        continue;
                    } else if (leaf_count == 0 and block.refinement + 1 < self.cell_permute.maxRefinement()) {
                        // Accept
                        try blocks.append(allocator, .{
                            .boundary = block.boundary,
                            .base = block.base,
                            .refinement = block.refinement + 1,
                        });

                        continue;
                    }

                    assert(block.refinement > 0);

                    const children = cells.items(.children)[block.base];

                    assert(children != null_index);

                    for (AxisMask.enumerate()) |split| {
                        var boundary = block.boundary;

                        for (split.innerFaces()) |face| {
                            boundary[face.toLinear()] = false;
                        }

                        try stack.append(allocator, .{
                            .boundary = boundary,
                            .refinement = block.refinement - 1,
                            .base = children + split.toLinear(),
                        });
                    }
                }

                // Update level to blocks
                self.level_to_blocks.set(level + 1, blocks.items.len);
            }

            // Transfer to blocks

            self.blocks.clearRetainingCapacity();

            for (blocks.items) |meta| {
                // Find cell offset and cell total.
                var cell_offset: usize = meta.base;

                for (0..meta.refinement) |_| {
                    cell_offset = cells.items(.children)[cell_offset];
                }

                try self.blocks.append(self.gpa, .{
                    .bounds = cells.items(.bounds)[meta.base],
                    .size = cellSizeFromRefinement(meta.refinement),
                    .boundary = meta.boundary,
                    .refinement = meta.refinement,
                    .cells = cell_offset,
                });
            }
        }

        /// Computes the block to node offset map.
        fn buildBlockToNodes(self: *@This()) !void {
            try self.block_to_nodes.resize(self.gpa, self.blocks.items.len + 1);

            var offset: usize = 0;
            self.block_to_nodes.set(0, offset);

            for (self.blocks.items, 0..) |block, block_id| {
                var total: usize = 1;

                for (0..N) |axis| {
                    total *= self.cell_size[axis] * block.size[axis] + 2 * M;
                }

                offset += total;

                self.block_to_nodes.set(block_id + 1, offset);
            }
        }

        /// Builds extra per-cell information like neighbors.
        fn buildCells(self: *@This(), mesh: *const TreeMesh) !void {
            // Cache pointers
            const cells = mesh.cells.slice();

            // Reset and reserve
            try self.cells.resize(self.gpa, cells.len);
            // Set root
            self.cells.items(.neighbors)[0] = [1]usize{boundary_index} ** Region.count;

            // Loop through every non root cell
            for (1..cells.len) |cell| {
                // Store all neighbors of this cell
                var neighbors: [Region.count]usize = undefined;

                // Get split index
                const parent = cells.items(.parent)[cell];
                const children = cells.items(.children)[parent];
                const split = AxisMask.fromLinear(cell - cells.items(.children)[parent]);

                for (Region.enumerate()) |region| {
                    // Central region is simply null
                    if (std.meta.eql(region, Region.central())) {
                        neighbors[region.linear()] = null_index;
                        continue;
                    }

                    // Get neighbor split
                    var nsplit = split;

                    for (0..N) |axis| {
                        if (region.sides[axis] != .middle) {
                            nsplit.toggle(axis);
                        }
                    }

                    // Masked region
                    const mregion = region.maskedBySplit(split);

                    if (std.meta.eql(mregion, Region.central())) {
                        // Inner face
                        neighbors[region.linear()] = children + nsplit.toLinear();

                        continue;
                    }

                    const neighbor = self.cells.items(.neighbors)[parent][mregion.linear()];
                    assert(neighbor != null_index);

                    if (neighbor == boundary_index) {
                        neighbors[region.linear()] = boundary_index;
                    } else {
                        const nchildren = cells.items(.children)[neighbor];

                        if (nchildren == null_index) {
                            neighbors[region.linear()] = null_index;
                        } else {
                            neighbors[region.linear()] = nchildren + nsplit.toLinear();
                        }
                    }

                    // // Start with current cell
                    // var neighbor: usize = cell;

                    // for (0..N) |axis| {
                    //     assert(neighbor != boundary_index and neighbor != null_index);

                    //     // If side is middle skip axis.
                    //     if (region.sides[axis] == .middle) {
                    //         continue;
                    //     }

                    //     // Find face to traverse
                    //     const face = FaceIndex{
                    //         .side = region.sides[axis] == .right,
                    //         .axis = axis,
                    //     };

                    //     // Get neighbor, searching parent neighbors if necessary
                    //     neighbor = cells.items(.neighbors)[neighbor][face.toLinear()];

                    //     if (neighbor == null_index or neighbor == boundary_index) {
                    //         break;
                    //     }

                    //     // if (neighbor == null_index) {
                    //     //     assert(coarse == false);

                    //     //     coarse = true;
                    //     //     // Get parent of neighbor
                    //     //     const neighbor_parent = cells.items(.parent)[neighbor];
                    //     //     neighbor = cells.items(.neighbors)[neighbor_parent][face.toLinear()];
                    //     // } else if (neighbor == boundary_index) {
                    //     //     coarse = false;
                    //     //     break;
                    //     // }
                    // }

                    // if (neighbor == null_index) {
                    //     const mregion = region.maskedBySplit(split);

                    //     const parent_neighbor = self.cells.items(.neighbors)[parent][mregion.linear()];

                    //     assert(parent_neighbor != null_index);

                    //     if (parent_neighbor == boundary_index) {
                    //         neighbor = boundary_index;
                    //     } else {
                    //         const parent_neighbor_children = cells.items(.children)[parent_neighbor];

                    //         if (parent_neighbor_children != null_index) {
                    //             var neighbor_split = split;

                    //             for (0..N) |axis| {
                    //                 if (region.sides[axis] != .middle) {
                    //                     neighbor_split.toggle(axis);
                    //                 }
                    //             }

                    //             neighbor = parent_neighbor_children + neighbor_split.toLinear();
                    //         }
                    //     }
                    // }

                    // neighbors[region.linear()] = neighbor;
                }

                // Set neighbors
                self.cells.items(.neighbors)[cell] = neighbors;
            }

            // Set block and index of cells
            for (0..self.numBlocks()) |block_id| {
                const block = self.blockFromId(block_id);

                var cell_indices = IndexSpace.fromSize(block.size).cartesianIndices();
                while (cell_indices.next()) |index| {
                    const cell = self.cellFromBlock(block_id, index);
                    self.cells.items(.block)[cell] = block_id;
                    self.cells.items(.index)[cell] = index;
                }
            }
        }

        /// Total number of cells in a block given refinement.
        fn cellTotalFromRefinement(refinement: usize) usize {
            var result: usize = 1;

            for (0..refinement) |_| {
                result *= AxisMask.count;
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

        /// Returns the number of nodes required to store node vectors on the mesh.
        pub fn numNodes(self: *const @This()) usize {
            return self.block_to_nodes.total();
        }

        /// Returns the number of nodes on the base level of the mesh.
        pub fn numBaseNodes(self: *const @This()) usize {
            return self.block_to_nodes.size(0);
        }

        /// Number of blocks in the mesh.
        pub fn numBlocks(self: *const @This()) usize {
            return self.blocks.items.len;
        }

        pub fn blockNodes(self: *const @This(), lock_id: usize, data: anytype) @TypeOf(data) {
            return self.block_to_nodes.slice(lock_id, data);
        }

        /// Retrieves a block descriptor from a `block_id`.
        pub fn blockFromId(self: *const @This(), block_id: usize) Block {
            return self.blocks.items[block_id];
        }

        pub fn cellFromBlock(self: *const @This(), block_id: usize, index: [N]usize) usize {
            const block = self.blocks.items[block_id];
            const linear = IndexSpace.fromSize(block.size).linearFromCartesian(index);

            assert(block.refinement <= self.cell_permute.maxRefinement());

            return block.cells + self.cell_permute.permutation(block.refinement)[linear];
        }

        /// Retrieves the spatial spacing of a given level
        pub fn spacing(self: *const @This(), level: usize) [N]f64 {
            var result: [N]f64 = self.blocks.items[0].bounds.size;

            for (0..N) |axis| {
                result[axis] /= @floatFromInt(self.cell_size[axis]);
            }

            for (0..level) |_| {
                for (0..N) |axis| {
                    result[axis] /= 2.0;
                }
            }

            return result;
        }

        pub fn minSpacing(self: *const @This()) f64 {
            const sp = self.spacing(self.level_to_blocks.len() - 2);

            var result = sp[0];

            for (1..N) |axis| {
                result = @max(result, sp[axis]);
            }

            return result;
        }
    };
}
