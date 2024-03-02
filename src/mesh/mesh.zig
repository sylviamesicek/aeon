const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;

const geometry = @import("../geometry/geometry.zig");
const utils = @import("../utils.zig");

const manager = @import("manager.zig");
const multigrid = @import("multigrid.zig");
const permute = @import("permute.zig");

pub const MultigridMethod = multigrid.MultigridMethod;
pub const NodeManager = manager.NodeManager;

/// A null index.
pub const null_index: usize = std.math.maxInt(usize);
/// This relies on the fact that the root node can never be a neighbor.
pub const boundary_index: usize = std.math.maxInt(usize) - 1;

/// A quadtree based mesh. The domain must be rectangular.
pub fn Mesh(comptime N: usize) type {
    return struct {
        /// Internal Allocator
        gpa: Allocator,
        /// Physical Bounds of the mesh.
        bounds: RealBox,
        /// The cells that make up this mesh
        cells: MultiArrayList(Cell),
        /// A set of offsets for each level of the mesh
        /// to the corresponding cells.
        levels: RangeMap,

        // *****************************
        // Aliases

        const Self = @This();

        const AxisMask = geometry.AxisMask(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const RealBox = geometry.RealBox(N);
        const Region = geometry.Region(N);

        const Range = utils.Range;
        const RangeMap = utils.RangeMap;

        // *******************************
        // Subtypes

        /// A cell in the mesh.
        const Cell = struct {
            /// The physical bounds of a cell
            bounds: RealBox,
            /// The level a cell lies on.
            level: usize,
            /// The parent of this cell (null if root).
            parent: usize,
            /// The index of the first child of this cell (null if leaf).
            children: usize,
            /// The neighbor cell along each face (null_index if coarser, boundary_index if a physical boundary).
            neighbors: [FaceIndex.count]usize,
        };

        // *******************************
        // Initialization

        /// Creates a new mesh on a rectangular domain.
        pub fn init(allocator: Allocator, bounds: RealBox) !Self {
            var cells: MultiArrayList(Cell) = .{};
            errdefer cells.deinit(allocator);

            try cells.append(allocator, .{
                .level = 0,
                .bounds = bounds,
                .parent = null_index,
                .children = null_index,
                .neighbors = [1]usize{boundary_index} ** FaceIndex.count,
            });

            var levels: RangeMap = .{};
            errdefer levels.deinit(allocator);

            try levels.append(allocator, 0);
            try levels.append(allocator, 1);

            return .{
                .gpa = allocator,
                .bounds = bounds,
                .cells = cells,
                .levels = levels,
            };
        }

        /// Deinitializes the mesh.
        pub fn deinit(self: *Self) void {
            self.cells.deinit(self.gpa);
            self.levels.deinit(self.gpa);
        }

        pub fn clone(self: *const Self, allocator: Allocator) !Self {
            var cells = try self.cells.clone(allocator);
            errdefer cells.deinit(allocator);

            var levels = try self.levels.clone(allocator);
            errdefer levels.deinit(allocator);

            return .{
                .gpa = allocator,
                .bounds = self.bounds,
                .cells = cells,
                .levels = levels,
            };
        }

        /// Returns the number of cells in the mesh
        pub fn numCells(self: *const Self) usize {
            return self.cells.len;
        }

        /// Returns the number of levels in the mesh
        pub fn numLevels(self: *const Self) usize {
            return self.levels.len();
        }

        /// Retrieves the physical bounds of the mesh
        pub fn physicalBounds(self: *const Self) RealBox {
            return self.bounds;
        }

        // **************************
        // Cells

        /// Checks whether a cell is a leaf.
        pub fn isLeaf(self: *const Self, cell: usize) bool {
            return self.cells.items(.children)[cell] == null_index;
        }

        /// Checks whether a cell is the root.
        pub fn isRoot(_: *const Self, cell: usize) bool {
            return cell == 0;
        }

        // ******************************
        // Flags

        /// Returns the number of cells which will be added to the mesh
        /// after performing refinement with the given flags.
        pub fn numNewCells(self: *const Self, flags: []const bool) usize {
            assert(flags.len == self.numCells());

            var result: usize = 0;

            for (0..self.numCells()) |cell| {
                // If cell is flagged and is a leaf, it counts
                if (flags[cell] and self.isLeaf(cell)) {
                    result += AxisMask.count;
                }
            }

            return result;
        }

        /// Returns the number of levels which will be added to the mesh
        /// after performing refinement with the given flags.
        pub fn numNewLevels(self: *const Self, flags: []const bool) usize {
            assert(flags.len == self.numCells());

            const finest = self.numLevels() - 1;
            const range = self.levels.range(finest);

            for (range.start..range.end) |cell| {
                if (flags[cell]) {
                    return 1;
                }
            }

            return 0;
        }

        /// Checks that the given refinement flags are smooth
        pub fn checkRefineFlags(self: *const Self, flags: []const bool) bool {
            assert(flags.len == self.numCells());

            const cells = self.cells.slice();

            for (1..cells.len) |cell| {
                const flag = flags[cell];
                const parent = cells.items(.parent)[cell];

                if (flag == false) {
                    continue;
                }

                // Get split index
                const split = AxisMask.fromLinear(cell - cells.items(.children)[parent]);
                // Iterate over a "split space".
                for (AxisMask.enumerate()[1..]) |dir| {
                    // Get neighboring node
                    var neighbor: usize = cell;
                    // Is the neighbor coarser than this node
                    var neighbor_coarse: bool = false;
                    // Loop over axes
                    for (split.outerFaces()) |face| {
                        if (dir.isSet(face.axis) == false) {
                            continue;
                        }
                        // Get neighbor, searching parent neighbors
                        var traverse = cells.items(.neighbors)[neighbor][face.toLinear()];

                        if (traverse == null_index) {
                            assert(!neighbor_coarse);
                            // Get parent of neighbor
                            const neighbor_parent = cells.items(.parent)[neighbor];
                            // We had to traverse a fine-coarse boundary
                            neighbor_coarse = true;
                            // We continue searching in case we eventually cross a boundary
                            traverse = cells.items(.neighbors)[neighbor_parent][face.toLinear()];
                        } else if (traverse == boundary_index) {
                            // No update needed if face is on boundary
                            neighbor_coarse = false;
                            neighbor = boundary_index;
                            break;
                        }
                        // Update node
                        neighbor = traverse;
                    }

                    // If neighbor is more coarse, it must already be tagged for refinement
                    if (neighbor_coarse and cells.items(.children)[neighbor] == null_index and !flags[neighbor]) {
                        return false;
                    }
                }
            }

            return true;
        }

        /// Smooths the given refinement flags such that there is never
        /// more than a 2:1 refinement difference across edges and corners.
        pub fn smoothRefineFlags(self: *const Self, flags: []bool) void {
            assert(flags.len == self.numCells());

            const cells = self.cells.slice();

            // Smooth flags
            while (true) {
                var is_smooth = true;

                for (1..cells.len) |cell| {
                    const flag = flags[cell];
                    const parent = cells.items(.parent)[cell];

                    if (flag == false or cells.items(.children)[cell] != null_index) {
                        continue;
                    }

                    // Get split index
                    const split = AxisMask.fromLinear(cell - cells.items(.children)[parent]);
                    // Iterate over a "split space".
                    for (AxisMask.enumerate()[1..]) |dir| {
                        // Get neighboring node
                        var neighbor: usize = cell;
                        // Is the neighbor coarser than this node
                        var neighbor_coarse: bool = false;
                        // Loop over axes
                        for (split.outerFaces()) |face| {
                            if (dir.isSet(face.axis) == false) {
                                continue;
                            }
                            // Get neighbor, searching parent neighbors
                            var traverse = cells.items(.neighbors)[neighbor][face.toLinear()];

                            if (traverse == null_index) {
                                assert(!neighbor_coarse);
                                // Get parent of neighbor
                                const neighbor_parent = cells.items(.parent)[neighbor];
                                // We had to traverse a fine-coarse boundary
                                neighbor_coarse = true;
                                // We continue searching in case we eventually cross a boundary
                                traverse = cells.items(.neighbors)[neighbor_parent][face.toLinear()];
                            } else if (traverse == boundary_index) {
                                // No update needed if face is on boundary
                                neighbor_coarse = false;
                                neighbor = boundary_index;
                                break;
                            }
                            // Update node
                            neighbor = traverse;
                        }

                        // If neighbor is more coarse, it must already be tagged for refinement
                        if (neighbor_coarse and cells.items(.children)[neighbor] == null_index and !flags[neighbor]) {
                            flags[neighbor] = true;
                            is_smooth = false;
                        }
                    }
                }

                if (is_smooth) {
                    break;
                }
            }

            assert(self.checkRefineFlags(flags));
        }

        /// Refines a mesh. The mesh is regenerated such that every leaf
        /// for which `flags[cell] = true` is split into hyperquadrants and
        /// added to the mesh.
        pub fn refine(self: *Self, allocator: Allocator, flags: []const bool) !void {
            assert(self.checkRefineFlags(flags));

            // ************************************
            // Regenerate mesh

            const num_cells = self.numNewCells(flags) + self.numCells();
            const num_levels = self.numNewLevels(flags) + self.numLevels();

            // ************************************
            // Compute new to old map

            const new_to_old: []usize = try allocator.alloc(usize, num_cells);
            defer allocator.free(new_to_old);

            var cursor: usize = 0;

            // Root -> Root
            new_to_old[cursor] = 0;
            cursor += 1;

            for (0..self.numCells()) |cell| {
                if (self.isLeaf(cell) and !flags[cell]) {
                    continue;
                }

                if (self.isLeaf(cell) and flags[cell]) {
                    // These are truly "new" cells, they have no older counterpart.
                    @memset(new_to_old[cursor .. cursor + AxisMask.count], null_index);
                    cursor += AxisMask.count;
                } else {
                    // This is not a leaf, so add children
                    const children = self.cells.items(.children)[cell];

                    for (0..AxisMask.count) |off| {
                        new_to_old[cursor + off] = children + off;
                    }

                    cursor += AxisMask.count;
                }
            }

            // Transfer all data to a scratch buffer
            var scratch = try self.cells.clone(allocator);
            defer scratch.deinit(allocator);
            // Cache pointers
            const cells = scratch.slice();

            // Clear current data
            self.cells.shrinkRetainingCapacity(0);
            try self.cells.ensureTotalCapacity(self.gpa, num_cells);

            self.levels.clear();

            // Add root
            self.cells.appendAssumeCapacity(.{
                .level = 0,
                .bounds = self.bounds,
                .parent = null_index,
                .children = null_index,
                .neighbors = [1]usize{boundary_index} ** FaceIndex.count,
            });

            try self.levels.append(allocator, 0);
            try self.levels.append(allocator, 1);

            for (0..num_levels - 1) |coarse| {
                const range = self.levels.range(coarse);
                const target = coarse + 1;

                for (range.start..range.end) |new_cell| {
                    const old_cell = new_to_old[new_cell];

                    if (old_cell == null_index) {
                        // This is a newly added cell, so it has no children
                        continue;
                    }

                    const old_flag: bool = flags[old_cell];
                    const old_children: usize = cells.items(.children)[old_cell];
                    const old_bounds: RealBox = cells.items(.bounds)[old_cell];

                    if (old_children == null_index and !old_flag) {
                        // The coarse cell is truly a leaf and hasn't been tagged for refinement
                        continue;
                    }

                    const new_children = self.cells.len;
                    // Update children field
                    self.cells.items(.children)[new_cell] = new_children;

                    for (AxisMask.enumerate()) |child| {
                        const bounds = old_bounds.split(child);

                        self.cells.appendAssumeCapacity(.{
                            .bounds = bounds,
                            .parent = new_cell,
                            .children = null_index,
                            .neighbors = [1]usize{null_index} ** FaceIndex.count,
                            .level = target,
                        });
                    }
                }

                // Continue building level -> cell map
                try self.levels.append(allocator, self.cells.len);
            }

            const new_cells = self.cells.slice();
            assert(new_cells.len == num_cells);

            // *****************************************
            // Compute neighbors

            const neighbors = new_cells.items(.neighbors);
            // Set root neighbors
            neighbors[0] = [1]usize{boundary_index} ** FaceIndex.count;

            for (0..new_cells.len) |cell| {
                const children = new_cells.items(.children)[cell];
                assert(children != boundary_index);

                if (children == null_index) {
                    continue;
                }

                // We loop over the children of the current cell
                for (AxisMask.enumerate()) |child| {
                    const child_id = children + child.toLinear();
                    // Inner faces are trivial to set
                    for (child.innerFaces()) |face| {
                        neighbors[child_id][face.toLinear()] = children + child.toggled(face.axis).toLinear();
                    }

                    // Outer faces are a bit more complicated
                    for (child.outerFaces()) |face| {
                        const cell_neighbor = neighbors[cell][face.toLinear()];
                        // If this cell has children, the neighbors must not be null if we have properly smoothed the flags

                        // if (cell_neighbor == null_index) {
                        //     std.debug.print("Cell ID {}\n", .{cell_id});
                        //     std.debug.print("Neighbors {any}\n", .{neighbors[cell_id]});
                        //     std.debug.print("Face {}\n", .{face});
                        //     std.debug.print("Children {}\n", .{children});
                        //     neighbors[child_id][face.toLinear()] = null_index;
                        //     continue;
                        // }

                        assert(cell_neighbor != null_index);

                        if (cell_neighbor == boundary_index) {
                            neighbors[child_id][face.toLinear()] = boundary_index;
                        } else {
                            const cell_neighbor_children = new_cells.items(.children)[cell_neighbor];

                            if (cell_neighbor_children == null_index) {
                                // assert(new_cells.items(.children)[child_id] == null_index);
                                neighbors[child_id][face.toLinear()] = null_index;
                            } else {
                                neighbors[child_id][face.toLinear()] = cell_neighbor_children + child.toggled(face.axis).toLinear();
                            }
                        }
                    }
                }
            }
        }

        /// Flags every cell of the mesh for refinement, then executes `refine`.
        pub fn refineGlobal(self: *@This(), allocator: Allocator) !void {
            const flags = try allocator.alloc(bool, self.numCells());
            defer allocator.free(flags);

            @memset(flags, true);

            try self.refine(allocator, flags);
        }
    };
}

/// Provides an additional block structure for quadtree based meshes.
/// This unifies uniform blocks of cells, so that storage and computation
/// is more efficient, and less duplicate cells are used.
pub fn BlockStructure(comptime N: usize) type {
    return struct {
        /// The set of blocks that make up the mesh.
        blocks: MultiArrayList(Block),
        /// Additional info stored per mesh cell.
        cells: MultiArrayList(Cell),
        /// Map from levels to block ranges.
        level_to_blocks: RangeMap,
        /// A cell permutation structure,
        cell_permute: CellPermutation,

        const AxisMask = geometry.AxisMask(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const RealBox = geometry.RealBox(N);
        const Region = geometry.Region(N);

        const CellPermutation = permute.CellPermutation(N);

        const Range = utils.Range;
        const RangeMap = utils.RangeMap;

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
            /// Parent of this block
            parent: usize = null_index,
            /// Position within parent
            index: [N]usize = [1]usize{0} ** N,
            /// Cell offset
            cells: usize,
        };

        /// Additional information stored for each cell of the mesh.
        pub const Cell = struct {
            /// Block that this cell belongs to.
            block: usize = null_index,
            /// Position within block of this cell.
            index: [N]usize = [1]usize{0} ** N,
            /// Cache of neighboring cells.
            neighbors: [Region.count]usize,
        };

        /// Builds a new block structure from a mesh.
        pub fn init(allocator: Allocator, mesh: *const Mesh(N), max_refinement: usize) !@This() {
            const cell_permute = try CellPermutation.init(allocator, max_refinement);
            errdefer cell_permute.deinit(allocator);

            assert(max_refinement > 0);

            var self = @This(){
                .blocks = .{},
                .cells = .{},
                .level_to_blocks = .{},
                .cell_permute = cell_permute,
            };
            errdefer self.deinit(allocator);

            try self.build(allocator, mesh);

            return self;
        }

        /// Frees the data associated with this block structure.
        pub fn deinit(self: *@This(), allocator: Allocator) void {
            self.blocks.deinit(allocator);
            self.cells.deinit(allocator);
            self.level_to_blocks.deinit(allocator);
            self.cell_permute.deinit(allocator);
        }

        /// Builds a new `NodeManager` from a mesh, using the given allocator for scratch allocations.
        pub fn build(self: *@This(), allocator: Allocator, mesh: *const Mesh(N)) !void {
            // ****************************
            // Computes Blocks + Offsets
            try self.buildBlocks(allocator, mesh);

            // *****************************
            // Compute neighbors
            try self.buildCells(allocator, mesh);

            // *****************************
            // Compute block cell maps
            self.buildBlockCellMaps(mesh);
        }

        /// Runs a recursive algorithm that traverses the tree from root to leaves, computing valid blocks while doing so.
        /// This algorithm works level by level (queueing up blocks for use on the next level). And thus preserves the invariant
        /// that all blocks on the same level are contiguous in the block array.
        fn buildBlocks(self: *@This(), allocator: Allocator, mesh: *const Mesh(N)) !void {
            // Cache
            const cells = mesh.cells.slice();
            const levels = mesh.numLevels();

            // Reset level offsets
            try self.level_to_blocks.resize(allocator, levels);

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
            var blocks: ArrayListUnmanaged(BlockMeta) = .{};
            defer blocks.deinit(allocator);
            // Stack for each level
            var stack: ArrayListUnmanaged(BlockMeta) = .{};
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
                    } else if (leaf_count == 0 and block.refinement + 1 < self.cell_permute.numLevels()) {
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
            self.blocks.shrinkRetainingCapacity(0);

            for (blocks.items) |meta| {
                // Find cell offset and cell total.
                var cell_offset: usize = meta.base;

                for (0..meta.refinement) |_| {
                    cell_offset = cells.items(.children)[cell_offset];
                }

                try self.blocks.append(allocator, .{
                    .bounds = cells.items(.bounds)[meta.base],
                    .size = cellSizeFromRefinement(meta.refinement),
                    .boundary = meta.boundary,
                    .refinement = meta.refinement,
                    .cells = cell_offset,
                });
            }
        }

        /// Builds extra per-cell information like neighbors.
        fn buildCells(self: *@This(), allocator: Allocator, mesh: *const Mesh(N)) !void {
            // Cache pointers
            const cells = mesh.cells.slice();

            // Reset and reserve
            try self.cells.resize(allocator, cells.len);
            // Set root
            self.cells.set(0, Cell{
                .neighbors = [1]usize{boundary_index} ** Region.count,
            });

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
                        neighbors[region.toLinear()] = null_index;
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
                        neighbors[region.toLinear()] = children + nsplit.toLinear();

                        continue;
                    }

                    const neighbor = self.cells.items(.neighbors)[parent][mregion.toLinear()];

                    if (neighbor == null_index) {
                        neighbors[region.toLinear()] = null_index;
                        continue;
                    }

                    assert(neighbor != null_index);

                    if (neighbor == boundary_index) {
                        neighbors[region.toLinear()] = boundary_index;
                    } else {
                        const nchildren = cells.items(.children)[neighbor];

                        if (nchildren == null_index) {
                            neighbors[region.toLinear()] = null_index;
                        } else {
                            neighbors[region.toLinear()] = nchildren + nsplit.toLinear();
                        }
                    }
                }

                // Set neighbors
                self.cells.items(.neighbors)[cell] = neighbors;
            }
        }

        /// Fills cell to block, and block to parent maps.
        fn buildBlockCellMaps(self: *@This(), mesh: *const Mesh(N)) void {
            // Set block and index of cells
            for (0..self.numBlocks()) |block| {
                var cell_indices = IndexSpace.fromSize(self.blocks.items(.size)[block]).cartesianIndices();
                while (cell_indices.next()) |index| {
                    const cell = self.blockCell(block, index);
                    self.cells.items(.block)[cell] = block;
                    self.cells.items(.index)[cell] = index;
                }
            }

            // Set block parents and indices
            for (0..self.numBlocks()) |block| {
                // Get cell in bottom left corner
                const cell = self.blocks.items(.cells)[block];
                // Get parent
                const parent = mesh.cells.items(.parent)[cell];

                if (parent == null_index) {
                    self.blocks.items(.parent)[block] = 0;
                    self.blocks.items(.index)[block] = [1]usize{0} ** N;
                } else {
                    self.blocks.items(.parent)[block] = self.cells.items(.block)[parent];
                    self.blocks.items(.index)[block] = self.cells.items(.index)[parent];
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

        // *****************************
        // Helpers *********************
        // *****************************

        pub fn numLevels(self: *const @This()) usize {
            return self.level_to_blocks.len();
        }

        pub fn levelBlocks(self: *const @This(), level_id: usize) Range {
            return self.level_to_blocks.range(level_id);
        }

        /// Number of blocks in the mesh.
        pub fn numBlocks(self: *const @This()) usize {
            return self.blocks.len;
        }

        /// Finds the index of the cell that lies at the given position in the block.
        pub fn blockCell(self: *const @This(), block_id: usize, index: [N]usize) usize {
            const size = self.blocks.items(.size)[block_id];
            const refinement = self.blocks.items(.refinement)[block_id];
            const cells = self.blocks.items(.cells)[block_id];

            const linear = IndexSpace.fromSize(size).linearFromCartesian(index);

            assert(refinement <= self.cell_permute.numLevels());

            return cells + self.cell_permute.permutation(refinement)[linear];
        }

        pub fn blockBounds(self: *const @This(), block_id: usize) RealBox {
            return self.blocks.items(.bounds)[block_id];
        }

        pub fn blockSize(self: *const @This(), block_id: usize) [N]usize {
            return self.blocks.items(.size)[block_id];
        }

        pub fn blockBoundary(self: *const @This(), block_id: usize) [FaceIndex.count]bool {
            return self.blocks.items(.boundary)[block_id];
        }

        pub fn blockParent(self: *const @This(), block_id: usize) usize {
            return self.blocks.items(.parent)[block_id];
        }

        pub fn blockParentIndex(self: *const @This(), block_id: usize) [N]usize {
            return self.blocks.items(.index)[block_id];
        }

        pub fn numCells(self: *const @This()) usize {
            return self.cells.len;
        }

        pub fn cellNeighbors(self: *const @This(), cell_id: usize) [Region.count]usize {
            return self.cells.items(.neighbors)[cell_id];
        }

        pub fn cellBlock(self: *const @This(), cell_id: usize) usize {
            return self.cells.items(.block)[cell_id];
        }

        pub fn cellBlockIndex(self: *const @This(), cell_id: usize) [N]usize {
            return self.cells.items(.index)[cell_id];
        }

        /// Checks whether or not the mesh datastructure is wellformed.
        pub fn isWellFormed(self: *const @This()) bool {
            for (0..self.cells.len) |cell_id| {
                const neighbors = self.cells.items(.neighbors)[cell_id];

                for (Region.enumerate()) |region| {
                    if (region.toLinear() == Region.central().toLinear()) {
                        continue;
                    }

                    const neighbor = neighbors[region.toLinear()];

                    if (neighbor != null_index and neighbor != boundary_index) {
                        // Check that this cell's neighbor's neighbor is itself.
                        if (cell_id != self.cells.items(.neighbors)[neighbor][region.reversed().toLinear()]) {
                            return false;
                        }
                    }
                }
            }

            return true;
        }
    };
}

test {
    _ = manager;
    _ = permute;
}
