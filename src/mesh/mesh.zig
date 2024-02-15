const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;

const geometry = @import("../geometry/geometry.zig");
const utils = @import("../utils.zig");

const permute = @import("permute.zig");

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
        bounds: RealBox(N),
        /// The cells that make up this mesh
        cells: MultiArrayList(Cell(N)),
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
            var cells: MultiArrayList(Cell(N)) = .{};
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
            const scratch = try self.cells.clone(allocator);
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

test {
    _ = permute;
}
