const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;

const geometry = @import("../geometry/geometry.zig");

const manager = @import("manager.zig");
const multigrid = @import("multigrid.zig");
const permute = @import("permute.zig");
const worker = @import("worker.zig");

pub const MultigridMethod = multigrid.MultigridMethod;
pub const NodeManager = manager.NodeManager;
pub const NodeWorker = worker.NodeWorker;

/// A null index.
pub const null_index: usize = std.math.maxInt(usize);
/// This relies on the fact that the root node can never be a neighbor.
pub const boundary_index: usize = std.math.maxInt(usize) - 1;

/// A cell in a quadtree.
pub fn Cell(comptime N: usize) type {
    const num_faces = geometry.FaceIndex(N).count;

    return struct {
        /// The physical bounds of a cell
        bounds: RealBox,
        /// The level a cell lies on.
        level: usize,
        /// The parent of this cell (null if root).
        parent: usize,
        /// The index of the first child of this cell (null if leaf).
        children: usize,
        /// The neighbor cell along each face (null_index if coarser, boundary_index if a physical boundary).
        neighbors: [num_faces]usize,

        const RealBox = geometry.RealBox(N);
    };
}

/// A quadtree based mesh. The domain must be rectangular.
pub fn TreeMesh(comptime N: usize) type {
    return struct {
        gpa: Allocator,
        /// The cells that make up this mesh
        cells: MultiArrayList(Cell(N)),
        /// A set of offsets for each level of the mesh.
        level_to_cells: ArrayListUnmanaged(usize),

        const Self = @This();
        const AxisMask = geometry.AxisMask(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const RealBox = geometry.RealBox(N);
        const Region = geometry.Region(N);

        /// Creates a new tree mesh of the given bounds.
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

            var level_to_cells: ArrayListUnmanaged(usize) = .{};
            errdefer level_to_cells.deinit(allocator);

            try level_to_cells.append(allocator, 0);
            try level_to_cells.append(allocator, 1);

            return .{
                .gpa = allocator,
                .cells = cells,
                .level_to_cells = level_to_cells,
            };
        }

        /// Deinitializes the mesh.
        pub fn deinit(self: *Self) void {
            self.level_to_cells.deinit(self.gpa);
            self.cells.deinit(self.gpa);
        }

        /// Returns the number of cells.
        pub fn numCells(self: *const Self) usize {
            return self.cells.len;
        }

        /// Returns the number of levels on the mesh.
        pub fn numLevels(self: *const Self) usize {
            return self.level_to_cells.items.len - 1;
        }

        /// Retrieves the physical bounds of the
        pub fn physicalBounds(self: *const Self) RealBox {
            return self.cells.items(.bounds)[0];
        }

        // Smooths refinement flags to ensure proper 2:1 interfaces.
        pub fn smoothRefineFlags(self: *const Self, flags: []bool) void {
            assert(flags.len == self.cells.len);

            const cells = self.cells.slice();

            // Ensure only leaf nodes are tagged for refinement
            for (0..cells.len) |idx| {
                if (cells.items(.children)[idx] != null_index) {
                    flags[idx] = false;
                }
            }

            // Smooth flags
            while (true) {
                var is_smooth = true;

                for (1..cells.len) |cell| {
                    // Is the current node flagged for refinement
                    const flag = flags[cell];

                    if (flag == false or cells.items(.children)[cell] != null_index) {
                        continue;
                    }

                    const parent = cells.items(.parent)[cell];

                    // Get split index
                    const split = AxisMask.fromLinear(cell - cells.items(.children)[parent]);
                    // Iterate over a "split space".
                    for (AxisMask.enumerate()[1..]) |direction| {
                        // Neighboring node
                        var neighbor: usize = cell;
                        // Is the nieghbor coarser than this node
                        var neighbor_coarse: bool = false;
                        // Loop over axes
                        for (split.outerFaces()) |face| {
                            // Skip traversing this axis if cart[axis] == 0
                            if (direction.isSet(face.axis) == false) {
                                continue;
                            }
                            // Get neighbor, searching parent neighbors if necessary
                            var traverse = cells.items(.neighbors)[neighbor][face.toLinear()];

                            if (traverse == null_index) {
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

                        // If neighbor is more coarse, tag for refinement
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
        }

        pub fn refine(self: *Self, allocator: Allocator, flags: []bool) !void {
            // Perfom Smoothing
            self.smoothRefineFlags(flags);

            // *************************************************
            // Regenerate quadtree structure

            // Transfer all mesh data to a scratch buffer
            var scratch = try self.cells.clone(allocator);
            defer scratch.deinit(allocator);
            // Cache pointers
            const old_cells = scratch.slice();

            // Find number of cells to be added
            var new_node_count: usize = 0;
            for (0..old_cells.len) |idx| {
                // If cell is flagged and is a leaf
                if (flags[idx] and old_cells.items(.children)[idx] == null_index) {
                    new_node_count += AxisMask.count;
                }
            }
            // And erase current mesh data (other than root)
            self.cells.shrinkRetainingCapacity(1);
            self.level_to_cells.shrinkRetainingCapacity(2);
            // Check level offsets
            assert(self.level_to_cells.items[0] == 0);
            assert(self.level_to_cells.items[1] == 1);

            // Set root children to null (updated later).
            self.cells.items(.children)[0] = null_index;
            // Ensure total capacity is such that we need not allocate in the loop
            try self.cells.ensureTotalCapacity(self.gpa, old_cells.len + new_node_count);

            // Perform refinement
            const IndexMap = struct { old: usize, new: usize };

            // Temporary map structures
            var map: ArrayListUnmanaged(IndexMap) = .{};
            defer map.deinit(allocator);
            // Add root map
            try map.append(allocator, .{ .old = 0, .new = 0 });

            var map_tmp: ArrayListUnmanaged(IndexMap) = .{};
            defer map_tmp.deinit(allocator);

            var target: usize = 1;

            while (map.items.len > 0) : (target += 1) {
                // Reset temporary map
                map_tmp.shrinkRetainingCapacity(0);
                // Loop over current level
                for (map.items) |m| {
                    // Add all children of this element to self.nodes
                    // and update map_tmp
                    const coarse_bounds: RealBox = old_cells.items(.bounds)[m.old];
                    const coarse_children = old_cells.items(.children)[m.old];
                    const coarse_flag = flags[m.old];

                    // Only continue processing if this node has pre-existing children or is tagged for refinement
                    if (coarse_children == null_index and coarse_flag == false) {
                        self.cells.items(.children)[m.new] = null_index;
                        continue;
                    }

                    // Starting index of new children
                    const new_children = self.cells.len;
                    // Update current cell's children index
                    self.cells.items(.children)[m.new] = new_children;

                    // Add children to self.nodes
                    for (AxisMask.enumerate()) |child| {
                        // If this node already had children, they need to be iterated, add to tmp map
                        if (coarse_children != null_index) {
                            const old_child_index = coarse_children + child.toLinear();
                            const new_child_index = new_children + child.toLinear();

                            try map_tmp.append(allocator, .{
                                .old = old_child_index,
                                .new = new_child_index,
                            });
                        }

                        // Compute new bounds
                        const bounds = coarse_bounds.split(child);

                        // Add new child node
                        self.cells.appendAssumeCapacity(.{
                            .bounds = bounds,
                            .parent = m.new,
                            .children = null_index,
                            .neighbors = [1]usize{null_index} ** FaceIndex.count,
                            .level = target,
                        });
                    }
                }

                // Swap map_tmp with map for next level
                std.mem.swap(ArrayListUnmanaged(IndexMap), &map, &map_tmp);
                // Update level offsets
                try self.level_to_cells.append(self.gpa, self.cells.len);
            }

            // Remove last levels if it contains no cells
            while (true) {
                const last = self.level_to_cells.items.len - 1;

                if (self.level_to_cells.items[last] == self.level_to_cells.items[last - 1]) {
                    _ = self.level_to_cells.pop();
                } else {
                    break;
                }
            }

            const new_cells = self.cells.slice();

            assert(new_cells.len > 0);

            // ********************************
            // Compute neighbors

            const neighbors = new_cells.items(.neighbors);

            // Root should have only boundary neighbors
            neighbors[0] = [1]usize{boundary_index} ** FaceIndex.count;

            for (0..new_cells.len) |cell_id| {
                const children = new_cells.items(.children)[cell_id];
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
                        const cell_neighbor = neighbors[cell_id][face.toLinear()];
                        // If this cell has children, the neighbors must not be null if we have properly smoothed the flags

                        if (cell_neighbor == null_index) {
                            std.debug.print("Cell ID {}\n", .{cell_id});
                            std.debug.print("Neighbors {any}\n", .{neighbors[cell_id]});
                            std.debug.print("Face {}\n", .{face});
                            std.debug.print("Children {}\n", .{children});
                            neighbors[child_id][face.toLinear()] = null_index;
                            continue;
                        }

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

            // // Set neighbors
            // for (1..new_cells.len) |cell| {
            //     const parent = new_cells.items(.parent)[cell];
            //     var neighbors = &new_cells.items(.neighbors)[cell];

            //     const sidx = cell - new_cells.items(.children)[parent];
            //     const split = AxisMask.fromLinear(sidx);

            //     for (FaceIndex.enumerate()) |face| {
            //         // If neighbor is coarser check if it has children (so we can update self)
            //         if (neighbors[face.toLinear()] == null_index) {
            //             // Neighbor on coarser level
            //             const coarse_neighbor = new_cells.items(.neighbors)[parent][face.toLinear()];

            //             assert(coarse_neighbor != null_index);

            //             const coarse_neighbor_children = new_cells.items(.children)[coarse_neighbor];

            //             if (coarse_neighbor_children != null_index) {
            //                 // Update neighbors
            //                 neighbors[face.toLinear()] = coarse_neighbor_children + split.toggled(face.axis).toLinear();
            //                 continue;
            //             }
            //         }
            //     }
            // }
        }

        /// Refines the mesh globally.
        pub fn refineGlobal(self: *@This(), allocator: Allocator) !void {
            const flags = try allocator.alloc(bool, self.numCells());
            defer allocator.free(flags);

            @memset(flags, true);
            self.smoothRefineFlags(flags);

            try self.refine(allocator, flags);
        }

        /// Checks whether or not the mesh datastructure is wellformed.
        pub fn isWellFormed(self: *const @This()) bool {
            // Check levels to cells
            if (self.level_to_cells.getLast() != self.cells.len) {
                return false;
            }

            for (0..self.numLevels()) |level| {
                const start = self.level_to_cells.items[level];
                const end = self.level_to_cells.items[level + 1];

                for (start..end) |cell_id| {
                    if (self.cells.items(.level)[cell_id] != level) {
                        return false;
                    }
                }
            }

            // Check Parents/Children
            for (0..self.numCells()) |cell_id| {
                const parent = self.cells.items(.parent)[cell_id];
                const children = self.cells.items(.children)[cell_id];

                if (parent != null_index) {
                    if (self.cells.items(.children)[parent] == null_index) {
                        return false;
                    }
                    if (cell_id < self.cells.items(.children)[parent] or cell_id >= self.cells.items(.children)[parent] + AxisMask.count) {
                        return false;
                    }
                }

                if (children != null_index) {
                    for (AxisMask.enumerate()) |split| {
                        const child_id = children + split.toLinear();

                        if (self.cells.items(.parent)[child_id] != cell_id) {
                            return false;
                        }
                    }
                }
            }

            // Check neighbors
            for (0..self.numCells()) |cell_id| {
                const children = self.cells.items(.children)[cell_id];

                if (children != null_index) {
                    for (AxisMask.enumerate()) |split| {
                        const children_id = children + split.toLinear();

                        for (split.innerFaces()) |face| {
                            const child_neighbor = self.cells.items(.neighbors)[children_id][face.toLinear()];

                            // Check if inner face is boundary or null
                            if (child_neighbor == boundary_index or child_neighbor == null_index) {
                                return false;
                            }

                            const tsplit = split.toggled(face.axis);
                            if (children + tsplit.toLinear() != child_neighbor) {
                                return false;
                            }
                        }

                        for (split.outerFaces()) |face| {
                            const neighbor = self.cells.items(.neighbors)[cell_id][face.toLinear()];
                            const child_neighbor = self.cells.items(.neighbors)[children_id][face.toLinear()];

                            if ((neighbor == boundary_index) != (child_neighbor == boundary_index)) {
                                return false;
                            }
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
    _ = multigrid;
    _ = permute;
    _ = worker;
}

test "tree mesh global refinement" {
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;

    var mesh = try TreeMesh(2).init(allocator, .{
        .origin = .{ 0.0, 0.0 },
        .size = .{ 1.0, 1.0 },
    });
    defer mesh.deinit();

    // Global refine
    for (0..1) |_| {
        try mesh.refineGlobal(allocator);
    }

    for (0..3) |_| {
        const flags = try allocator.alloc(bool, mesh.numCells());
        defer allocator.free(flags);

        for (mesh.cells.items(.bounds), flags) |bounds, *flag| {
            if (bounds.origin[0] == 0.0 and bounds.origin[1] == 0.0) {
                flag.* = true;
            }
        }

        try mesh.refine(allocator, flags);
    }

    // try expect(mesh.numLevels() == 5);
    try expect(mesh.isWellFormed());
}
