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
pub const boundary_index: usize = 0;

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
        /// A flag for whether this cell if flagged for refinement or coarsening.
        flag: bool = false,

        const RealBox = geometry.RealBox(N);
    };
}

/// A quadtree based mesh.
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
        // const SplitIndex = geometry.SplitIndex(N);

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

        pub fn deinit(self: *Self) void {
            self.level_to_cells.deinit(self.gpa);
            self.cells.deinit(self.gpa);
        }

        pub fn numLevels(self: *const Self) usize {
            return self.level_to_cells.items.len - 1;
        }

        // Smooths refinement flags to ensure proper 2:1 interfaces.
        pub fn smoothRefineFlags(self: *Self) void {
            const cells = self.cells.slice();

            // Ensure only leaf nodes are tagged for refinement
            for (0..cells.len) |idx| {
                if (cells.items(.children)[idx] != null_index) {
                    cells.items(.flag)[idx] = false;
                }
            }

            // Smooth flags
            var is_smooth = false;

            while (!is_smooth) {
                is_smooth = true;

                for (1..cells.len) |cell| {
                    // Is the current node flagged for refinement
                    const flag = cells.items(.flag)[cell];

                    if (!flag) {
                        continue;
                    }

                    const parent = cells.items(.parent)[cell];

                    // Get split index
                    const split = AxisMask.fromLinear(cell - cells.items(.children)[parent]);
                    const split_unpacked = split.unpack();
                    // Iterate over a "split space".
                    for (AxisMask.enumerate()[1..]) |osplit| {
                        const osplit_unpacked = osplit.unpack();
                        // Neighboring node
                        var neighbor: usize = cell;
                        // Is the nieghbor coarser than this node
                        var neighbor_coarse: bool = false;
                        // Loop over axes
                        for (0..N) |axis| {
                            // Skip traversing this axis if cart[axis] == 0
                            if (!osplit_unpacked[axis]) {
                                continue;
                            }
                            // Find face to traverse
                            const face = FaceIndex{
                                .side = split_unpacked[axis],
                                .axis = axis,
                            };
                            // Get neighbor, searching parent neighbors if necessary
                            var traverse = cells.items(.neighbors)[cell][face.toLinear()];

                            if (traverse == null_index) {
                                const neighbor_parent = cells.items(.parent)[neighbor];
                                // We had to traverse a fine-coarse boundary
                                neighbor_coarse = true;
                                traverse = cells.items(.neighbors)[neighbor_parent][face.toLinear()];
                            } else if (traverse == boundary_index) {
                                // No update needed if face is on boundary
                                neighbor_coarse = false;
                                continue;
                            }

                            // Update node
                            neighbor = traverse;
                        }
                        // If neighbor is more coarse, tag for refinement
                        if (neighbor_coarse and cells.items(.children)[neighbor] == null_index) {
                            cells.items(.flag)[neighbor] = true;
                            is_smooth = false;
                        }
                    }
                }
            }
        }

        pub fn refine(self: *Self, allocator: Allocator) !void {
            // Perfom Smoothing
            self.smoothRefineFlags();
            // Transfer all mesh data to a scratch buffer
            var scratch = try self.cells.clone(allocator);
            defer scratch.deinit(allocator);
            // Cache pointers
            const old_cells = scratch.slice();
            // Find number of cells to be added
            var new_node_count: usize = 0;
            for (0..old_cells.len) |idx| {
                if (old_cells.items(.flag)[idx]) {
                    new_node_count += 1;
                }
            }
            // And erase current mesh data (other than root)
            self.cells.shrinkRetainingCapacity(1);
            self.level_to_cells.shrinkRetainingCapacity(1);
            // Set root children to null (updated later).
            self.cells.items(.children)[0] = null_index;
            // Ensure total capacity is such that we need not allocate in the loop
            try self.cells.ensureTotalCapacity(self.gpa, old_cells.len + new_node_count);

            // Perform refinement
            const IndexMap = struct { old: usize, new: usize };

            // Temporary map structures
            var map: ArrayListUnmanaged(IndexMap) = .{};
            defer map.deinit(allocator);

            try map.append(allocator, .{ .old = 0, .new = 0 });

            var map_tmp: ArrayListUnmanaged(IndexMap) = .{};
            defer map_tmp.deinit(allocator);

            var target: usize = 1;

            // Add offset of level 1
            try self.level_to_cells.append(self.gpa, self.cells.len);

            while (map.items.len > 0) : (target += 1) {
                // Reset temporary map
                map_tmp.shrinkRetainingCapacity(0);
                // Loop over current level
                for (map.items) |m| {
                    // Add all children of this element to self.nodes
                    // and update map_tmp

                    const coarse_bounds: RealBox = old_cells.items(.bounds)[m.old];
                    const coarse_children = old_cells.items(.children)[m.old];
                    const coarse_flag = old_cells.items(.flag)[m.old];
                    const coarse_neighbors = old_cells.items(.neighbors)[m.old];

                    // Only continue if this node has pre-existing children or is tagged for refinement
                    if (coarse_children != null_index and !coarse_flag) {
                        continue;
                    }

                    // Starting index of new children
                    const new_children = self.cells.len;
                    // Update current cell's children index
                    self.cells.items(.children)[m.new] = new_children;

                    // Add children to self.nodes
                    for (AxisMask.enumerate()) |child| {
                        var flag = false;
                        // If this node already had children, they need to be iterated, add to tmp map
                        if (coarse_children != null_index) {
                            const old_child_index = coarse_children + child.toLinear();
                            const new_child_index = new_children + child.toLinear();

                            try map_tmp.append(allocator, .{
                                .old = old_child_index,
                                .new = new_child_index,
                            });

                            flag = old_cells.items(.flag)[old_child_index];
                        }

                        // Compute new bounds
                        const bounds = coarse_bounds.split(child);

                        // Compute new neighbors
                        var neighbors: [FaceIndex.count]usize = undefined;

                        const sides = child.unpack();

                        for (FaceIndex.enumerate()) |face| {
                            neighbors[face.toLinear()] = if (sides[face.axis] != face.side)
                                // Inner face
                                new_children + child.toggled(face.axis).toLinear()
                            else if (coarse_neighbors[face.toLinear()] == boundary_index)
                                // Propogate boundary
                                boundary_index
                            else
                                // Decide in next step
                                null_index;
                        }

                        // Add new child node
                        self.cells.appendAssumeCapacity(.{
                            .bounds = bounds,
                            .parent = m.new,
                            .children = null_index,
                            .neighbors = neighbors,
                            .level = target,
                            .flag = flag,
                        });
                    }
                }

                // Swap map_tmp with map for next level
                std.mem.swap(ArrayListUnmanaged(IndexMap), &map, &map_tmp);
                // Update level offsets
                try self.level_to_cells.append(self.gpa, self.cells.len);
            }

            const new_cells = self.cells.slice();

            // Correct neighbors
            for (1..new_cells.len) |cell| {
                const children = new_cells.items(.children)[cell];
                const parent = new_cells.items(.parent)[cell];
                var neighbors = &new_cells.items(.neighbors)[cell];
                const flag = new_cells.items(.flag)[cell];

                // If this was not flagged for refinement
                if (!flag) {
                    continue;
                }

                const sidx = cell - new_cells.items(.children)[parent];
                const split = AxisMask.fromLinear(sidx);

                for (FaceIndex.enumerate()) |face| {
                    // If this nodes is a leaf and neighbor is coarser check if it has children
                    if (neighbors[face.toLinear()] == null_index) {
                        // Neighbor on coarser level
                        const coarse_neighbor = new_cells.items(.neighbors)[parent][face.toLinear()];
                        const coarse_neighbor_children = new_cells.items(.children)[coarse_neighbor];
                        // We assume flags were properly smoothed
                        assert(coarse_neighbor_children != null_index);
                        // Update neighbors
                        neighbors[face.toLinear()] = coarse_neighbor_children + split.toggled(face.axis).toLinear();
                    }

                    // Neighbor had better not be 0
                    const neighbor = neighbors[face.axis];

                    if (neighbor == boundary_index) {
                        continue;
                    }

                    const neighbor_children = new_cells.items(.children)[neighbor];

                    if (neighbor_children == null_index) {
                        // Neighbor is a leaf, so this node's children must have faces set to null
                        for (AxisMask.enumerate()) |child| {
                            if (child.isSet(face.axis) == face.side) {
                                const child_node = children + child.toLinear();
                                new_cells.items(.neighbors)[child_node][face.axis] = null_index;
                            }
                        }
                    } else {
                        // Neighbor has children so we update this node's children accordingly
                        for (AxisMask.enumerate()) |child| {
                            if (child.isSet(face.axis) == face.side) {
                                const child_node = children + child.toLinear();
                                const other_node = neighbor_children + child.toggled(face.axis).toLinear();
                                new_cells.items(.neighbors)[child_node][face.axis] = other_node;
                            }
                        }
                    }
                }
            }
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
        @memset(mesh.cells.items(.flag), true);
        try mesh.refine(allocator);
    }

    try expect(mesh.cells.len == 5);

    // std.debug.print("{}\n", .{mesh});
}
