const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;

const geometry = @import("../geometry/geometry.zig");
const numFaces = geometry.numFaces;
const numSplitIndices = geometry.numSplitIndices;

/// A null index.
pub const null_index: usize = std.math.maxInt(usize);
/// This relies on the fact that the root node can never be a neighbor.
pub const boundary_index: usize = 0;

/// A node in a quadtree.
pub fn Node(comptime N: usize) type {
    return struct {
        /// The physical bounds of a node
        bounds: RealBox,
        /// The level a node lies on
        level: usize,
        /// The parent of this node (null if root).
        parent: usize,
        /// The index of the first child of this node (null if leaf).
        children: usize,
        /// The neighbor node along each face (null if coarser).
        neighbors: [numFaces(N)]usize,
        /// A flag for whether this node if flagged for refinement or coarsening.
        flag: bool = false,

        const RealBox = geometry.RealBox(N);
    };
}

/// A quadtree based mesh.
pub fn TreeMesh(comptime N: usize) type {
    return struct {
        gpa: Allocator,
        /// The nodes that make up this mesh
        nodes: MultiArrayList(Node(N)),
        /// A set of offsets for each level of the mesh.
        offsets: ArrayListUnmanaged(usize),
        /// The number of cells along each edge of a node.
        node_width: usize,

        const Self = @This();
        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const RealBox = geometry.RealBox(N);
        const Region = geometry.Region(N);
        const SplitIndex = geometry.SplitIndex(N);

        pub fn init(allocator: Allocator, bounds: RealBox, node_width: usize) !Self {
            var nodes: MultiArrayList(Node(N)) = .{};
            errdefer nodes.deinit(allocator);

            try nodes.append(allocator, .{
                .level = 0,
                .bounds = bounds,
                .parent = null_index,
                .children = null_index,
                .neighbors = [1]usize{boundary_index} ** numFaces(N),
            });

            var offsets: ArrayListUnmanaged(usize) = .{};
            errdefer offsets.deinit(allocator);

            try offsets.append(allocator, 0);
            try offsets.append(allocator, 1);

            return .{
                .gpa = allocator,
                .nodes = nodes,
                .offsets = offsets,
                .node_width = node_width,
            };
        }

        pub fn deinit(self: *Self) void {
            self.offsets.deinit(self.gpa);
            self.nodes.deinit(self.gpa);
        }

        pub fn numLevels(self: *const Self) usize {
            return self.offsets.items.len - 1;
        }

        // Smooths refinement flags to ensure proper 2:1 interfaces.
        pub fn smoothRefineFlags(self: *Self) void {
            const nodes = self.nodes.slice();

            // Ensure only leaf nodes are tagged for refinement
            for (0..self.nodes.len) |idx| {
                if (nodes.items(.children)[idx] != null_index) {
                    nodes.items(.flag)[idx] = false;
                }
            }

            // Smooth flags
            var is_smooth = false;

            while (!is_smooth) {
                is_smooth = true;

                for (1..self.nodes.len) |node| {
                    // Is the current node flagged for refinement
                    const flag = nodes.items(.flag)[node];

                    if (!flag) {
                        continue;
                    }

                    const parent = nodes.items(.parent)[node];

                    // Get split index
                    const sidx = node - nodes.items(.children)[parent];
                    const split = SplitIndex.fromLinear(@intCast(sidx));
                    const split_sides = split.toCartesian();
                    // Iterate over a "split space".
                    for (SplitIndex.splitIndices()[1..]) |split_index| {
                        const cart = split_index.toCartesian();
                        // Neighboring node
                        var neighbor: usize = node;
                        // Is the nieghbor coarser than this node
                        var neighbor_coarse: bool = false;
                        // Loop over axes
                        for (0..N) |axis| {
                            // Skip traversing this axis if cart[axis] == 0
                            if (!cart[axis]) {
                                continue;
                            }
                            // Find face to traverse
                            const face = FaceIndex{
                                .side = split_sides[axis],
                                .axis = axis,
                            };
                            // Get neighbor, searching parent neighbors if necessary
                            var traverse = nodes.items(.neighbors)[node][face.toLinear()];

                            if (traverse == null_index) {
                                const neighbor_parent = nodes.items(.parent)[neighbor];
                                // We had to traverse a fine-coarse boundary
                                neighbor_coarse = true;
                                traverse = nodes.items(.neighbors)[neighbor_parent][face.toLinear()];
                            } else if (traverse == boundary_index) {
                                // No update needed if face is on boundary
                                neighbor_coarse = false;
                                continue;
                            }

                            // Update node
                            neighbor = traverse;
                        }
                        // If neighbor is more coarse, tag for refinement
                        if (neighbor_coarse and nodes.items(.children)[neighbor] == null_index) {
                            nodes.items(.flag)[neighbor] = true;
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
            var scratch = try self.nodes.clone(allocator);
            defer scratch.deinit(allocator);
            // Cache pointers
            const old_nodes = scratch.slice();
            // And erase current mesh data (other than root)
            self.nodes.shrinkRetainingCapacity(1);
            self.offsets.shrinkRetainingCapacity(1);
            // Set root children to null (updated later).
            self.nodes.items(.children)[0] = null_index;
            // Find number of nodes to be added
            var new_node_count: usize = 0;

            for (0..self.nodes.len) |idx| {
                if (old_nodes.items(.flag)[idx]) {
                    new_node_count += 1;
                }
            }
            // Ensure total capacity is such that we need not allocate in the loop
            try self.nodes.ensureTotalCapacity(self.gpa, old_nodes.len + new_node_count);

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
            try self.offsets.append(self.gpa, self.nodes.len);

            while (map.items.len > 0) : (target += 1) {
                // Add level offsets

                // Reset temporary map
                map_tmp.shrinkRetainingCapacity(0);
                // Loop over current level
                for (map.items) |m| {
                    // Add all children of this element to self.nodes
                    // and update map_tmp

                    const coarse_bounds: RealBox = old_nodes.items(.bounds)[m.old];
                    const coarse_children = old_nodes.items(.children)[m.old];
                    const coarse_flag = old_nodes.items(.flag)[m.old];
                    const coarse_neighbors = old_nodes.items(.neighbors)[m.old];

                    // Only continue if this node has pre-existing children or is tagged for refinement
                    if (coarse_children != null_index and !coarse_flag) {
                        continue;
                    }

                    // Starting index of new children
                    const new_children = self.nodes.len;
                    // Update current node's children index
                    self.nodes.items(.children)[m.new] = new_children;

                    // Add children to self.nodes
                    for (SplitIndex.splitIndices()) |child| {
                        var flag = false;
                        // If this node already had children, they need to be iterated, add to tmp map
                        if (coarse_children != null_index) {
                            const old_child_index = coarse_children + child.linear;
                            const new_child_index = new_children + child.linear;

                            try map_tmp.append(allocator, .{
                                .old = old_child_index,
                                .new = new_child_index,
                            });

                            flag = old_nodes.items(.flag)[old_child_index];
                        }

                        // Compute new bounds
                        const bounds = coarse_bounds.split(child);

                        // Compute new neighbors
                        var neighbors: [numFaces(N)]usize = undefined;

                        const sides = child.toCartesian();

                        for (FaceIndex.faces()) |face| {
                            neighbors[face.toLinear()] = if (sides[face.axis] != face.side)
                                // Inner face
                                new_children + child.reverseAxis(face.axis).linear
                            else if (coarse_neighbors[face.toLinear()] == boundary_index)
                                // Propogate boundary
                                boundary_index
                            else
                                // Decide in next step
                                null_index;
                        }

                        // Add new child node
                        self.nodes.appendAssumeCapacity(.{
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
                try self.offsets.append(self.gpa, self.nodes.len);
            }

            const new_nodes = self.nodes.slice();

            // Correct neighbors
            for (1..new_nodes.len) |node| {
                const children = new_nodes.items(.children)[node];
                const parent = new_nodes.items(.parent)[node];
                var neighbors = &new_nodes.items(.neighbors)[node];
                const flag = new_nodes.items(.flag)[node];

                // If this was not flagged for refinement
                if (!flag) {
                    continue;
                }

                const sidx = node - new_nodes.items(.children)[parent];
                const split = SplitIndex.fromLinear(sidx);

                for (FaceIndex.faces()) |face| {
                    // If this nodes is a leaf and neighbor is coarser check if it has children
                    if (neighbors[face.toLinear()] == null_index) {
                        // Neighbor on coarser level
                        const coarse_neighbor = new_nodes.items(.neighbors)[parent][face.toLinear()];
                        const coarse_neighbor_children = new_nodes.items(.children)[coarse_neighbor];
                        // We assume flags were properly smoothed
                        assert(coarse_neighbor_children != null_index);
                        // Find child on on opposite side of face
                        const other = split.reverseAxis(face.axis);
                        // Update neighbors
                        neighbors[face.toLinear()] = coarse_neighbor_children + other.linear;
                    }

                    // Neighbor had better not be 0
                    const neighbor = neighbors[face.axis];

                    if (neighbor == boundary_index) {
                        continue;
                    }

                    const neighbor_children = new_nodes.items(.children)[neighbor];

                    if (neighbor_children == null_index) {
                        // Neighbor is a leaf, so this node's children must have faces set to null
                        for (SplitIndex.splitIndices()) |child| {
                            if (child.toCartesian()[face.axis] == face.side) {
                                const child_node = children + child.linear;
                                new_nodes.items(.neighbors)[child_node][face.axis] = null_index;
                            }
                        }
                    } else {
                        // Neighbor has children so we update this node's children accordingly
                        for (SplitIndex.splitIndices()) |child| {
                            if (child.toCartesian()[face.axis] == face.side) {
                                const child_node = children + child.linear;
                                const other_node = neighbor_children + child.reverseAxis(face.axis).linear;
                                new_nodes.items(.neighbors)[child_node][face.axis] = other_node;
                            }
                        }
                    }
                }
            }
        }
    };
}

test "tree mesh global refinement" {
    const expect = std.testing.expect;
    const allocator = std.testing.allocator;

    var mesh = try TreeMesh(2).init(allocator, .{
        .origin = .{ 0.0, 0.0 },
        .size = .{ 1.0, 1.0 },
    }, 16);
    defer mesh.deinit();

    // Global refine
    for (0..1) |_| {
        @memset(mesh.nodes.items(.flag), true);
        try mesh.refine(allocator);
    }

    try expect(mesh.nodes.len == 5);

    // std.debug.print("{}\n", .{mesh});
}
