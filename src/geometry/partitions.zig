const std = @import("std");
const heap = std.sort.heap;
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;

const Box = @import("box.zig").Box;
const IndexSpace = @import("index.zig").IndexSpace;

/// A set of tiles in some index space, grouped into several partitions.
/// Individual tiles may be "tagged" as active, and this object provides an
/// interface into grouping active tiles into partitions as well as specify "clusters" of
/// tiles which may not be bisected by these partitions (ie, there is a direct parent-child
/// relationship between a partition and a set of clusters).
pub fn PartitionSpace(comptime N: usize) type {
    return struct {
        gpa: Allocator,
        /// Size of the space to be partitioned.
        size: [N]usize,
        /// The set of partitions in the partitions list.
        parts: ArrayListUnmanaged(Partition),
        /// The set of clusters in the partition space.
        clusters: []const IndexBox,
        /// The parent partition for each cluster.
        parents: []usize,
        /// A buffer of cluster indices.
        children: []usize,

        const Self = @This();
        const IndexBox = Box(N, usize);

        pub const Partition = struct {
            /// The bounds of the partition in index space.
            bounds: Box(N, usize),
            /// The child clusters of the partition in index space.
            children_offset: usize,
            children_total: usize,
        };

        /// Builds a new partition space from an allocator, size and set of defined clusters.
        pub fn init(allocator: Allocator, size: [N]usize, clusters: []const IndexBox) !Self {
            var owned_clusters = try allocator.alloc(IndexBox, clusters.len);
            errdefer allocator.free(owned_clusters);

            @memcpy(owned_clusters, clusters);

            var parents = try allocator.alloc(usize, clusters.len);
            errdefer allocator.free(parents);

            @memset(parents, 0);

            var children = try allocator.alloc(usize, clusters.len);
            errdefer allocator.free(children);

            for (0..children.len) |i| {
                children[i] = i;
            }

            var parts: ArrayListUnmanaged(Partition) = .{};
            errdefer parts.deinit(allocator);

            try parts.append(allocator, .{
                .bounds = .{
                    .origin = [1]usize{0} ** N,
                    .size = size,
                },
                .children_offset = 0,
                .children_total = clusters.len,
            });

            return .{
                .gpa = allocator,
                .size = size,
                .parts = parts,
                .clusters = owned_clusters,
                .parents = parents,
                .children = children,
            };
        }

        // Clears all memory held by the partitions object.
        pub fn deinit(self: *Self) void {
            self.parts.deinit(self.gpa);
            self.gpa.free(self.clusters);
            self.gpa.free(self.parents);
            self.gpa.free(self.children);
        }

        /// Iterates over computed partitions in partition space.
        pub fn partitions(self: Self) []const Partition {
            return self.parts.items;
        }

        /// Computes the efficiency of a given partition.
        pub fn computeEfficiency(self: Self, subblock: IndexBox, tags: []const bool) f64 {
            const subspace = IndexSpace(N){ .size = subblock.size };
            const space = IndexSpace(N){ .size = self.size };

            const n_total = subspace.total();
            var n_tagged: usize = 0;
            var indices = subspace.cartesianIndices();

            while (indices.next()) |local| {
                var global: [N]usize = undefined;

                for (0..N) |i| {
                    global[i] = subblock.origin[i] + local[i];
                }

                const linear = space.linearFromCartesian(global);

                if (tags[linear]) {
                    n_tagged += 1;
                }
            }

            const f_tagged: f64 = @floatFromInt(n_tagged);
            const f_total: f64 = @floatFromInt(n_total);

            return f_tagged / f_total;
        }

        /// Runs Berger-Rigoutsos point clustering algorithm over the given set
        /// of tiles, while avoiding splitting prescribed "clusters" of tiles.
        /// These tiles are assumed to already be tagged.
        pub fn build(
            self: *Self,
            tags: []const bool,
            max_tiles: usize,
            efficiency: f64,
        ) !void {

            // 1. Reset partitions and higherarchy.
            // ************************************

            self.parts.shrinkRetainingCapacity(0);

            @memset(self.parents, 0);

            for (0..self.children.len) |i| {
                self.children[i] = i;
            }

            // 2. Allocate signatures and masks.
            // *********************************

            // Find the required buffer length
            var buffer_length: usize = 0;

            for (0..N) |i| {
                buffer_length += self.size[i];
            }

            // Allocate scratch data
            var signature_buffer: []usize = try self.gpa.alloc(usize, buffer_length);
            defer self.gpa.free(signature_buffer);

            var mask_buffer: []bool = try self.gpa.alloc(bool, buffer_length);
            defer self.gpa.free(mask_buffer);

            // Fill cluster buffer with initial data.
            for (0..self.clusters.len) |i| {
                self.children[i] = i;
            }

            // Build slices for each axis.
            var signatures: [N][]usize = undefined;
            var masks: [N][]bool = undefined;

            var buffer_offset: usize = 0;

            for (0..N) |i| {
                const size: usize = self.size[i];

                signatures[i] = signature_buffer[buffer_offset..(buffer_offset + size)];
                masks[i] = mask_buffer[buffer_offset..(buffer_offset + size)];

                buffer_offset += size;
            }

            // Allocate stack partitions.
            var stack = ArrayListUnmanaged(Partition){};
            defer stack.deinit(self.gpa);

            // Add first partition to stack
            try stack.append(
                self.gpa,
                .{
                    .bounds = .{
                        .origin = [1]usize{0} ** N,
                        .size = self.size,
                    },
                    .children_offset = 0,
                    .children_total = self.clusters.len,
                },
            );

            // Begin recusively dividing partitions.
            stack_pop: while (stack.popOrNull()) |full| {
                // Compute signatures
                self.computeSignatures(full, tags, signatures);
                self.computeMasks(full, masks);

                var lower_bounds: [N]usize = full.bounds.origin;
                var upper_bounds: [N]usize = undefined;

                for (0..N) |i| {
                    upper_bounds[i] = full.bounds.origin[i] + full.bounds.size[i];
                }

                // Cull any unnessessary spacing on either side.
                for (0..N) |axis| {
                    while (signatures[axis][lower_bounds[axis]] == 0 and !masks[axis][lower_bounds[axis]]) {
                        lower_bounds[axis] += 1;
                    }

                    while (signatures[axis][upper_bounds[axis] - 1] == 0 and !masks[axis][upper_bounds[axis] - 1]) {
                        upper_bounds[axis] -= 1;
                    }

                    if (lower_bounds[axis] >= upper_bounds[axis]) {
                        // Partition has an axis with zero width, so ignore it and move to next partition.
                        continue :stack_pop;
                    }
                }

                // Use computed bounds to build culled subblock
                var culled_subblock: IndexBox = undefined;

                for (0..N) |axis| {
                    culled_subblock.origin[axis] = lower_bounds[axis];
                    culled_subblock.size[axis] = upper_bounds[axis] - lower_bounds[axis];
                }

                const culled_space: IndexSpace(N) = .{ .size = culled_subblock.size };

                const partition: Partition = .{
                    .bounds = culled_subblock,
                    .children_offset = full.children_offset,
                    .children_total = full.children_total,
                };

                // Check efficiency and maximum block sidelength
                const cur_efficiency = self.computeEfficiency(culled_subblock, tags);
                const longest = culled_space.longestAxis();

                // If these checks pass, we can add this cell as a new block.
                if (cur_efficiency >= efficiency and culled_space.size[longest] <= max_tiles) {
                    try self.parts.append(self.gpa, partition);
                    continue :stack_pop;
                }

                // If they don't pass, we must partition the subblock into two
                // and add both to the stack.

                // First we check for holes
                var found_hole: bool = false;
                var hole_indices: [N]usize = [1]usize{0} ** N;

                for (0..N) |axis| {
                    for (lower_bounds[axis]..upper_bounds[axis]) |i| {
                        if (signatures[axis][i] == 0 and i > hole_indices[axis]) {
                            hole_indices[axis] = i;
                            found_hole = true;
                        }
                    }
                }

                if (found_hole) {
                    // Choose hole with largest index.
                    var hole_axis: usize = 0;
                    var hole_index: usize = 0;

                    for (0..N) |axis| {
                        if (hole_indices[axis] >= hole_index) {
                            hole_index = hole_indices[axis];
                            hole_axis = axis;
                        }
                    }

                    if (self.splitBoxes(masks, partition, lower_bounds, upper_bounds, hole_axis, hole_index)) |split| {
                        try stack.append(self.gpa, split.right);
                        try stack.append(self.gpa, split.left);
                    } else {
                        try self.parts.append(self.gpa, partition);
                    }

                    continue :stack_pop;
                }

                // If we find no holes, we now check for inflection points
                var found_inflection: bool = false;

                // We detect inflection points using the stencil -1 3 -3 1. This
                // is derived by applying the 2nd order laplacian 1 -2 1 to two
                // adjacent points and taking their difference
                var inflection_indices: [N]usize = [1]usize{0} ** N;
                var inflection_amounts: [N]isize = [1]isize{0} ** N;

                for (0..N) |axis| {
                    for ((lower_bounds[axis] + 2)..(upper_bounds[axis] - 1)) |i| {
                        const s_i: isize = @intCast(signatures[axis][i]);
                        const s_i_plus_one: isize = @intCast(signatures[axis][i + 1]);
                        const s_i_minus_one: isize = @intCast(signatures[axis][i - 1]);
                        const s_i_minus_two: isize = @intCast(signatures[axis][i - 2]);

                        const lap_i: isize = s_i_minus_one - 2 * s_i + s_i_plus_one;
                        const lap_i_minus_one: isize = s_i_minus_two - 2 * s_i_minus_one + s_i;

                        // If signs are the same continue.
                        if ((lap_i > 0) == (lap_i_minus_one > 0)) {
                            continue;
                        }

                        found_inflection = true;

                        var amount: isize = lap_i - lap_i_minus_one;
                        if (amount < 0) {
                            amount = -amount;
                        }

                        if (amount >= inflection_amounts[axis]) {
                            inflection_indices[axis] = i;
                        }
                    }
                }

                if (found_inflection) {
                    // Choose inflection with largest index
                    var inflection_axis: usize = 0;
                    var inflection_index: usize = 0;

                    for (0..N) |axis| {
                        if (inflection_indices[axis] >= inflection_index) {
                            inflection_index = inflection_indices[axis];
                            inflection_axis = axis;
                        }
                    }

                    if (self.splitBoxes(masks, partition, lower_bounds, upper_bounds, inflection_axis, inflection_index)) |split| {
                        try stack.append(self.gpa, split.right);
                        try stack.append(self.gpa, split.left);
                    } else {
                        try self.parts.append(self.gpa, partition);
                    }

                    continue :stack_pop;
                }

                // If we find no inflection points, we simply split
                // down the middle of the longest axis.
                const mid_axis = longest;
                const mid_index = (upper_bounds[mid_axis] + lower_bounds[mid_axis]) / 2;

                if (self.splitBoxes(masks, partition, lower_bounds, upper_bounds, mid_axis, mid_index)) |split| {
                    try stack.append(self.gpa, split.right);
                    try stack.append(self.gpa, split.left);
                } else {
                    try self.parts.append(self.gpa, partition);
                }
            }

            // Fill parents array.
            self.computeParents();
        }

        fn computeSignatures(self: *const Self, partition: Partition, tags: []const bool, signatures: [N][]usize) void {
            const subblock: IndexBox = partition.bounds;
            const space: IndexSpace(N) = .{ .size = self.size };
            const subspace: IndexSpace(N) = subblock.space();

            for (0..N) |axis| {
                @memset(signatures[axis][subblock.origin[axis]..(subblock.origin[axis] + subblock.size[axis])], 0);

                for (0..subblock.size[axis]) |i| {
                    var sig: usize = 0;

                    var iterator = subspace.cartesianSliceIndices(axis, i);

                    while (iterator.next()) |local| {
                        var global: [N]usize = undefined;

                        for (0..N) |j| {
                            global[j] = subblock.origin[j] + local[j];
                        }

                        const linear = space.linearFromCartesian(global);

                        if (tags[linear]) {
                            sig += 1;
                        }
                    }

                    signatures[axis][subblock.origin[axis] + i] += sig;
                }
            }
        }

        fn computeMasks(self: *const Self, partition: Partition, masks: [N][]bool) void {
            const subblock = partition.bounds;

            for (0..N) |axis| {
                @memset(masks[axis][subblock.origin[axis]..(subblock.origin[axis] + subblock.size[axis])], false);

                const offset: usize = partition.children_offset;
                const total: usize = partition.children_total;

                for (self.children[offset..(offset + total)]) |child| {
                    const bounds = self.clusters[child];

                    @memset(masks[axis][bounds.origin[axis]..(bounds.origin[axis] + bounds.size[axis])], true);
                }
            }
        }

        fn computeParents(self: *Self) void {
            for (self.parts.items, 0..) |part, i| {
                for (self.children[part.children_offset..(part.children_offset + part.children_total)]) |child| {
                    self.parents[child] = i;
                }
            }
        }

        const SplitContext = struct {
            clusters: []const IndexBox,
            axis: usize,
        };

        fn splitLessThanFn(context: SplitContext, lhs: usize, rhs: usize) bool {
            return context.clusters[lhs].origin[context.axis] < context.clusters[rhs].origin[context.axis];
        }

        fn splitBoxes(
            self: *const Self,
            masks: [N][]bool,
            parent: Partition,
            lower: [N]usize,
            upper: [N]usize,
            axis: usize,
            index: usize,
        ) ?struct { left: Partition, right: Partition } {
            assert(index > 0);

            // Sort children
            const context: SplitContext = .{
                .clusters = self.clusters,
                .axis = axis,
            };

            // Use heap sort.
            heap(usize, self.children[parent.children_offset..(parent.children_total + parent.children_offset)], context, splitLessThanFn);

            // Find adjusted index which does not bisect cluster.
            var index_inc: usize = index;
            var index_dec: usize = index;

            var split_index: usize = undefined;

            while (true) {
                if (index_inc < upper[axis]) {
                    if (masks[axis][index_inc] and masks[axis][index_inc - 1]) {
                        index_inc += 1;
                    } else {
                        split_index = index_inc;
                        break;
                    }
                } else if (index_dec > 0) {
                    if (masks[axis][index_dec] and masks[axis][index_dec - 1]) {
                        index_dec -= 1;
                    } else {
                        split_index = index_dec;
                        break;
                    }
                } else {
                    // We can not split the current partition in a way that
                    // doesn't bisect a cluster, so accept it.
                    return null;
                }
            }

            // Find index to split children clusters
            var clusters_index = parent.children_total;
            // Perform search in reverse as we favour higher indices.
            // Worst case: O(n).
            while (clusters_index > 0 and self.clusters[self.children[parent.children_offset + clusters_index - 1]].origin[axis] >= split_index) {
                clusters_index -= 1;
            }

            // Generate left and right partitions
            var left: Partition = undefined;
            var right: Partition = undefined;

            for (0..N) |i| {
                left.bounds.origin[i] = lower[i];
                left.bounds.size[i] = upper[i] - lower[i];
                right.bounds.origin[i] = lower[i];
                right.bounds.size[i] = upper[i] - lower[i];
            }

            left.bounds.origin[axis] = lower[axis];
            left.bounds.size[axis] = index - lower[axis];
            right.bounds.origin[axis] = index;
            right.bounds.size[axis] = upper[axis] - index;

            left.children_offset = parent.children_offset;
            left.children_total = clusters_index;
            right.children_offset = parent.children_offset + clusters_index;
            right.children_total = parent.children_total - clusters_index;

            return .{
                .left = left,
                .right = right,
            };
        }
    };
}

fn buildTags(comptime N: usize, allocator: Allocator, size: [N]usize, tagged: []const [N]usize) ![]bool {
    const space: IndexSpace(N) = .{ .size = size };
    const total = space.total();

    var tags: []bool = try allocator.alloc(bool, total);
    errdefer allocator.free(tags);

    @memset(tags, false);

    for (tagged) |t| {
        tags[space.linearFromCartesian(t)] = true;
    }

    return tags;
}

test "tiles basic" {
    const expect = std.testing.expect;
    const expectEqualSlices = std.testing.expectEqualSlices;
    const allocator = std.testing.allocator;

    const size = [3]usize{ 5, 5, 5 };

    const tagged = [_][3]usize{
        [_]usize{ 0, 0, 0 },
        [_]usize{ 3, 2, 1 },
        [_]usize{ 0, 1, 3 },
    };

    const tags: []bool = try buildTags(3, allocator, size, &tagged);
    defer allocator.free(tags);

    var partition_space = try PartitionSpace(3).init(allocator, size, &[_]Box(3, usize){});
    defer partition_space.deinit();

    const buffer_length: usize = size[0] + size[1] + size[2];
    var signature_buffer: []usize = try std.testing.allocator.alloc(usize, buffer_length);
    defer std.testing.allocator.free(signature_buffer);

    var signatures: [3][]usize = undefined;

    var buffer_offset: usize = 0;
    for (0..3) |i| {
        signatures[i] = signature_buffer[buffer_offset..(buffer_offset + size[i])];
        buffer_offset += size[i];
    }

    partition_space.computeSignatures(.{
        .bounds = .{
            .origin = [3]usize{ 0, 0, 0 },
            .size = [3]usize{ 5, 5, 5 },
        },
        .children_offset = 0,
        .children_total = 0,
    }, tags, signatures);

    try expectEqualSlices(usize, &[5]usize{ 2, 0, 0, 1, 0 }, signatures[0]);
    try expectEqualSlices(usize, &[5]usize{ 1, 1, 1, 0, 0 }, signatures[1]);
    try expectEqualSlices(usize, &[5]usize{ 1, 1, 0, 1, 0 }, signatures[2]);

    const efficiency = partition_space.computeEfficiency(.{
        .origin = [3]usize{ 0, 0, 0 },
        .size = [3]usize{ 4, 4, 4 },
    }, tags);

    try expect(efficiency == 3.0 / 64.0);
}

test "tile partitioning" {
    const expect = std.testing.expect;
    const expectEqualSlices = std.testing.expectEqualSlices;
    const allocator = std.testing.allocator;

    const size = [2]usize{ 5, 5 };

    const tagged = [_][2]usize{
        [_]usize{ 2, 2 },
        [_]usize{ 3, 2 },
        [_]usize{ 2, 3 },
        [_]usize{ 3, 3 },
    };

    const tags: []bool = try buildTags(2, allocator, size, &tagged);
    defer allocator.free(tags);

    var partition_space = try PartitionSpace(2).init(allocator, size, &[_]Box(2, usize){});
    defer partition_space.deinit();

    try partition_space.build(tags, 4, 0.7);

    const partitions: []const PartitionSpace(2).Partition = partition_space.partitions();

    try expect(partitions.len == 1);
    try expectEqualSlices(usize, &[2]usize{ 2, 2 }, &partitions[0].bounds.origin);
    try expectEqualSlices(usize, &[2]usize{ 2, 2 }, &partitions[0].bounds.size);
}
