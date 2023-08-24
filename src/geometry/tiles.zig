const std = @import("std");
const heap = std.sort.heap;
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;

const Box = @import("box.zig").Box;
const IndexSpace = @import("index.zig").IndexSpace;

/// A set of tiles in some index space. Individual tiles may be "tagged" as active,
/// and one may use the `Partitions` object to generate blocks that surrond these
/// active tiles, as well as specify "clusters" of tiles which may not be bisected
/// by these blocks (ie, there is a direct parent-child relationship between a block
/// and a set of clusters).
pub fn Tiles(comptime N: usize) type {
    return struct {
        size: [N]usize,
        tags: []const bool,
        clusters: []const IndexBox,

        const Self = @This();
        const IndexBox = Box(N, usize);

        /// Computes the "efficiency" of a certain partition, ie, the ratio
        /// of active to total tiles covered by the partition.
        pub fn computeEfficiency(self: Self, subblock: IndexBox) f64 {
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

                if (self.tags[linear]) {
                    n_tagged += 1;
                }
            }

            const f_tagged: f64 = @floatFromInt(n_tagged);
            const f_total: f64 = @floatFromInt(n_total);

            return f_tagged / f_total;
        }

        // /// Tags all tiles that lie in a cluster.
        // pub fn tagClusters(self: *Self) void {
        //     const space: IndexSpace(N) = .{ .size = self.size };

        //     for (self.clusters) |cluster| {
        //         const cluster_space: IndexSpace(N) = .{ .size = cluster.widths };
        //         var indices = space.cartesianIndices();

        //         while (indices.next()) |local| {
        //             var global: [N]usize = undefined;

        //             for (0..N) |axis| {
        //                 global[axis] = cluster.origin[axis] + local[axis];
        //             }

        //             const linear: usize = cluster_space.linearFromCartesian(global);
        //             self.clusters[linear] = true;
        //         }
        //     }
        // }
    };
}

/// A single partition of some index space.
pub fn Partition(comptime N: usize) type {
    return struct {
        /// The bounds of the partition in index space.
        bounds: Box(N, usize),
        /// The child clusters of the partition in index space.
        children_offset: usize,
        children_count: usize,
    };
}

/// Represents a set of partitions of a tiles object.
pub fn Partitions(comptime N: usize) type {
    return struct {
        /// The set of partitions in the partitions list.
        blocks: MultiArrayList(Partition(N)) = MultiArrayList(Partition(N)){},
        /// The parent partition for each cluster.
        parents: ArrayListUnmanaged(usize) = ArrayListUnmanaged(usize){},
        /// A buffer of cluster indices.
        children: ArrayListUnmanaged(usize) = ArrayListUnmanaged(usize){},

        const Self = @This();
        const IndexBox = Box(N, usize);

        // Clears all memory held by the partitions object.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.blocks.deinit(allocator);
            self.parents.deinit(allocator);
            self.children.deinit(allocator);
        }

        /// Runs Berger-Rigoutsos point clustering algorithm over the given set
        /// of tiles, while avoiding splitting prescribed "clusters" of tiles.
        /// These tiles are assumed to already be tagged.
        pub fn compute(
            self: *Self,
            allocator: Allocator,
            tiles: Tiles(N),
            max_tiles: usize,
            efficiency: f64,
        ) !void {
            self.blocks.shrinkRetainingCapacity(0);
            try self.parents.resize(allocator, tiles.clusters.len);
            try self.children.resize(allocator, tiles.clusters.len);

            // Find the required buffer length
            var buffer_length: usize = 0;

            for (0..N) |i| {
                buffer_length += tiles.size[i];
            }

            // Allocate scratch data
            var signature_buffer: []usize = try allocator.alloc(usize, buffer_length);
            defer allocator.free(signature_buffer);

            var mask_buffer: []bool = try allocator.alloc(bool, buffer_length);
            defer allocator.free(mask_buffer);

            // Fill cluster buffer with initial data.
            for (0..tiles.clusters.len) |i| {
                self.children.items[i] = i;
            }

            // Build slices for each axis.
            var signatures: [N][]usize = undefined;
            var masks: [N][]bool = undefined;

            var buffer_offset: usize = 0;

            for (0..N) |i| {
                const size: usize = tiles.size[i];

                signatures[i] = signature_buffer[buffer_offset..(buffer_offset + size)];
                masks[i] = mask_buffer[buffer_offset..(buffer_offset + size)];

                buffer_offset += size;
            }

            // Allocate stack partitions.
            var stack = ArrayListUnmanaged(Partition(N)){};
            defer stack.deinit(allocator);

            // Add first partition to stack
            try stack.append(
                allocator,
                .{
                    .bounds = .{
                        .origin = [1]usize{0} ** N,
                        .size = tiles.size,
                    },
                    .children_offset = 0,
                    .children_count = tiles.clusters.len,
                },
            );

            // Begin recusively dividing partitions.
            stack_pop: while (stack.popOrNull()) |full| {
                // Compute signatures
                self.computeSignatures(signatures, tiles, full);
                self.computeMasks(masks, tiles, full);

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

                const partition: Partition(N) = .{
                    .bounds = culled_subblock,
                    .children_offset = full.children_offset,
                    .children_count = full.children_count,
                };

                // Check efficiency and maximum block sidelength
                const cur_efficiency = tiles.computeEfficiency(culled_subblock);
                const longest = culled_space.longestAxis();

                // If these checks pass, we can add this cell as a new block.
                if (cur_efficiency >= efficiency and culled_space.size[longest] <= max_tiles) {
                    try self.blocks.append(allocator, partition);
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

                    if (self.splitBoxes(masks, tiles, partition, lower_bounds, upper_bounds, hole_axis, hole_index)) |split| {
                        try stack.append(allocator, split.right);
                        try stack.append(allocator, split.left);
                    } else {
                        try self.blocks.append(allocator, partition);
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

                    if (self.splitBoxes(masks, tiles, partition, lower_bounds, upper_bounds, inflection_axis, inflection_index)) |split| {
                        try stack.append(allocator, split.right);
                        try stack.append(allocator, split.left);
                    } else {
                        try self.blocks.append(allocator, partition);
                    }

                    continue :stack_pop;
                }

                // If we find no inflection points, we simply split
                // down the middle of the longest axis.
                const mid_axis = longest;
                const mid_index = (upper_bounds[mid_axis] + lower_bounds[mid_axis]) / 2;

                if (self.splitBoxes(masks, tiles, partition, lower_bounds, upper_bounds, mid_axis, mid_index)) |split| {
                    try stack.append(allocator, split.right);
                    try stack.append(allocator, split.left);
                } else {
                    try self.blocks.append(allocator, partition);
                }
            }

            // Fill parents array.
            self.computeParents();
        }

        fn computeSignatures(self: *const Self, signatures: [N][]usize, tiles: Tiles(N), partition: Partition(N)) void {
            _ = self;
            const subblock = partition.bounds;
            const space: IndexSpace(N) = .{ .size = tiles.size };
            const subspace: IndexSpace(N) = .{ .size = subblock.size };

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

                        if (tiles.tags[linear]) {
                            sig += 1;
                        }
                    }

                    signatures[axis][subblock.origin[axis] + i] += sig;
                }
            }
        }

        fn computeMasks(self: *const Self, masks: [N][]bool, tiles: Tiles(N), partition: Partition(N)) void {
            const subblock = partition.bounds;

            for (0..N) |axis| {
                @memset(masks[axis][subblock.origin[axis]..(subblock.origin[axis] + subblock.size[axis])], false);

                const offset: usize = partition.children_offset;
                const count: usize = partition.children_count;

                for (self.children.items[offset..(offset + count)]) |child| {
                    const bounds = tiles.clusters[child];

                    @memset(masks[axis][bounds.origin[axis]..(bounds.origin[axis] + bounds.size[axis])], true);
                }
            }
        }

        fn computeParents(self: *Self) void {
            for (self.blocks.items(.children_offset), self.blocks.items(.children_count), 0..) |offset, count, i| {
                for (self.children.items[offset..(offset + count)]) |child| {
                    self.parents.items[child] = i;
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
            tiles: Tiles(N),
            parent: Partition(N),
            lower: [N]usize,
            upper: [N]usize,
            axis: usize,
            index: usize,
        ) ?struct { left: Partition(N), right: Partition(N) } {
            assert(index > 0);

            // Sort children
            const context: SplitContext = .{
                .clusters = tiles.clusters,
                .axis = axis,
            };

            // Use heap sort.
            heap(usize, self.children.items[parent.children_offset..(parent.children_count + parent.children_offset)], context, splitLessThanFn);

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
            var clusters_index = parent.children_count;
            // Perform search in reverse as we favour higher indices.
            // Worst case: O(n).
            while (clusters_index > 0 and tiles.clusters[self.children.items[parent.children_offset + clusters_index - 1]].origin[axis] >= split_index) {
                clusters_index -= 1;
            }

            // Generate left and right partitions
            var left: Partition(N) = undefined;
            var right: Partition(N) = undefined;

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
            left.children_count = clusters_index;
            right.children_offset = parent.children_offset + clusters_index;
            right.children_count = parent.children_count - clusters_index;

            return .{
                .left = left,
                .right = right,
            };
        }
    };
}

fn buildTiles(comptime N: usize, allocator: Allocator, size: [N]usize, tagged: []const [N]usize, clusters: []Box(N, usize)) !Tiles(N) {
    const space: IndexSpace(N) = .{ .size = size };
    const total = space.total();

    var tags: []bool = try allocator.alloc(bool, total);
    errdefer allocator.free(tags);

    @memset(tags, false);

    for (tagged) |t| {
        tags[space.linearFromCartesian(t)] = true;
    }

    return .{ .size = size, .tags = tags, .clusters = clusters };
}

test "tiles basic" {
    const expect = std.testing.expect;
    const expectEqualSlices = std.testing.expectEqualSlices;

    const size = [3]usize{ 5, 5, 5 };

    const tagged = [_][3]usize{
        [_]usize{ 0, 0, 0 },
        [_]usize{ 3, 2, 1 },
        [_]usize{ 0, 1, 3 },
    };

    var tiles: Tiles(3) = try buildTiles(3, std.testing.allocator, size, &tagged, &[_]Box(3, usize){});
    defer std.testing.allocator.free(tiles.tags);

    const buffer_length: usize = size[0] + size[1] + size[2];
    var signature_buffer: []usize = try std.testing.allocator.alloc(usize, buffer_length);
    defer std.testing.allocator.free(signature_buffer);

    var signatures: [3][]usize = undefined;

    var buffer_offset: usize = 0;
    for (0..3) |i| {
        signatures[i] = signature_buffer[buffer_offset..(buffer_offset + size[i])];
        buffer_offset += size[i];
    }

    var partitions: Partitions(3) = .{};
    defer partitions.deinit(std.testing.allocator);

    partitions.computeSignatures(signatures, tiles, .{
        .bounds = .{
            .origin = [3]usize{ 0, 0, 0 },
            .size = [3]usize{ 5, 5, 5 },
        },
        .children_offset = 0,
        .children_count = 0,
    });

    try expectEqualSlices(usize, &[5]usize{ 2, 0, 0, 1, 0 }, signatures[0]);
    try expectEqualSlices(usize, &[5]usize{ 1, 1, 1, 0, 0 }, signatures[1]);
    try expectEqualSlices(usize, &[5]usize{ 1, 1, 0, 1, 0 }, signatures[2]);

    const efficiency = tiles.computeEfficiency(.{
        .origin = [3]usize{ 0, 0, 0 },
        .size = [3]usize{ 4, 4, 4 },
    });

    try expect(efficiency == 3.0 / 64.0);
}

test "tile partitioning" {
    const expect = std.testing.expect;
    const expectEqualSlices = std.testing.expectEqualSlices;

    const size = [2]usize{ 5, 5 };

    const tagged = [_][2]usize{
        [_]usize{ 2, 2 },
        [_]usize{ 3, 2 },
        [_]usize{ 2, 3 },
        [_]usize{ 3, 3 },
    };

    var tiles: Tiles(2) = try buildTiles(2, std.testing.allocator, size, &tagged, &[_]Box(2, usize){});
    defer std.testing.allocator.free(tiles.tags);

    var partitions: Partitions(2) = .{};
    defer partitions.deinit(std.testing.allocator);

    try partitions.compute(std.testing.allocator, tiles, 4, 0.7);

    try expect(partitions.blocks.items(.bounds).len == 1);

    const bounds = partitions.blocks.items(.bounds)[0];
    try expectEqualSlices(usize, &[2]usize{ 2, 2 }, &bounds.origin);
    try expectEqualSlices(usize, &[2]usize{ 2, 2 }, &bounds.size);
}
