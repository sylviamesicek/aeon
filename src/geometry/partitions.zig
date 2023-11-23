const std = @import("std");
const heap = std.sort.heap;
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;

/// A set of tiles in some index space, grouped into several partitions.
/// Individual tiles may be "tagged" as active, and this object provides an
/// interface into grouping active tiles into partitions as well as specify "clusters" of
/// tiles which may not be bisected by these partitions (ie, there is a direct parent-child
/// relationship between a partition and a set of clusters).
pub fn Partitions(comptime N: usize) type {
    return struct {
        /// The list of partitions
        bounds: []const IndexBox,
        /// Children of each partition
        children: []const Range,
        /// Parents
        parents: []const usize,
        /// An index buffer from children to clusters
        buffer: []const usize,

        const Self = @This();
        const IndexBox = @import("box.zig").IndexBox(N);
        const IndexSpace = @import("index.zig").IndexSpace(N);

        pub const Range = struct {
            start: usize,
            end: usize,
        };

        /// Computes a set of partitions for a tile space given a number of tagged tiles, a set of clusters (which may not be bisected)
        /// a prescribed efficiency, and a maximum number of tiles for each partition.
        pub fn init(allocator: Allocator, size: [N]usize, tags: []const bool, clusters: []const IndexBox, max_tiles: usize, min_efficiency: f64) !Self {

            // **************************
            // General Setup ************
            //***************************

            // Get tag space
            const tag_space = IndexSpace.fromSize(size);
            // Check tags match given bounds
            assert(tag_space.total() == tags.len);

            // Setup default cluster buffer

            var cluster_buffer = try allocator.alloc(usize, clusters.len);
            errdefer allocator.free(cluster_buffer);

            for (0..cluster_buffer.len) |i| {
                cluster_buffer[i] = i;
            }

            // ******************************
            // Context **********************
            // ******************************

            // Find the required buffer length
            var buffer_length: usize = 0;

            for (0..N) |i| {
                buffer_length += size[i];
            }

            // Allocate scratch data
            var signature_buffer: []usize = try allocator.alloc(usize, buffer_length);
            defer allocator.free(signature_buffer);

            var mask_buffer: []bool = try allocator.alloc(bool, buffer_length);
            defer allocator.free(mask_buffer);

            // Build slices for each axis.
            var signatures: [N][]usize = undefined;
            var masks: [N][]bool = undefined;

            var buffer_offset: usize = 0;

            for (0..N) |i| {
                signatures[i] = signature_buffer[buffer_offset..(buffer_offset + size[i])];
                masks[i] = mask_buffer[buffer_offset..(buffer_offset + size[i])];

                buffer_offset += size[i];
            }

            var ctx: Context = .{
                .size = size,
                .clusters = clusters,
                .tags = tags,
                .buffer = cluster_buffer,
                .signatures = signatures,
                .masks = masks,
            };

            // *************************
            // Recurse through stack ***
            // *************************

            // Set default partition
            var stack: ArrayListUnmanaged(Partition) = .{};
            defer stack.deinit(allocator);

            try stack.append(
                allocator,
                .{
                    .bounds = .{
                        .origin = [1]usize{0} ** N,
                        .size = size,
                    },
                    .children = .{ .start = 0, .end = clusters.len },
                },
            );

            // Store partitions
            var partitions: ArrayListUnmanaged(Partition) = .{};
            defer partitions.deinit(allocator);

            // Begin recusively dividing partitions.
            stack_pop: while (stack.popOrNull()) |partition| {
                // Compute signatures
                ctx.computeSignatures(partition.bounds);
                ctx.computeMasks(partition.bounds, partition.children);

                // Set bounds for this partition
                var lower_bounds: [N]usize = partition.bounds.origin;
                var upper_bounds: [N]usize = partition.bounds.origin;

                for (0..N) |i| {
                    upper_bounds[i] += partition.bounds.size[i];
                }

                // Cull any unnessessary spacing on either side.
                for (0..N) |axis| {
                    while (ctx.signatures[axis][lower_bounds[axis]] == 0 and !ctx.masks[axis][lower_bounds[axis]]) {
                        lower_bounds[axis] += 1;
                    }

                    while (ctx.signatures[axis][upper_bounds[axis] - 1] == 0 and !ctx.masks[axis][upper_bounds[axis] - 1]) {
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

                const culled_space: IndexSpace = IndexSpace.fromBox(culled_subblock);

                const culled_partition: Partition = .{
                    .bounds = culled_subblock,
                    .children = partition.children,
                };

                // Check efficiency and maximum block sidelength
                const efficiency = ctx.computeEfficiency(culled_subblock);
                const longest = culled_space.longestAxis();

                // If these checks pass, we can add this cell as a new block.
                if (efficiency >= min_efficiency and culled_space.size[longest] <= max_tiles) {
                    try partitions.append(allocator, culled_partition);
                    continue :stack_pop;
                }

                // If they don't pass, we must partition the subblock into two
                // and add both to the stack.

                // First we check for holes
                var found_hole: bool = false;
                var hole_indices: [N]usize = [1]usize{0} ** N;

                for (0..N) |axis| {
                    for (lower_bounds[axis]..upper_bounds[axis]) |i| {
                        if (ctx.signatures[axis][i] == 0 and i > hole_indices[axis]) {
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

                    if (ctx.splitBoxes(culled_partition, lower_bounds, upper_bounds, hole_axis, hole_index)) |split| {
                        try stack.append(allocator, split.right);
                        try stack.append(allocator, split.left);
                    } else {
                        try partitions.append(allocator, culled_partition);
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

                    if (ctx.splitBoxes(culled_partition, lower_bounds, upper_bounds, inflection_axis, inflection_index)) |split| {
                        try stack.append(allocator, split.right);
                        try stack.append(allocator, split.left);
                    } else {
                        try partitions.append(allocator, culled_partition);
                    }

                    continue :stack_pop;
                }

                // If we find no inflection points, we simply split
                // down the middle of the longest axis.
                const mid_axis = longest;
                const mid_index = (upper_bounds[mid_axis] + lower_bounds[mid_axis]) / 2;

                if (ctx.splitBoxes(culled_partition, lower_bounds, upper_bounds, mid_axis, mid_index)) |split| {
                    try stack.append(allocator, split.right);
                    try stack.append(allocator, split.left);
                } else {
                    try partitions.append(allocator, culled_partition);
                }
            }

            const bounds = try allocator.alloc(IndexBox, partitions.items.len);
            errdefer allocator.free(bounds);

            const children = try allocator.alloc(Range, partitions.items.len);
            errdefer allocator.free(children);

            for (partitions.items, 0..) |partition, idx| {
                bounds[idx] = partition.bounds;
                children[idx] = partition.children;
            }

            const parents = try allocator.alloc(usize, clusters.len);
            errdefer allocator.free(parents);

            for (children, 0..) |c, idx| {
                for (cluster_buffer[c.start..c.end]) |cluster| {
                    parents[cluster] = idx;
                }
            }

            return .{
                .bounds = bounds,
                .children = children,
                .parents = parents,
                .buffer = cluster_buffer,
            };
        }

        /// Deinitialises a partitions data structure, freeing all internal lists and maps.
        pub fn deinit(self: *const Self, allocator: Allocator) void {
            allocator.free(self.bounds);
            allocator.free(self.children);
            allocator.free(self.parents);
            allocator.free(self.buffer);
        }

        pub fn len(self: *const Self) usize {
            return self.bounds.len;
        }

        // ***************************
        // Implementation ************
        // ***************************

        const Partition = struct {
            bounds: IndexBox,
            children: Range,
        };

        const Context = struct {
            size: [N]usize,
            tags: []const bool,
            clusters: []const IndexBox,
            buffer: []usize,
            signatures: [N][]usize,
            masks: [N][]bool,

            /// Computes the efficiency of a given subblock.
            fn computeEfficiency(
                ctx: *const Context,
                subblock: IndexBox,
            ) f64 {
                const subspace = IndexSpace.fromSize(subblock.size);
                const space = IndexSpace.fromSize(ctx.size);

                const n_total = subspace.total();
                var n_tagged: usize = 0;
                var indices = subspace.cartesianIndices();

                while (indices.next()) |local| {
                    var global: [N]usize = undefined;

                    for (0..N) |i| {
                        global[i] = subblock.origin[i] + local[i];
                    }

                    const linear = space.linearFromCartesian(global);

                    if (ctx.tags[linear]) {
                        n_tagged += 1;
                    }
                }

                const f_tagged: f64 = @floatFromInt(n_tagged);
                const f_total: f64 = @floatFromInt(n_total);

                return f_tagged / f_total;
            }

            /// Computes the signatures along each axis over some subblock
            fn computeSignatures(ctx: *Context, subblock: IndexBox) void {
                const space: IndexSpace = IndexSpace.fromSize(ctx.size);
                const subspace: IndexSpace = IndexSpace.fromBox(subblock);

                for (0..N) |axis| {
                    // Reset the signatures in this region
                    @memset(ctx.signatures[axis][subblock.origin[axis]..(subblock.origin[axis] + subblock.size[axis])], 0);

                    for (0..subblock.size[axis]) |i| {
                        var sig: usize = 0;

                        var iterator = subspace.cartesianSliceIndices(axis, i);

                        while (iterator.next()) |local| {
                            var global: [N]usize = undefined;

                            for (0..N) |j| {
                                global[j] = subblock.origin[j] + local[j];
                            }

                            const linear = space.linearFromCartesian(global);

                            if (ctx.tags[linear]) {
                                sig += 1;
                            }
                        }

                        ctx.signatures[axis][subblock.origin[axis] + i] += sig;
                    }
                }
            }

            fn computeMasks(
                ctx: *Context,
                subblock: IndexBox,
                children: Range,
            ) void {
                for (0..N) |axis| {
                    @memset(ctx.masks[axis][subblock.origin[axis]..(subblock.origin[axis] + subblock.size[axis])], false);

                    for (ctx.buffer[children.start..children.end]) |child| {
                        const bounds = ctx.clusters[child];

                        @memset(ctx.masks[axis][bounds.origin[axis]..(bounds.origin[axis] + bounds.size[axis])], true);
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
                ctx: *const Context,
                parent: Partition,
                lower: [N]usize,
                upper: [N]usize,
                axis: usize,
                index: usize,
            ) ?struct { left: Partition, right: Partition } {
                assert(index > 0);

                // Sort children
                const context: SplitContext = .{
                    .clusters = ctx.clusters,
                    .axis = axis,
                };

                // Use heap sort.
                heap(usize, ctx.buffer[parent.children.start..parent.children.end], context, splitLessThanFn);

                // Find adjusted index which does not bisect cluster.
                var index_inc: usize = index;
                var index_dec: usize = index;

                var split_index: usize = undefined;

                while (true) {
                    if (index_inc < upper[axis]) {
                        if (ctx.masks[axis][index_inc] and ctx.masks[axis][index_inc - 1]) {
                            index_inc += 1;
                        } else {
                            split_index = index_inc;
                            break;
                        }
                    } else if (index_dec > 0) {
                        if (ctx.masks[axis][index_dec] and ctx.masks[axis][index_dec - 1]) {
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
                var clusters_index = parent.children.end;
                // Perform search in reverse as we favour higher indices.
                // Worst case: O(n).
                while (clusters_index > parent.children.start and ctx.clusters[ctx.buffer[clusters_index - 1]].origin[axis] >= split_index) {
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

                left.children.start = parent.children.start;
                left.children.end = clusters_index;
                right.children.start = clusters_index;
                right.children.end = parent.children.end;

                return .{
                    .left = left,
                    .right = right,
                };
            }
        };
    };
}

fn buildTags(comptime N: usize, allocator: Allocator, size: [N]usize, tagged: []const [N]usize) ![]bool {
    const IndexSpace = @import("index.zig").IndexSpace(N);

    const space: IndexSpace = .{ .size = size };
    const total = space.total();

    var tags: []bool = try allocator.alloc(bool, total);
    errdefer allocator.free(tags);

    @memset(tags, false);

    for (tagged) |t| {
        tags[space.linearFromCartesian(t)] = true;
    }

    return tags;
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

    var partitions = try Partitions(2).init(allocator, size, tags, &.{}, 4, 0.7);
    defer partitions.deinit(allocator);

    try expect(partitions.len() == 1);
    try expectEqualSlices(usize, &[2]usize{ 2, 2 }, &partitions.bounds[0].origin);
    try expectEqualSlices(usize, &[2]usize{ 2, 2 }, &partitions.bounds[0].size);
}
