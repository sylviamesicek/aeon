const std = @import("std");
const sort = std.sort;
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const ArrayList = std.ArrayList;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const MultiArrayList = std.MultiArrayList;
const assert = std.debug.assert;

const box = @import("box.zig");
const index = @import("index.zig");

/// An container for the results of block clustering. This includes the resulting clusters, and a map from each cluster
/// to indices into the original slice of blocks passed to the algorithm.
pub fn BlockClusters(comptime N: usize) type {
    return struct {
        clusters: []const IndexBox,
        children: []const []const usize,
        buffer: []const usize,

        const IndexBox = box.IndexBox(N);

        pub fn deinit(self: @This(), allocator: Allocator) void {
            allocator.free(self.clusters);
            allocator.free(self.buffer);
        }
    };
}

/// Provides access to various point and block clustering algorithms, primarily used for block structured
/// adaptive mesh refinement.
pub fn ClusterSpace(comptime N: usize) type {
    return struct {
        /// Size of the tile space upon which clusters will be defined.
        size: [N]usize,
        /// The minimum efficiency of each resulting cluster.
        min_efficiency: f64,
        /// The maximum number of tiles along any given side for the resulting cluster (defaults to `maxInt(usize)`).
        max_tiles: usize = std.math.maxInt(usize),

        const IndexBox = box.IndexBox(N);
        const IndexSpace = index.IndexSpace(N);

        pub fn indexSpace(self: @This()) IndexSpace {
            return IndexSpace.fromSize(self.size);
        }

        /// Berger-Rigoutso point clustering. The resulting slice is owned by the caller, and must be freed using the same allocator.
        pub fn points(self: @This(), allocator: Allocator, tags: []bool) ![]IndexBox {
            // **************************
            // General Setup ************
            // **************************

            // Get tag space
            const tag_space = self.indexSpace();
            // Check tags match given bounds
            assert(tag_space.total() == tags.len);

            // ****************************
            // Context ********************
            // ****************************
            var buffer_len: usize = 0;

            for (0..N) |i| {
                buffer_len += self.size[i];
            }

            const buffer = try allocator.alloc(usize, buffer_len);
            defer allocator.free(buffer);

            var signatures: [N][]usize = undefined;

            var buffer_off: usize = 0;

            for (0..N) |i| {
                signatures[i] = buffer[buffer_off..(buffer_off + self.size[i])];
                buffer_off += self.size[i];
            }

            // ****************************
            // Recurese through stack *****
            // ****************************

            // Set default cluster
            var stack: ArrayListUnmanaged(IndexBox) = .{};
            defer stack.deinit(allocator);

            try stack.append(allocator, .{
                .origin = [1]usize{0} ** N,
                .size = self.size,
            });

            // Store accepted clusters
            var clusters: ArrayListUnmanaged(IndexBox) = .{};
            errdefer clusters.deinit(allocator);

            // Begin recursively dividing partitions

            stack_pop: while (stack.popOrNull()) |block| {
                self.computeSignatures(block, tags, signatures);

                // Set bounds for this partition
                var lower_bounds: [N]usize = block.origin;
                var upper_bounds: [N]usize = block.origin;

                for (0..N) |i| {
                    upper_bounds[i] += block.size[i];
                }

                // Cull any unnessessary spacing on either side.
                for (0..N) |axis| {
                    while (signatures[axis][lower_bounds[axis]] == 0) {
                        lower_bounds[axis] += 1;
                    }

                    while (signatures[axis][upper_bounds[axis] - 1] == 0) {
                        upper_bounds[axis] -= 1;
                    }

                    if (lower_bounds[axis] >= upper_bounds[axis]) {
                        // Block has an axis with zero width, so ignore it and move to next partition.
                        continue :stack_pop;
                    }
                }

                // Use computed bounds to build culled subblock
                var culled_block: IndexBox = undefined;

                for (0..N) |axis| {
                    culled_block.origin[axis] = lower_bounds[axis];
                    culled_block.size[axis] = upper_bounds[axis] - lower_bounds[axis];
                }

                const culled_space: IndexSpace = IndexSpace.fromBox(culled_block);

                // Check efficiency and maximum block sidelength
                const efficiency = self.computePointEfficiency(culled_block, tags);
                const longest = culled_space.longestAxis();

                // If these checks pass, we can add this cell as a new block
                if (efficiency >= self.min_efficiency and culled_block.size[longest] <= self.max_tiles) {
                    try clusters.append(allocator, culled_block);
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

                    const split = splitBoxes(culled_block, hole_axis, hole_index);
                    try stack.append(allocator, split.right);
                    try stack.append(allocator, split.left);
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
                    if (upper_bounds[axis] - lower_bounds[axis] <= 3) {
                        continue;
                    }

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

                    const split = splitBoxes(culled_block, inflection_axis, inflection_index);
                    try stack.append(allocator, split.right);
                    try stack.append(allocator, split.left);
                    continue :stack_pop;
                }

                // If we find no inflection points, we simply split
                // down the middle of the longest axis.
                const mid_axis = longest;
                const mid_index = (upper_bounds[mid_axis] + lower_bounds[mid_axis]) / 2;

                const split = splitBoxes(culled_block, mid_axis, mid_index);
                try stack.append(allocator, split.right);
                try stack.append(allocator, split.left);
            }

            return clusters.toOwnedSlice(allocator);
        }

        /// Block clustering based on finding the largest holes between blocks.
        /// The resulting memory is owned by the caller, and must be freed using the same allocator.
        pub fn blocks(self: @This(), allocator: Allocator, blks: []const IndexBox) !BlockClusters(N) {
            // ****************************
            // General ********************
            // ****************************

            const index_buffer: []const usize = try allocator.alloc(usize, blks.len);
            errdefer allocator.free(index_buffer);

            for (0..index_buffer.len) |i| {
                index_buffer[i] = i;
            }

            // ****************************
            // Context ********************
            // ****************************

            var buffer_len: usize = 0;

            for (0..N) |i| {
                buffer_len += self.size[i];
            }

            const mask_buffer = try allocator.alloc(bool, buffer_len);
            defer allocator.free(mask_buffer);

            var masks: [N][]bool = undefined;

            var buffer_off: usize = 0;

            for (0..N) |i| {
                masks[i] = mask_buffer[buffer_off..(buffer_off + self.size[i])];
                buffer_off += self.size[i];
            }

            // *************************
            // Recurse through stack ***
            // *************************

            // Set default partition
            var stack: ArrayListUnmanaged(BlockCluster) = .{};
            defer stack.deinit(allocator);

            try stack.append(
                allocator,
                .{
                    .bounds = .{
                        .origin = [1]usize{0} ** N,
                        .size = self.size,
                    },
                    .children = index_buffer,
                },
            );

            // Store partitions
            var clusters: MultiArrayList(BlockCluster) = .{};
            errdefer clusters.deinit(allocator);

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(allocator);
            defer arena.deinit();

            var scratch: Allocator = arena.allocator();

            // Begin recusively dividing clusters.
            stack_pop: while (stack.popOrNull()) |cluster| {
                // Reset arena allocator
                defer _ = arena.reset(.retain_capacity);

                computeMasks(cluster, blks, masks);

                // Set bounds for this partition
                var lower_bounds: [N]usize = cluster.block.origin;
                var upper_bounds: [N]usize = cluster.block.origin;

                for (0..N) |i| {
                    upper_bounds[i] += cluster.block.size[i];
                }

                // Cull any unnessessary spacing on either side.
                for (0..N) |axis| {
                    while (!masks[axis][lower_bounds[axis]]) {
                        lower_bounds[axis] += 1;
                    }

                    while (!masks[axis][upper_bounds[axis] - 1]) {
                        upper_bounds[axis] -= 1;
                    }

                    if (lower_bounds[axis] >= upper_bounds[axis]) {
                        // Partition has an axis with zero width, so ignore it and move to next partition.
                        continue :stack_pop;
                    }
                }

                // Use computed bounds to build culled subblock
                var culled_block: IndexBox = undefined;

                for (0..N) |axis| {
                    culled_block.origin[axis] = lower_bounds[axis];
                    culled_block.size[axis] = upper_bounds[axis] - lower_bounds[axis];
                }

                const culled_space: IndexSpace = IndexSpace.fromBox(culled_block);

                const culled_cluster: BlockCluster = .{
                    .block = culled_block,
                    .children = cluster.children,
                };

                // Compute efficiency (while accounting for overlaps)
                const n_total: usize = culled_space.total();

                const culled_tags: []bool = scratch.alloc(bool, n_total);
                defer scratch.free(culled_tags);

                const efficiency = computeClusterEfficiency(culled_cluster, blks, culled_tags);

                const longest = culled_space.longestAxis();

                // If these checks pass, the culled cluster is suitable.
                if (efficiency >= self.min_efficiency and culled_block.size[longest] <= self.max_tiles) {
                    try clusters.append(allocator, culled_cluster);
                    continue :stack_pop;
                }

                // If they don't pass, we must search for the optimal axis and index along which to slice.
                var found_hole: bool = false;
                var hole_indices: [N]usize = [1]usize{0} ** N;

                for (0..N) |axis| {
                    var run_active = false;
                    var hole_index = 0;
                    var hole_size = 0;

                    for (lower_bounds[axis]..upper_bounds[axis]) |i| {
                        if (run_active and masks[axis[i]]) {
                            const size = i + 1 - hole_index;

                            if (size >= hole_size) {
                                hole_indices[axis] = hole_index;
                                found_hole = true;
                            }

                            run_active = false;
                        } else if (!run_active and !masks[axis][i]) {
                            hole_index = i;
                            run_active = true;
                        }
                    }

                    if (run_active) {
                        const size = upper_bounds[axis] - hole_index;

                        if (size >= hole_size) {
                            hole_indices[axis] = hole_index;
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

                    const split = self.splitBoxes(culled_block, hole_axis, hole_index);

                    // Sort children using heap sort
                    const ctx: SplitContext = .{
                        .blocks = blks,
                        .axis = hole_axis,
                    };

                    // Use heap sort O(n*log(n))
                    sort.heap(usize, cluster.children, ctx, splitLessThanFn);

                    // Find index to split children clusters.
                    var left_len: usize = cluster.children.len;
                    // Perform search in reverse as we favour higher indices.
                    // Worst case: O(n)
                    while (left_len > 0 and blks[cluster.children[left_len - 1]].origin[hole_axis] >= hole_index) {
                        left_len -= 1;
                    }

                    const left: BlockCluster = .{
                        .block = split.left,
                        .children = cluster.children[0..left_len],
                    };

                    const right: BlockCluster = .{
                        .block = split.right,
                        .children = cluster.children[left_len..],
                    };

                    try stack.append(allocator, right);
                    try stack.append(allocator, left);

                    continue :stack_pop;
                }

                // We have failed to find a hole to split, accept the current cluster
                try clusters.append(allocator, culled_cluster);
            }

            const slice = clusters.toOwnedSlice();

            return .{
                .clusters = slice.items(.block),
                .children = slice.items(.children),
                .buffer = index_buffer,
            };
        }

        const BlockCluster = struct {
            block: IndexBox,
            children: []const usize,
        };

        fn splitBoxes(block: IndexBox, axis: usize, idx: usize) struct { left: IndexBox, right: IndexBox } {
            var left: IndexBox = block;
            var right: IndexBox = block;

            left.size[axis] = idx - block.origin[axis];
            right.origin[axis] = idx;
            right.size[axis] = block.origin[axis] + block.size[axis] - idx;

            return .{
                .left = left,
                .right = right,
            };
        }

        /// Computes the efficiency of a given subblock.
        fn computePointEfficiency(self: @This(), block: IndexBox, tags: []const bool) f64 {
            const subspace = IndexSpace.fromSize(block.size);
            const space = IndexSpace.fromSize(self.size);

            const n_total = subspace.total();
            var n_tagged: usize = 0;
            var indices = subspace.cartesianIndices();

            while (indices.next()) |local| {
                const global = block.globalFromLocal(local);
                const linear = space.linearFromCartesian(global);

                if (tags[linear]) {
                    n_tagged += 1;
                }
            }

            const f_tagged: f64 = @floatFromInt(n_tagged);
            const f_total: f64 = @floatFromInt(n_total);

            return f_tagged / f_total;
        }

        fn computeClusterEfficiency(cluster: BlockCluster, blks: []const IndexBox, tags: []bool) f64 {
            @memset(tags, false);

            for (cluster.children) |child| {
                var local_block: IndexBox = undefined;
                local_block.origin = cluster.block.localFromGlobal(blks[child].origin);
                local_block.size = blks[child].size;

                IndexSpace.fromBox(cluster.block).fillWindow(local_block, bool, tags, true);
            }

            var n_tagged: usize = 0;

            for (tags) |tagged| {
                if (tagged) {
                    n_tagged += 1;
                }
            }

            const f_tagged: f64 = @floatFromInt(n_tagged);
            const f_total: f64 = @floatFromInt(tags.len);

            return f_tagged / f_total;
        }

        /// Computes the signatures along each axis over some subblock
        fn computeSignatures(self: @This(), block: IndexBox, tags: []const bool, signatures: [N][]usize) void {
            const space: IndexSpace = IndexSpace.fromSize(self.size);
            const subspace: IndexSpace = IndexSpace.fromBox(block);

            for (0..N) |axis| {
                for (0..block.size[axis]) |i| {
                    var sig: usize = 0;

                    var iterator = subspace.cartesianSliceIndices(axis, i);

                    while (iterator.next()) |local| {
                        const global = block.globalFromLocal(local);
                        const linear = space.linearFromCartesian(global);

                        if (tags[linear]) {
                            sig += 1;
                        }
                    }

                    signatures[axis][block.origin[axis] + i] = sig;
                }
            }
        }

        /// Computes the masks for each axis over some subcluster
        fn computeMasks(cluster: BlockCluster, blks: []const IndexBox, masks: [N][]bool) void {
            for (0..N) |axis| {
                @memset(masks[axis][cluster.block.origin[axis]..(cluster.block.origin[axis] + cluster.block.size[axis])], false);

                for (cluster.children) |child| {
                    const bounds = blks[child];

                    @memset(masks[axis][bounds.origin[axis]..(bounds.origin[axis] + bounds.size[axis])], true);
                }
            }
        }

        const SplitContext = struct {
            blocks: []const IndexBox,
            axis: usize,
        };

        fn splitLessThanFn(context: SplitContext, lhs: usize, rhs: usize) bool {
            return context.blocks[lhs].origin[context.axis] < context.blocks[rhs].origin[context.axis];
        }
    };
}

test "point clustering" {
    const IndexSpace = index.IndexSpace(2);
    const expectEqualDeep = std.testing.expectEqualDeep;

    const allocator = std.testing.allocator;

    const size: [2]usize = .{ 16, 16 };

    const tags = try allocator.alloc(bool, size[0] * size[1]);
    defer allocator.free(tags);

    {
        @memset(tags, false);

        const tagged = [_][2]usize{
            [_]usize{ 2, 2 },
            [_]usize{ 3, 2 },
            [_]usize{ 2, 3 },
            [_]usize{ 3, 3 },
            [_]usize{ 3, 4 },
            [_]usize{ 3, 5 },
            [_]usize{ 7, 3 },
            [_]usize{ 8, 3 },
            [_]usize{ 7, 4 },
            [_]usize{ 8, 4 },
        };

        const index_space = IndexSpace.fromSize(size);

        for (tagged) |i| {
            tags[index_space.linearFromCartesian(i)] = true;
        }
    }

    var cluster_space = ClusterSpace(2){
        .size = size,
        .min_efficiency = 0.75,
    };

    const clusters1 = try cluster_space.points(allocator, tags);
    defer allocator.free(clusters1);

    try expectEqualDeep(clusters1[0], .{ .origin = .{ 2, 2 }, .size = .{ 2, 4 } });
    try expectEqualDeep(clusters1[1], .{ .origin = .{ 7, 3 }, .size = .{ 2, 2 } });

    cluster_space.min_efficiency = 1.0;

    const clusters2 = try cluster_space.points(allocator, tags);
    defer allocator.free(clusters2);

    try expectEqualDeep(clusters2[0], .{ .origin = .{ 2, 2 }, .size = .{ 2, 2 } });
    try expectEqualDeep(clusters2[1], .{ .origin = .{ 3, 4 }, .size = .{ 1, 2 } });
    try expectEqualDeep(clusters2[2], .{ .origin = .{ 7, 3 }, .size = .{ 2, 2 } });
}
