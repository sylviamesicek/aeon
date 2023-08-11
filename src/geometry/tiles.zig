const std = @import("std");

const Box = @import("box.zig").Box;
const IndexSpace = @import("index.zig").IndexSpace;

/// Provides an interface for grid generation on a single level using
/// the Berger-Rigoutsos point clustering algorithm.
pub fn Tiles(comptime N: usize) type {
    return struct {
        size: [N]usize,
        tagged: std.ArrayListUnmanaged(bool),
        blocks: std.ArrayListUnmanaged(IndexBox),

        const Self = @This();
        const IndexBox = Box(N, usize);

        pub fn init(allocator: std.mem.Allocator, size: [N]usize) !Self {
            const space: IndexSpace(N) = .{ .size = size };
            const total = space.total();

            var tagged = try std.ArrayListUnmanaged(bool).initCapacity(allocator, total);
            errdefer tagged.deinit(allocator);

            tagged.appendNTimesAssumeCapacity(false, total);

            var blocks = std.ArrayListUnmanaged(IndexBox){};
            errdefer blocks.deinit(allocator);

            return .{
                .size = size,
                .tagged = tagged,
                .blocks = blocks,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.tagged.deinit(allocator);
            self.blocks.deinit(allocator);
        }

        pub fn resetTags(self: *Self) void {
            const len = self.tagged.items.len;
            self.tagged.clearRetainingCapacity();
            self.tagged.appendNTimesAssumeCapacity(false, len);
        }

        pub fn tag(self: *Self, tile: [N]usize) void {
            const space: IndexSpace(N) = .{ .size = self.size };
            self.tagged.items[space.linearFromCartesian(tile)] = true;
        }

        pub fn tagMany(self: *Self, tiles: []const [N]usize) void {
            const space: IndexSpace(N) = .{ .size = self.size };

            for (tiles) |tile| {
                self.tagged.items[space.linearFromCartesian(tile)] = true;
            }
        }

        pub fn efficiency(self: Self, subblock: IndexBox) f64 {
            const subspace = IndexSpace(N){ .size = subblock.widths };
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

                if (self.tagged.items[linear]) {
                    n_tagged += 1;
                }
            }

            const f_tagged: f64 = @floatFromInt(n_tagged);
            const f_total: f64 = @floatFromInt(n_total);

            return f_tagged / f_total;
        }

        pub fn partition(self: Self, allocator: std.mem.Allocator, max_tiles: usize, efficiency_ratio: f64, blocks: *std.ArrayListUnmanaged(IndexBox)) !void {

            // Allocate signatures
            var signatures: Signatures(N) = try Signatures(N).initCapacity(allocator, self.size);
            defer signatures.deinit(allocator);

            // Build stack of subblocks
            var stack = std.ArrayListUnmanaged(IndexBox){};
            defer stack.deinit(allocator);

            try stack.append(allocator, .{
                .origin = [1]usize{0} ** N,
                .widths = self.size,
            });

            stack_pop: while (stack.popOrNull()) |subblock| {
                // Remove edges
                signatures.computeAssumeCapacity(self, subblock);

                var lower_bounds: [N]usize = [1]usize{0} ** N;
                var upper_bounds: [N]usize = subblock.widths;

                for (0..N) |axis| {
                    while (signatures.axes[axis].items[lower_bounds[axis]] == 0) {
                        lower_bounds[axis] += 1;
                    }

                    while (signatures.axes[axis].items[upper_bounds[axis] - 1] == 0) {
                        upper_bounds[axis] -= 1;
                    }

                    if (lower_bounds[axis] >= upper_bounds[axis]) {
                        continue :stack_pop;
                    }
                }

                // Use computed bounds to build culled subblock
                var culled_subblock: IndexBox = undefined;

                for (0..N) |axis| {
                    culled_subblock.origin[axis] = subblock.origin[axis] + lower_bounds[axis];
                    culled_subblock.widths[axis] = upper_bounds[axis] - lower_bounds[axis];
                }

                const culled_space: IndexSpace(N) = .{ .size = culled_subblock.widths };

                // Check efficiency and maximum block sidelength
                const ratio = self.efficiency(culled_subblock);
                const longest = culled_space.longestAxis();

                // If these checks pass, we can add this cell as a new block.
                if (ratio >= efficiency_ratio and culled_space.size[longest] <= max_tiles) {
                    try blocks.append(allocator, culled_subblock);
                    continue :stack_pop;
                }

                // If they don't pass, we must partition the subblock into two
                // and add both to the stack.

                // First we check for holes
                var found_hole: bool = false;
                var hole_indices: [N]usize = [1]usize{0} ** N;

                for (0..N) |axis| {
                    for (lower_bounds[axis]..upper_bounds[axis]) |i| {
                        if (signatures.axes[axis].items[i] == 0 and i > hole_indices[axis]) {
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

                    const split = split_boxes(subblock.origin, lower_bounds, upper_bounds, hole_axis, hole_index);
                    try blocks.append(allocator, split.right);
                    try blocks.append(allocator, split.left);
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
                        const s_i: isize = @intCast(signatures.axes[axis].items[i]);
                        const s_i_plus_one: isize = @intCast(signatures.axes[axis].items[i + 1]);
                        const s_i_minus_one: isize = @intCast(signatures.axes[axis].items[i - 1]);
                        const s_i_minus_two: isize = @intCast(signatures.axes[axis].items[i - 2]);

                        const lap_i = s_i_minus_one - 2 * s_i + s_i_plus_one;
                        const lap_i_minus_one = s_i_minus_two - 2 * s_i_minus_one + s_i;

                        // If signs are the same continue.
                        if ((lap_i > 0) == (lap_i_minus_one > 0)) {
                            continue;
                        }

                        found_inflection = true;

                        var amount = lap_i - lap_i_minus_one;
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

                    const split = split_boxes(subblock.origin, lower_bounds, upper_bounds, inflection_axis, inflection_index);
                    try blocks.append(allocator, split.right);
                    try blocks.append(allocator, split.left);
                    continue :stack_pop;
                }

                // If we find no inflection points, we simply split
                // down the middle of the longest axis.

                const mid_axis = longest;
                const mid_index = (upper_bounds[mid_axis] + lower_bounds[mid_axis]) / 2;

                const split = split_boxes(subblock.origin, lower_bounds, upper_bounds, mid_axis, mid_index);
                try blocks.append(allocator, split.right);
                try blocks.append(allocator, split.left);
            }
        }

        fn split_boxes(origin: [N]usize, lower: [N]usize, upper: [N]usize, axis: usize, index: usize) struct { left: IndexBox, right: IndexBox } {
            var left: IndexBox = undefined;
            var right: IndexBox = undefined;

            for (0..N) |i| {
                left.origin[i] = origin[i] + lower[i];
                left.widths[i] = upper[i] - lower[i];
                right.origin[i] = origin[i] + lower[i];
                right.widths[i] = upper[i] - lower[i];
            }

            left.origin[axis] = origin[axis] + lower[axis];
            left.widths[axis] = index - lower[axis];
            right.origin[axis] = origin[axis] + index;
            right.widths[axis] = upper[axis] - index;

            return .{
                .left = left,
                .right = right,
            };
        }
    };
}

/// An N-dimensional array of signatures for each axis.
fn Signatures(comptime N: usize) type {
    return struct {
        axes: [N]std.ArrayListUnmanaged(usize),

        const Self = @This();
        const IndexBox = Box(N, usize);

        pub fn initCapacity(allocator: std.mem.Allocator, size: [N]usize) !Self {
            // Allocate signature array lists
            var signatures: [N]std.ArrayListUnmanaged(usize) = undefined;

            errdefer {
                for (0..N) |axis| {
                    signatures[axis].deinit(allocator);
                }
            }

            for (0..N) |axis| {
                signatures[axis] = try std.ArrayListUnmanaged(usize).initCapacity(allocator, size[axis]);
            }

            return .{
                .axes = signatures,
            };
        }

        pub fn computeAssumeCapacity(self: *Self, tiles: Tiles(N), subblock: IndexBox) void {
            // Clear existing data
            for (0..N) |axis| {
                self.axes[axis].clearRetainingCapacity();
            }

            // Build space and subspace
            const subspace = IndexSpace(N){ .size = subblock.widths };
            const space = IndexSpace(N){ .size = tiles.size };

            // Fill signatures
            for (0..N) |axis| {
                for (0..subspace.size[axis]) |i| {
                    var signature: usize = 0;

                    var iterator = subspace.cartesianSliceIndices(axis, i);

                    while (iterator.next()) |local| {
                        var global: [N]usize = undefined;

                        for (0..N) |j| {
                            global[j] = subblock.origin[j] + local[j];
                        }

                        const linear = space.linearFromCartesian(global);

                        if (tiles.tagged.items[linear]) {
                            signature += 1;
                        }
                    }

                    self.axes[axis].appendAssumeCapacity(signature);
                }
            }
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            for (0..N) |axis| {
                self.axes[axis].deinit(allocator);
            }
        }
    };
}

test "tiles basic" {
    const expect = std.testing.expect;
    const eql = std.mem.eql;

    const space = IndexSpace(3){ .size = [3]usize{ 5, 5, 5 } };

    var tiles: Tiles(3) = try Tiles(3).init(std.testing.allocator, space.size);
    defer tiles.deinit(std.testing.allocator);

    // Fill tagging data.

    const tag_array = [_][3]usize{
        [_]usize{ 0, 0, 0 },
        [_]usize{ 3, 2, 1 },
        [_]usize{ 0, 1, 3 },
    };

    tiles.tagMany(&tag_array);

    var signatures = try Signatures(3).initCapacity(std.testing.allocator, tiles.size);
    defer signatures.deinit(std.testing.allocator);

    signatures.computeAssumeCapacity(tiles, .{
        .origin = [3]usize{ 0, 0, 0 },
        .widths = [3]usize{ 5, 5, 5 },
    });

    try expect(eql(usize, signatures.axes[0].items, &[5]usize{ 2, 0, 0, 1, 0 }));
    try expect(eql(usize, signatures.axes[1].items, &[5]usize{ 1, 1, 1, 0, 0 }));
    try expect(eql(usize, signatures.axes[2].items, &[5]usize{ 1, 1, 0, 1, 0 }));

    const efficiency = tiles.efficiency(.{
        .origin = [3]usize{ 0, 0, 0 },
        .widths = [3]usize{ 4, 4, 4 },
    });

    try expect(efficiency == 3.0 / 64.0);
}

test "tile partitioning" {
    const expect = std.testing.expect;
    const eql = std.mem.eql;

    var tiles: Tiles(2) = try Tiles(2).init(std.testing.allocator, [2]usize{ 5, 5 });
    defer tiles.deinit(std.testing.allocator);

    const tag_array = [_][2]usize{
        [_]usize{ 2, 2 },
        [_]usize{ 3, 2 },
        [_]usize{ 2, 3 },
        [_]usize{ 3, 3 },
    };

    tiles.tagMany(&tag_array);

    var blocks = std.ArrayListUnmanaged(Box(2, usize)){};
    defer blocks.deinit(std.testing.allocator);

    try tiles.partition(std.testing.allocator, 4, 0.7, &blocks);

    const box: Box(2, usize) = blocks.items[0];
    try expect(eql(usize, &box.origin, &[2]usize{ 2, 2 }));
    try expect(eql(usize, &box.widths, &[2]usize{ 2, 2 }));
}
