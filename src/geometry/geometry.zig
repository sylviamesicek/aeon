const std = @import("std");

const block = @import("block.zig");

pub const IndexSpace = @import("index.zig").IndexSpace;
pub const Box = @import("box.zig").Box;
pub const Block = block.Block;
pub const Signatures = block.Signatures;

/// Represents one level of a mesh. This includes an array of
/// rectangular blocks that make up that particular level (which needs
/// not extend over the whole domain), and information for transforming
/// between index space and physical space.
///
/// TODO: Rework init/deinit/partition API to be more zig-ish.
pub fn Geometry(comptime N: usize) type {
    return struct {
        // Fields
        allocator: std.mem.Allocator,
        physical_bounds: Box(N, f64),
        index_space: IndexSpace(N),
        tile_width: usize,
        ghost_width: usize,
        blocks: BlockList,
        dof_offsets: OffsetList,

        // Aliases
        const Self = @This();
        const BlockList = std.ArrayListUnmanaged(Box(N, usize));
        const OffsetList = std.ArrayListUnmanaged(usize);

        /// Geometry
        pub fn init(allocator: std.mem.Allocator, physical_bounds: Box(N, f64)) Self {
            return .{
                .allocator = allocator,
                .physical_bounds = physical_bounds,
                .index_space = .{ .size = [1]usize{0} ** N },
                .tile_width = 1,
                .ghost_width = 1,
                .blocks = BlockList{},
                .dof_offsets = OffsetList{},
            };
        }

        pub fn deinit(self: *Self) void {
            self.blocks.deinit(self.allocator);
            self.dof_offsets.deinit(self.allocator);
        }

        pub const PartitionConfig = struct {
            tile_width: usize,
            max_tiles: usize,
            ghost_width: usize,
            efficiency: f64 = 0.7,
        };

        pub fn partition(self: *Self, tagged: Block(N, bool), config: PartitionConfig) !void {
            self.index_space = .{ .size = tagged.size };
            self.tile_width = config.tile_width;
            self.ghost_width = config.ghost_width;

            // Clear blocks and dof offsets.
            self.blocks.clearRetainingCapacity();
            self.dof_offsets.clearRetainingCapacity();

            // Allocate signatures
            var signatures: Signatures(N) = try Signatures(N).initCapacity(self.allocator, tagged.size);
            defer signatures.deinit(self.allocator);

            // Build stack of subblocks
            var stack = std.ArrayListUnmanaged(Box(N, usize)){};
            defer stack.deinit(self.allocator);

            try stack.append(self.allocator, .{
                .origin = [1]usize{0} ** N,
                .widths = tagged.size,
            });

            stack_pop: while (stack.popOrNull()) |subblock| {
                // Remove edges
                signatures.computeAssumeCapacity(tagged, subblock);

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
                var culled_subblock: Box(N, usize) = undefined;

                for (0..N) |axis| {
                    culled_subblock.origin[axis] = subblock.origin[axis] + lower_bounds[axis];
                    culled_subblock.widths[axis] = upper_bounds[axis] - lower_bounds[axis];
                }

                const culled_space: IndexSpace(N) = .{ .size = culled_subblock.widths };

                // Check efficiency and maximum block sidelength
                const efficiency = tagged.efficiency(culled_subblock);
                const longest = culled_space.longestAxis();

                // If these checks pass, we can add this cell as a new block.
                if (efficiency >= config.efficiency and culled_space.size[longest] <= config.max_tiles) {
                    try self.blocks.append(self.allocator, culled_subblock);
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
                    try self.blocks.append(self.allocator, split.right);
                    try self.blocks.append(self.allocator, split.left);
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
                    try self.blocks.append(self.allocator, split.right);
                    try self.blocks.append(self.allocator, split.left);
                    continue :stack_pop;
                }

                // If we find no inflection points, we simply split
                // down the middle of the longest axis.

                const mid_axis = longest;
                const mid_index = (upper_bounds[mid_axis] + lower_bounds[mid_axis]) / 2;

                const split = split_boxes(subblock.origin, lower_bounds, upper_bounds, mid_axis, mid_index);
                try self.blocks.append(self.allocator, split.right);
                try self.blocks.append(self.allocator, split.left);
            }
        }

        fn split_boxes(origin: [N]usize, lower: [N]usize, upper: [N]usize, axis: usize, index: usize) struct { left: Box(N, usize), right: Box(N, usize) } {
            var left: Box(N, usize) = undefined;
            var right: Box(N, usize) = undefined;

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

test {
    _ = @import("block.zig");
    _ = @import("box.zig");
    _ = @import("index.zig");
}

test "geometry" {
    const expect = std.testing.expect;
    const eql = std.mem.eql;

    var geometry = Geometry(2).init(std.testing.allocator, .{
        .origin = [2]f64{ 0.0, 0.0 },
        .widths = [2]f64{ 1.0, 1.0 },
    });
    defer geometry.deinit();

    const space = IndexSpace(2){ .size = [2]usize{ 5, 5 } };

    var tagged: Block(2, bool) = try Block(2, bool).init(std.testing.allocator, space.size, false);
    defer tagged.deinit();

    const tag_array = [_][2]usize{
        [_]usize{ 2, 2 },
        [_]usize{ 3, 2 },
        [_]usize{ 2, 3 },
        [_]usize{ 3, 3 },
    };

    for (tag_array) |cart| {
        tagged.data.items[space.cartesianToLinear(cart)] = true;
    }

    try geometry.partition(tagged, .{
        .tile_width = 1,
        .max_tiles = 4,
        .ghost_width = 2,
        .efficiency = 0.7,
    });

    const box: Box(2, usize) = geometry.blocks.items[0];
    try expect(eql(usize, &box.origin, &[2]usize{ 2, 2 }));
    try expect(eql(usize, &box.widths, &[2]usize{ 2, 2 }));
}
