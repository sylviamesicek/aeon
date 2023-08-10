const std = @import("std");

pub const IndexSpace = @import("index.zig").IndexSpace;
pub const Box = @import("box.zig").Box;
pub const Block = @import("block.zig").Block;

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
            ghost_width: usize,
            max_size: usize,
            efficiency: f64 = 0.7,
        };

        pub fn partition(self: *Self, tagged: Block(N, bool), config: PartitionConfig) !void {
            _ = config;
            self.cells = tagged.size;

            var signatures = Signatures(N).init(self.allocator, tagged);
            defer signatures.deinit();
        }

        fn compute_efficiency(tagged: Block(N, bool), part: Box(N, usize)) f64 {
            const space = IndexSpace(N){ .size = part.widths };

            const offset = part.origin;
            var indices = space.cartesianIndices();

            var n_tagged: usize = 0;
            var linear: usize = 0;

            while (indices.next()) |local| {
                var global: [N]usize = undefined;

                for (0..N) |i| {
                    global[i] = offset[i] + local[i];
                }

                if (tagged.data[linear]) {
                    n_tagged += 1;
                }

                linear += 1;
            }

            const n_total = space.total();

            const f_tagged: f64 = @floatFromInt(n_tagged);
            const f_total: f64 = @floatFromInt(n_total);

            return f_tagged / f_total;
        }
    };
}

pub fn Signatures(comptime N: usize) type {
    return struct {
        axes: [N]std.ArrayListUnmanaged(usize),

        const Self = @This();

        fn init(allocator: std.mem.Allocator, tagged: Block(N, bool)) !Self {
            const space = IndexSpace(N){ .size = tagged.size };

            // Allocate signatures
            var signatures: [N]std.ArrayListUnmanaged(usize) = undefined;

            for (0..N) |axis| {
                signatures[axis] = std.ArrayListUnmanaged(usize){};
            }

            errdefer {
                for (0..N) |axis| {
                    signatures[axis].deinit(allocator);
                }
            }

            for (0..N) |axis| {
                try signatures[axis].ensureTotalCapacity(allocator, tagged.size[axis]);
            }

            // Fill signatures
            for (0..N) |axis| {
                for (0..space.size[axis]) |i| {
                    var signature: usize = 0;

                    var iterator = space.cartesianSliceIndices(axis, i);

                    while (iterator.next()) |cart| {
                        const linear = space.cartesianToLinear(cart);
                        if (tagged.data.items[linear]) {
                            signature += 1;
                        }
                    }

                    signatures[axis].appendAssumeCapacity(signature);
                }
            }

            return .{
                .axes = signatures,
            };
        }

        fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            for (0..N) |axis| {
                self.axes[axis].deinit(allocator);
            }
        }
    };
}

test {
    _ = @import("block.zig");
    _ = @import("box.zig");
    _ = @import("index.zig");
}

test "signatures" {
    const expect = std.testing.expect;
    const eql = std.mem.eql;

    const space = IndexSpace(3){ .size = [3]usize{ 5, 5, 5 } };

    var tagged: Block(3, bool) = try Block(3, bool).init(std.testing.allocator, space.size);
    defer tagged.deinit();

    // Fill tagging data.
    for (tagged.data.items) |*tag| {
        tag.* = false;
    }

    tagged.data.items[space.cartesianToLinear([3]usize{ 0, 0, 0 })] = true;
    tagged.data.items[space.cartesianToLinear([3]usize{ 3, 2, 1 })] = true;
    tagged.data.items[space.cartesianToLinear([3]usize{ 0, 1, 3 })] = true;

    var signatures = try Signatures(3).init(std.testing.allocator, tagged);
    defer signatures.deinit(std.testing.allocator);

    try expect(eql(usize, signatures.axes[0].items, &[5]usize{ 2, 0, 0, 1, 0 }));
    try expect(eql(usize, signatures.axes[1].items, &[5]usize{ 1, 1, 1, 0, 0 }));
    try expect(eql(usize, signatures.axes[2].items, &[5]usize{ 1, 1, 0, 1, 0 }));
}

test "geometry" {
    var geometry = Geometry(3).init(std.testing.allocator, .{
        .origin = [3]f64{ 0.0, 0.0, 0.0 },
        .widths = [3]f64{ 1.0, 1.0, 1.0 },
    });
    defer geometry.deinit();
}
