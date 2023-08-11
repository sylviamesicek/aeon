const std = @import("std");
const IndexSpace = @import("index.zig").IndexSpace;
const Box = @import("box.zig").Box;

/// An N-dimensional array with element T.
pub fn Block(comptime N: usize, comptime T: type) type {
    return struct {
        size: [N]usize,
        data: std.ArrayList(T),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, size: [N]usize) !Self {
            const space = IndexSpace(N){ .size = size };
            const total = space.total();

            var data = std.ArrayList(T).init(allocator);
            errdefer data.deinit();

            try data.resize(total);

            return .{
                .size = size,
                .data = data,
            };
        }

        pub fn deinit(self: Self) void {
            self.data.deinit();
        }

        pub fn efficiency(self: Self, subblock: Box(N, usize)) f64 {
            if (T != bool) {
                @compileError("Effeciency of block can only be calculated for T == bool");
            }

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

                const linear = space.cartesianToLinear(global);

                if (self.data.items[linear]) {
                    n_tagged += 1;
                }
            }

            const f_tagged: f64 = @floatFromInt(n_tagged);
            const f_total: f64 = @floatFromInt(n_total);

            return f_tagged / f_total;
        }
    };
}

/// An N-dimensional array of signatures for each axis.
/// It has a similar API to an unmanaged array list, as
/// it is essentially just an array of N `ArrayListUnmanaged(usize)`s.
pub fn Signatures(comptime N: usize) type {
    return struct {
        axes: [N]std.ArrayListUnmanaged(usize),

        const Self = @This();

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

        pub fn computeAssumeCapacity(self: *Self, block: Block(N, bool), subblock: Box(N, usize)) void {
            // Build space and subspace
            const subspace = IndexSpace(N){ .size = subblock.widths };
            const space = IndexSpace(N){ .size = block.size };

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

                        const linear = space.cartesianToLinear(global);

                        if (block.data.items[linear]) {
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

test "block" {
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

    var signatures = try Signatures(3).initCapacity(std.testing.allocator, [3]usize{ 5, 5, 5 });
    defer signatures.deinit(std.testing.allocator);

    signatures.computeAssumeCapacity(tagged, .{
        .origin = [3]usize{ 0, 0, 0 },
        .widths = [3]usize{ 5, 5, 5 },
    });

    try expect(eql(usize, signatures.axes[0].items, &[5]usize{ 2, 0, 0, 1, 0 }));
    try expect(eql(usize, signatures.axes[1].items, &[5]usize{ 1, 1, 1, 0, 0 }));
    try expect(eql(usize, signatures.axes[2].items, &[5]usize{ 1, 1, 0, 1, 0 }));

    const efficiency = tagged.efficiency(.{
        .origin = [3]usize{ 0, 0, 0 },
        .widths = [3]usize{ 4, 4, 4 },
    });

    try expect(efficiency == 3.0 / 64.0);
}
