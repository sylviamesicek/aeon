//! A module for interacting with input and output of data files. This includes
//! support for interacting with and producing vtk files for use with tools like
//! Paraview and VisIt.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayListUnmanaged = std.ArrayListUnmanaged;
const assert = std.debug.assert;

// Submodules

const vtkio = @import("vtkio.zig");
const VtuMeshOutput = vtkio.VtuMeshOutput;
const VtkCellType = vtkio.VtkCellType;

// Module imports

const basis = @import("../basis/basis.zig");
const common = @import("../common/common.zig");
const geometry = @import("../geometry/geometry.zig");
const tree = @import("../tree/tree.zig");

const System = common.System;
const SystemConst = common.SystemConst;
const isSystemTag = common.isSystemTag;
const null_index = tree.null_index;

/// A namespace for outputting data defined on a mesh.
pub fn DataOut(comptime N: usize, comptime M: usize) type {
    return struct {
        const Mesh = tree.TreeMesh(N);
        const NodeWorker = tree.NodeWorker(N, M);

        const IndexMixin = geometry.IndexMixin(N);
        const IndexSpace = geometry.IndexSpace(N);
        const NodeSpace = common.NodeSpace(N, M);

        pub fn writeVtk(
            comptime Tag: type,
            allocator: Allocator,
            worker: *const NodeWorker,
            sys: SystemConst(Tag),
            out_stream: anytype,
        ) !void {
            if (comptime !isSystemTag(Tag)) {
                @compileError("System must satisfy isSystemTag trait.");
            }

            const mesh = worker.mesh;
            const manager = worker.manager;

            assert(sys.len == worker.numNodes());

            const field_count = comptime std.enums.values(Tag).len;

            // Global Constants
            const cell_type: VtkCellType = switch (N) {
                1 => .line,
                2 => .quad,
                3 => .hexa,
                else => @compileError("Vtk Output not supported for N > 3"),
            };

            var positions: ArrayListUnmanaged(f64) = .{};
            defer positions.deinit(allocator);

            var vertices: ArrayListUnmanaged(usize) = .{};
            defer vertices.deinit(allocator);

            var fields = [1]ArrayListUnmanaged(f64){.{}} ** field_count;

            defer {
                for (&fields) |*field| {
                    field.deinit(allocator);
                }
            }

            for (0..manager.numBlocks()) |block_id| {
                const block = manager.blockFromId(block_id);
                const node_space = NodeSpace{
                    .bounds = block.bounds,
                    .size = IndexMixin.mul(block.size, manager.cell_size),
                };

                const block_sys = sys.slice(worker.map.offset(block_id), worker.map.size(block_id));

                var mcells = IndexSpace.fromSize(block.size).cartesianIndices();

                while (mcells.next()) |mcell| {
                    // Get Cell ID
                    const cell_id = manager.cellFromBlock(block_id, mcell);

                    if (mesh.cells.items(.children)[cell_id] == null_index) {
                        continue;
                    }

                    // This is a leaf
                    const origin = IndexMixin.mul(block.size, manager.cell_size);

                    // Append Point Data

                    const point_offset: usize = positions.items.len / N;

                    const point_size = IndexMixin.add(manager.cell_size, IndexMixin.splat(1));
                    const point_space = IndexSpace.fromSize(point_size);
                    const cell_space = IndexSpace.fromSize(manager.cell_size);

                    var points = point_space.cartesianIndices();

                    while (points.next()) |point| {
                        const position = node_space.vertexPosition(IndexMixin.toSigned(IndexMixin.add(origin, point)));
                        for (0..N) |i| {
                            try positions.append(allocator, position[i]);
                        }
                    }

                    // Append Vertex data

                    if (N == 1) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(IndexMixin.add(cell, IndexMixin.splat(1)));

                            try vertices.append(allocator, point_offset + v1);
                            try vertices.append(allocator, point_offset + v2);
                        }
                    } else if (N == 2) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [2]usize{ 0, 1 }));
                            const v3: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [2]usize{ 1, 1 }));
                            const v4: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [2]usize{ 1, 0 }));

                            try vertices.append(allocator, point_offset + v1);
                            try vertices.append(allocator, point_offset + v2);
                            try vertices.append(allocator, point_offset + v3);
                            try vertices.append(allocator, point_offset + v4);
                        }
                    } else if (N == 3) {
                        var cells = cell_space.cartesianIndices();

                        while (cells.next()) |cell| {
                            const v1: usize = point_space.linearFromCartesian(cell);
                            const v2: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [3]usize{ 0, 1, 0 }));
                            const v3: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [3]usize{ 1, 1, 0 }));
                            const v4: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [3]usize{ 1, 0, 0 }));
                            const v5: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [3]usize{ 0, 0, 1 }));
                            const v6: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [3]usize{ 0, 1, 3 }));
                            const v7: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [3]usize{ 1, 1, 3 }));
                            const v8: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [3]usize{ 1, 0, 3 }));

                            try vertices.append(allocator, point_offset + v1);
                            try vertices.append(allocator, point_offset + v2);
                            try vertices.append(allocator, point_offset + v3);
                            try vertices.append(allocator, point_offset + v4);
                            try vertices.append(allocator, point_offset + v5);
                            try vertices.append(allocator, point_offset + v6);
                            try vertices.append(allocator, point_offset + v7);
                            try vertices.append(allocator, point_offset + v8);
                        }
                    }

                    // Append Field data
                    var cells = cell_space.cartesianIndices();

                    while (cells.next()) |cell| {
                        const node = IndexMixin.add(origin, cell);
                        const linear = node_space.cellSpace().linearFromCartesian(node);

                        inline for (comptime std.enums.values(Tag), 0..) |field, idx| {
                            try fields[idx].append(allocator, block_sys.field(field)[linear]);
                        }
                    }
                }
            }

            var output: VtuMeshOutput = try VtuMeshOutput.init(allocator, .{
                .points = positions.items,
                .vertices = vertices.items,
                .cell_type = cell_type,
            });
            defer output.deinit();

            inline for (comptime std.meta.fieldNames(Tag), 0..) |name, id| {
                try output.addCellField(name, fields[id].items, 1);
            }

            try output.write(out_stream);
        }
    };
}
