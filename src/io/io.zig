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
const bsamr = @import("../bsamr/bsamr.zig");
const geometry = @import("../geometry/geometry.zig");
const methods = @import("../methods/methods.zig");
const nodes = @import("../nodes/nodes.zig");

const System = methods.System;
const SystemConst = methods.SystemConst;
const isSystemTag = methods.isSystemTag;

/// A namespace for outputting data defined on a mesh.
pub fn DataOut(comptime N: usize, comptime M: usize) type {
    return struct {
        const Mesh = bsamr.Mesh(N);
        const DofManager = bsamr.DofManager(N, M);
        const IndexMixin = geometry.IndexMixin(N);
        const IndexSpace = geometry.IndexSpace(N);
        const NodeSpace = nodes.NodeSpace(N, M);

        pub fn writeVtkLevel(
            comptime Tag: type,
            allocator: Allocator,
            grid: *const Mesh,
            dofs: *const DofManager,
            level_id: usize,
            sys: SystemConst(Tag),
            out_stream: anytype,
        ) !void {
            if (comptime !isSystemTag(Tag)) {
                @compileError("System must satisfy isSystemTag trait.");
            }

            assert(sys.len == dofs.numCells());

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

            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                const node_space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));
                const physical_bounds = grid.blockPhysicalBounds(block_id);

                const point_size = IndexMixin.add(node_space.size, IndexMixin.splat(1));

                const cell_space = IndexSpace.fromSize(node_space.size);
                const point_space = IndexSpace.fromSize(point_size);

                const point_offset: usize = positions.items.len / N;

                try positions.ensureUnusedCapacity(allocator, N * point_space.total());
                try vertices.ensureUnusedCapacity(allocator, cell_type.nvertices() * cell_space.total());

                const block_sys = sys.slice(dofs.cell_map.offset(block_id), dofs.cell_map.total(block_id));

                // Fill positions and vertices
                var points = point_space.cartesianIndices();

                while (points.next()) |point| {
                    const pos = physical_bounds.transformPos(node_space.vertexPosition(IndexMixin.toSigned(point)));
                    for (0..N) |i| {
                        positions.appendAssumeCapacity(pos[i]);
                    }
                }

                if (N == 1) {
                    var cells = cell_space.cartesianIndices();

                    while (cells.next()) |cell| {
                        const v1: usize = point_space.linearFromCartesian(cell);
                        const v2: usize = point_space.linearFromCartesian(IndexMixin.add(cell, IndexMixin.splat(1)));

                        vertices.appendAssumeCapacity(point_offset + v1);
                        vertices.appendAssumeCapacity(point_offset + v2);
                    }
                } else if (N == 2) {
                    var cells = cell_space.cartesianIndices();

                    while (cells.next()) |cell| {
                        const v1: usize = point_space.linearFromCartesian(cell);
                        const v2: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [2]usize{ 0, 1 }));
                        const v3: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [2]usize{ 1, 1 }));
                        const v4: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [2]usize{ 1, 0 }));

                        vertices.appendAssumeCapacity(point_offset + v1);
                        vertices.appendAssumeCapacity(point_offset + v2);
                        vertices.appendAssumeCapacity(point_offset + v3);
                        vertices.appendAssumeCapacity(point_offset + v4);
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

                        vertices.appendAssumeCapacity(point_offset + v1);
                        vertices.appendAssumeCapacity(point_offset + v2);
                        vertices.appendAssumeCapacity(point_offset + v3);
                        vertices.appendAssumeCapacity(point_offset + v4);
                        vertices.appendAssumeCapacity(point_offset + v5);
                        vertices.appendAssumeCapacity(point_offset + v6);
                        vertices.appendAssumeCapacity(point_offset + v7);
                        vertices.appendAssumeCapacity(point_offset + v8);
                    }
                }

                inline for (comptime std.enums.values(Tag), 0..) |field, idx| {
                    try fields[idx].appendSlice(allocator, block_sys.field(field));
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

        pub fn writeVtk(
            comptime Tag: type,
            allocator: Allocator,
            grid: *const Mesh,
            dofs: *const DofManager,
            sys: SystemConst(Tag),
            out_stream: anytype,
        ) !void {
            return writeVtkLevel(Tag, allocator, grid, dofs, grid.levels.len - 1, sys, out_stream);
        }

        pub fn writeVtkLevelNodes(
            comptime Tag: type,
            allocator: Allocator,
            grid: *const Mesh,
            dofs: *const DofManager,
            level_id: usize,
            sys: SystemConst(Tag),
            out_stream: anytype,
        ) !void {
            if (comptime !isSystemTag(Tag)) {
                @compileError("System must satisfy isSystemTag trait.");
            }

            assert(sys.len == dofs.numNodes());

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

            const level = grid.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                const node_space = NodeSpace.fromCellSize(grid.blockCellSize(block_id));
                const physical_bounds = grid.blockPhysicalBounds(block_id);

                const point_size = IndexMixin.add(node_space.size, IndexMixin.splat(1));

                const cell_space = IndexSpace.fromSize(node_space.size);
                const point_space = IndexSpace.fromSize(point_size);

                const point_offset: usize = positions.items.len / N;

                try positions.ensureUnusedCapacity(allocator, N * point_space.total());
                try vertices.ensureUnusedCapacity(allocator, cell_type.nvertices() * cell_space.total());

                const block_sys = sys.slice(dofs.node_map.offset(block_id), dofs.node_map.total(block_id));

                // Fill positions and vertices
                var points = point_space.cartesianIndices();

                while (points.next()) |point| {
                    const pos = physical_bounds.transformPos(node_space.vertexPosition(IndexMixin.toSigned(point)));
                    for (0..N) |i| {
                        positions.appendAssumeCapacity(pos[i]);
                    }
                }

                if (N == 1) {
                    var cells = cell_space.cartesianIndices();

                    while (cells.next()) |cell| {
                        const v1: usize = point_space.linearFromCartesian(cell);
                        const v2: usize = point_space.linearFromCartesian(IndexMixin.add(cell, IndexMixin.splat(1)));

                        vertices.appendAssumeCapacity(point_offset + v1);
                        vertices.appendAssumeCapacity(point_offset + v2);
                    }
                } else if (N == 2) {
                    var cells = cell_space.cartesianIndices();

                    while (cells.next()) |cell| {
                        const v1: usize = point_space.linearFromCartesian(cell);
                        const v2: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [2]usize{ 0, 1 }));
                        const v3: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [2]usize{ 1, 1 }));
                        const v4: usize = point_space.linearFromCartesian(IndexMixin.add(cell, [2]usize{ 1, 0 }));

                        vertices.appendAssumeCapacity(point_offset + v1);
                        vertices.appendAssumeCapacity(point_offset + v2);
                        vertices.appendAssumeCapacity(point_offset + v3);
                        vertices.appendAssumeCapacity(point_offset + v4);
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

                        vertices.appendAssumeCapacity(point_offset + v1);
                        vertices.appendAssumeCapacity(point_offset + v2);
                        vertices.appendAssumeCapacity(point_offset + v3);
                        vertices.appendAssumeCapacity(point_offset + v4);
                        vertices.appendAssumeCapacity(point_offset + v5);
                        vertices.appendAssumeCapacity(point_offset + v6);
                        vertices.appendAssumeCapacity(point_offset + v7);
                        vertices.appendAssumeCapacity(point_offset + v8);
                    }
                }

                var cells = node_space.cellSpace().cartesianIndices();

                while (cells.next()) |cell| {
                    inline for (comptime std.enums.values(Tag), 0..) |field, idx| {
                        const v = node_space.value(cell, block_sys.field(field));
                        try fields[idx].append(allocator, v);
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

        pub fn writeVtkNodes(
            comptime Tag: type,
            allocator: Allocator,
            grid: *const Mesh,
            dofs: *const DofManager,
            sys: SystemConst(Tag),
            out_stream: anytype,
        ) !void {
            return writeVtkLevelNodes(Tag, allocator, grid, dofs, grid.levels.len - 1, sys, out_stream);
        }
    };
}
