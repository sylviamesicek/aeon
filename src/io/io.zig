//! A module for interacting with input and output of data files. This includes
//! support for interacting with and producing vtk files for use with tools like
//! Paraview and VisIt.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayListUnmanaged = std.ArrayListUnmanaged;

// Submodules

const vtkio = @import("vtkio.zig");
const VtuMeshOutput = vtkio.VtuMeshOutput;
const VtkCellType = vtkio.VtkCellType;

// Module imports

const basis = @import("../basis/basis.zig");
const NodeSpace = basis.NodeSpace;
const StencilSpace = basis.StencilSpace;

const dofs = @import("../dofs/dofs.zig");
const DofUtils = dofs.DofUtils;

const geometry = @import("../geometry/geometry.zig");
const IndexSpace = geometry.IndexSpace;

const index = @import("../index.zig");
const Index = index.Index;

const meshes = @import("../mesh/mesh.zig");

const system = @import("../system.zig");
const SystemSlice = system.SystemSlice;
const SystemSliceConst = system.SystemSliceConst;
const isSystem = system.isSystem;

/// A namespace for outputting data defined on a mesh.
pub fn DataOut(comptime N: usize) type {
    return struct {
        const Mesh = meshes.Mesh(N);

        // Temporary
        // TODO fix to print all exposed blocks

        pub fn writeVtkLevel(comptime System: type, allocator: Allocator, mesh: *const Mesh, level_id: usize, sys: SystemSliceConst(System), out_stream: anytype) !void {
            if (comptime !isSystem(System)) {
                @compileError("System must satisfy isSystem trait.");
            }

            const field_count = comptime std.enums.values(System).len;

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

            const level = mesh.levels[level_id];

            for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                const stencil_space = DofUtils(N, 0).blockStencilSpace(mesh, block_id);

                const cell_size = stencil_space.size;
                const point_size = Index(N).add(cell_size, Index(N).splat(1));

                const cell_space = IndexSpace(N).fromSize(cell_size);
                const point_space = IndexSpace(N).fromSize(point_size);

                const point_offset: usize = positions.items.len / N;

                try positions.ensureUnusedCapacity(allocator, N * point_space.total());
                try vertices.ensureUnusedCapacity(allocator, cell_type.nvertices() * cell_space.total());

                // Fill positions and vertices
                var points = point_space.cartesianIndices();

                while (points.next()) |point| {
                    const pos = stencil_space.vertexPosition(Index.toSigned(point));
                    for (0..N) |i| {
                        positions.appendAssumeCapacity(pos[i]);
                    }
                }

                if (N == 1) {
                    var cells = cell_space.cartesianIndices();

                    while (cells.next()) |cell| {
                        const v1: usize = point_space.linearFromCartesian(cell);
                        const v2: usize = point_space.linearFromCartesian(Index.add(cell, Index.splat(1)));

                        vertices.appendAssumeCapacity(point_offset + v1);
                        vertices.appendAssumeCapacity(point_offset + v2);
                    }
                } else if (N == 2) {
                    var cells = cell_space.cartesianIndices();

                    while (cells.next()) |cell| {
                        const v1: usize = point_space.linearFromCartesian(cell);
                        const v2: usize = point_space.linearFromCartesian(Index.add(cell, [2]usize{ 0, 1 }));
                        const v3: usize = point_space.linearFromCartesian(Index.add(cell, [2]usize{ 1, 1 }));
                        const v4: usize = point_space.linearFromCartesian(Index.add(cell, [2]usize{ 1, 0 }));

                        vertices.appendAssumeCapacity(point_offset + v1);
                        vertices.appendAssumeCapacity(point_offset + v2);
                        vertices.appendAssumeCapacity(point_offset + v3);
                        vertices.appendAssumeCapacity(point_offset + v4);
                    }
                } else if (N == 3) {
                    var cells = cell_space.cartesianIndices();

                    while (cells.next()) |cell| {
                        const v1: usize = point_space.linearFromCartesian(cell);
                        const v2: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 0, 1, 0 }));
                        const v3: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 1, 0 }));
                        const v4: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 0, 0 }));
                        const v5: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 0, 0, 1 }));
                        const v6: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 0, 1, 3 }));
                        const v7: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 1, 3 }));
                        const v8: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 0, 3 }));

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
            }

            inline for (comptime std.enums.values(System), 0..) |field, idx| {
                try fields[idx].appendSlice(sys.field(field));
            }

            var grid: VtuMeshOutput = try VtuMeshOutput.init(allocator, .{
                .points = positions.items,
                .vertices = vertices.items,
                .cell_type = cell_type,
            });
            defer grid.deinit();

            inline for (comptime std.meta.fieldNames(System), 0..) |name, id| {
                try grid.addCellField(name, fields[id].items, 1);
            }

            try grid.write(out_stream);
        }

        pub fn writeVtk(comptime System: type, allocator: Allocator, mesh: *const Mesh, sys: system.SystemSliceConst(System), out_stream: anytype) !void {
            try writeVtkLevel(System, allocator, mesh, mesh.levels.len - 1, sys, out_stream);
        }

        // pub fn writeDofsToVtkOnLevel(
        //     comptime System: type,
        //     allocator: Allocator,
        //     mesh: *const Mesh,
        //     dof_map: Map,
        //     level: usize,
        //     sys: system.SystemSliceConst(System),
        //     out_stream: anytype,
        // ) !void {
        //     if (comptime !system.isSystem(System)) {
        //         @compileError("System must satisfy isSystem trait.");
        //     }

        //     const field_count = comptime std.enums.values(System).len;

        //     const vtkio = @import("../vtkio.zig");
        //     const VtuMeshOutput = vtkio.VtuMeshOutput;
        //     const VtkCellType = vtkio.VtkCellType;

        //     // Global Constants
        //     const cell_type: VtkCellType = switch (N) {
        //         1 => .line,
        //         2 => .quad,
        //         3 => .hexa,
        //         else => @compileError("Vtk Output not supported for N > 3"),
        //     };

        //     var positions: ArrayListUnmanaged(f64) = .{};
        //     defer positions.deinit(allocator);

        //     var vertices: ArrayListUnmanaged(usize) = .{};
        //     defer vertices.deinit(allocator);

        //     var fields = [1]ArrayListUnmanaged(f64){.{}} ** field_count;

        //     defer {
        //         for (&fields) |*field| {
        //             field.deinit(allocator);
        //         }
        //     }

        //     const top_level = mesh.levels[level];

        //     for (top_level.block_offset..top_level.block_offset + top_level.block_total) |block_id| {
        //         const block = mesh.blocks[block_id];
        //         _ = block;

        //         const stencil: StencilSpace = Utils.blockStencilSpace(mesh, block_id);

        //         const cell_size = stencil.cellSpace().sizeWithGhost();
        //         const point_size = Index.add(cell_size, Index.splat(1));

        //         const cell_space: IndexSpace = IndexSpace.fromSize(cell_size);
        //         const point_space: IndexSpace = IndexSpace.fromSize(point_size);

        //         const point_offset: usize = positions.items.len / N;

        //         try positions.ensureUnusedCapacity(allocator, N * point_space.total());
        //         try vertices.ensureUnusedCapacity(allocator, cell_type.nvertices() * cell_space.total());

        //         // Fill positions and vertices
        //         var points = point_space.cartesianIndices();

        //         while (points.next()) |point| {
        //             var idx = Index.toSigned(point);

        //             for (0..N) |i| {
        //                 idx[i] -= 2 * O;
        //             }

        //             const pos = stencil.vertexPosition(idx);

        //             for (0..N) |i| {
        //                 positions.appendAssumeCapacity(pos[i]);
        //             }
        //         }

        //         if (N == 1) {
        //             var cells = cell_space.cartesianIndices();

        //             while (cells.next()) |cell| {
        //                 const v1: usize = point_space.linearFromCartesian(cell);
        //                 const v2: usize = point_space.linearFromCartesian(Index.add(cell, Index.splat(1)));

        //                 vertices.appendAssumeCapacity(point_offset + v1);
        //                 vertices.appendAssumeCapacity(point_offset + v2);
        //             }
        //         } else if (N == 2) {
        //             var cells = cell_space.cartesianIndices();

        //             while (cells.next()) |cell| {
        //                 const v1: usize = point_space.linearFromCartesian(cell);
        //                 const v2: usize = point_space.linearFromCartesian(Index.add(cell, [2]usize{ 0, 1 }));
        //                 const v3: usize = point_space.linearFromCartesian(Index.add(cell, [2]usize{ 1, 1 }));
        //                 const v4: usize = point_space.linearFromCartesian(Index.add(cell, [2]usize{ 1, 0 }));

        //                 vertices.appendAssumeCapacity(point_offset + v1);
        //                 vertices.appendAssumeCapacity(point_offset + v2);
        //                 vertices.appendAssumeCapacity(point_offset + v3);
        //                 vertices.appendAssumeCapacity(point_offset + v4);
        //             }
        //         } else if (N == 3) {
        //             var cells = cell_space.cartesianIndices();

        //             while (cells.next()) |cell| {
        //                 const v1: usize = point_space.linearFromCartesian(cell);
        //                 const v2: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 0, 1, 0 }));
        //                 const v3: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 1, 0 }));
        //                 const v4: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 0, 0 }));
        //                 const v5: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 0, 0, 1 }));
        //                 const v6: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 0, 1, 3 }));
        //                 const v7: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 1, 3 }));
        //                 const v8: usize = point_space.linearFromCartesian(Index.add(cell, [3]usize{ 1, 0, 3 }));

        //                 vertices.appendAssumeCapacity(point_offset + v1);
        //                 vertices.appendAssumeCapacity(point_offset + v2);
        //                 vertices.appendAssumeCapacity(point_offset + v3);
        //                 vertices.appendAssumeCapacity(point_offset + v4);
        //                 vertices.appendAssumeCapacity(point_offset + v5);
        //                 vertices.appendAssumeCapacity(point_offset + v6);
        //                 vertices.appendAssumeCapacity(point_offset + v7);
        //                 vertices.appendAssumeCapacity(point_offset + v8);
        //             }
        //         }

        //         for (&fields) |*field| {
        //             try field.ensureUnusedCapacity(allocator, cell_space.total());
        //         }

        //         const block_sys = sys.slice(
        //             dof_map.offset(block_id),
        //             dof_map.total(block_id),
        //         );

        //         inline for (comptime std.enums.values(System), 0..) |field, idx| {
        //             fields[idx].appendSliceAssumeCapacity(block_sys.field(field));
        //         }
        //     }

        //     var grid: VtuMeshOutput = try VtuMeshOutput.init(allocator, .{
        //         .points = positions.items,
        //         .vertices = vertices.items,
        //         .cell_type = cell_type,
        //     });
        //     defer grid.deinit();

        //     inline for (comptime std.meta.fieldNames(System), 0..) |name, id| {
        //         try grid.addCellField(name, fields[id].items, 1);
        //     }

        //     try grid.write(out_stream);
        // }
    };
}
