const std = @import("std");
const Allocator = std.mem.Allocator;

const basis = @import("../basis/basis.zig");
const lac = @import("../lac/lac.zig");
const system = @import("../system.zig");
const meshes = @import("../mesh/mesh.zig");
const geometry = @import("../geometry/geometry.zig");
const index = @import("../index.zig");

const boundary = @import("boundary.zig");
const dofs = @import("dofs.zig");
const operator = @import("operator.zig");

/// A multigrid based elliptic solver which uses the given base solver to approximate the solution
/// on the lowest level of the mesh.
pub fn MultigridSolver(comptime N: usize, comptime O: usize, comptime BaseSolver: type) type {
    if (comptime !lac.isLinearSolver(BaseSolver)) {
        @compileError("Base solver must satisfy the a linear solver requirement.");
    }

    return struct {
        mesh: *const Mesh,
        block_map: []const usize,
        base_solver: *const BaseSolver,
        max_iters: usize,
        tolerance: f64,

        const Self = @This();
        const Mesh = meshes.Mesh(N);
        const DofUtils = dofs.DofUtils(N, O);
        const BoundaryUtils = boundary.BoundaryUtils(N, O);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);
        const CellSpace = basis.CellSpace(N, O, O);
        const StencilSpace = basis.StencilSpace(N, O, O);
        const SystemSlice = system.SystemSlice;
        const SystemSliceConst = system.SystemSliceConst;

        pub fn init(mesh: *const Mesh, block_map: []const usize, base_solver: *BaseSolver, max_iters: usize, tolerance: f64) Self {
            return .{
                .mesh = mesh,
                .block_map = block_map,
                .base_solver = base_solver,
                .max_iters = max_iters,
                .tolerance = tolerance,
            };
        }

        pub fn deinit(_: *Self) void {}

        pub fn solve(
            self: *Self,
            allocator: Allocator,
            oper: anytype,
            x: SystemSlice(@TypeOf(oper).System),
            b: SystemSliceConst(@TypeOf(oper).System),
            ctx: SystemSliceConst(@TypeOf(oper).Context),
        ) !void {
            // Alias
            const T = @TypeOf(oper);

            // Check trait constraints
            if (comptime !(operator.isMeshOperator(N, O)(T))) {
                @compileError("Oper must satisfy isMeshOperator traits.");
            }

            if (comptime std.meta.fields(T.System).len != 1) {
                @compileError("The multigrid solver only supports systems with 1 field currently.");
            }

            // Maximum number of dofs to store a window
            const max_window_dofs: usize = DofUtils.maxWindowDofs(self.mesh);

            // Allocate temporary vector to store scratch data (including tau correction).
            const tau = try SystemSlice(T.System).init(allocator, self.mesh.cell_total);
            defer tau.deinit(allocator);

            // Reset tau correction to 0.0
            @memset(tau, 0.0);

            // Allocate windows
            const ctx_window = try SystemSlice(T.Context).init(allocator, max_window_dofs);
            defer ctx_window.deinit(allocator);

            const x_window = try SystemSlice(T.System).init(allocator, max_window_dofs);
            defer x_window.deinit(allocator);

            const d_window = try SystemSlice(T.System).init(allocator, max_window_dofs);
            defer d_window.deinit(allocator);

            const b_window = try SystemSlice(T.System).init(allocator, max_window_dofs);
            defer b_window.deinit(allocator);

            // Run iterations
            var iteration = 0;

            while (iteration < self.max_iters) : (iteration += 1) {
                // Recurse down the mesh to the base level
                for (1..self.mesh.active_levels) |reverse_level| {
                    const level: usize = self.mesh.active_levels - 1 - reverse_level;
                    const target = self.mesh.getLevel(level);

                    // Perform smoothing
                    for (0..target.blockTotal()) |block| {
                        const block_dofs = DofUtils.windowDofs(self.mesh, level, block);

                        const ctx_slice = ctx_window.slice(0, block_dofs);
                        const x_slice = x_window.slice(0, block_dofs);
                        const b_slice = b_window.slice(0, block_dofs);
                        const d_slice = d_window.slice(0, block_dofs);

                        const cell_offset = self.mesh.blockCellOffset(level, block);
                        const cell_total = self.mesh.blockCellTotal(level, block);
                        const block_b = b.slice(cell_offset, cell_total);
                        const block_tau = tau.slice(cell_offset, cell_total);

                        const stencil_space = DofUtils.blockStencilSpace(self.mesh, level, block);
                        const cell_space = stencil_space.cellSpace();

                        // Fill right hand side
                        var cells = cell_space.cells();
                        var linear: usize = 0;

                        while (cells.next()) |cell| : (linear += 1) {
                            inline for (comptime std.enums.values(T.System)) |field| {
                                cell_space.setValue(
                                    cell,
                                    b_slice.field(field),
                                    block_b.field(field)[linear] + block_tau.field(field)[linear],
                                );
                            }
                        }

                        // Fill ctx slice
                        DofUtils.fillWindow(
                            self.mesh,
                            self.block_map,
                            level,
                            block,
                            DofUtils.operContextBoundary(oper),
                            ctx_slice,
                            ctx.toConst(),
                        );
                        // Fill x slice
                        DofUtils.fillWindow(
                            self.mesh,
                            self.block_map,
                            level,
                            block,
                            DofUtils.operSystemBoundary(oper),
                            x_slice,
                            x.toConst(),
                        );
                        // Run smoothing
                        DofUtils.jacobi(
                            stencil_space,
                            oper,
                            d_slice,
                            x_slice.toConst(),
                            b_slice.toConst(),
                            ctx_slice.toConst(),
                        );
                        // Copy smoothed result into solution vector.
                        DofUtils.copyFrom(
                            T.System,
                            self.mesh,
                            level,
                            block,
                            x,
                            d_slice,
                        );
                        // Restrict updated vector.
                        DofUtils.restrictFrom(
                            T.System,
                            self.mesh,
                            self.block_map,
                            level,
                            block,
                            x,
                            d_slice,
                        );
                    }

                    // Compute tau correction
                    for (0..target.blockTotal()) |block| {
                        const block_dofs = DofUtils.windowDofs(self.mesh, level, block);

                        const ctx_slice = ctx_window.slice(0, block_dofs);
                        const x_slice = x_window.slice(0, block_dofs);
                        const d_slice = d_window.slice(0, block_dofs);

                        const stencil_space = DofUtils.blockStencilSpace(self.mesh, level, block);

                        // TODO write DofUtils method which does this.

                        // Fill both x and ctx fully.
                        DofUtils.fillWindowFull(
                            self.mesh,
                            self.block_map,
                            level,
                            block,
                            DofUtils.operSystemBoundary(oper),
                            x_slice,
                            x.toConst(),
                        );

                        DofUtils.fillWindowFull(
                            self.mesh,
                            self.block_map,
                            level,
                            block,
                            DofUtils.operContextBoundary(oper),
                            ctx_slice,
                            ctx.toConst(),
                        );

                        // Apply operator
                        DofUtils.applyFull(
                            stencil_space,
                            oper,
                            d_slice,
                            x_slice.toConst(),
                            ctx_slice.toConst(),
                        );

                        // Restrict to tau
                        DofUtils.restrictFrom(
                            T.System,
                            self.mesh,
                            self.block_map,
                            level,
                            block,
                            tau,
                            d_slice,
                        );
                    }
                }
            }

            const base = self.mesh.getLevel(0);

            const field: T.System = comptime std.enums.values(T.System)[0];
            const x_field: []f64 = x.field(field)[0..base.cell_total];
            const b_field: []const f64 = b.field(field)[0..base.cell_total];

            const base_window_dofs = DofUtils.windowDofs(self.mesh, 0, 0);

            // Solve system using the base solver
            const base_linear_map: BaseLinearMap(T) = .{
                .self = self,
                .oper = oper,
                .ctx = ctx,
                .ctx_window = ctx_window.slice(0, base_window_dofs),
                .x_window = x_window.slice(0, base_window_dofs),
            };

            self.base_solver.solve(base_linear_map, x_field, b_field);
        }

        fn BaseLinearMap(comptime T: type) type {
            const field_name: []const u8 = comptime std.meta.fieldNames(T.System)[0];

            return struct {
                self: *const Self,
                oper: T,
                tau: SystemSlice(T.System),
                ctx: SystemSliceConst(T.Context),
                ctx_slice: SystemSlice(T.Context),
                x_slice: SystemSlice(T.System),

                pub fn apply(wrapper: *const @This(), output: []f64, input: []const f64) void {
                    const mesh = wrapper.self.mesh;

                    const block_map = wrapper.self.block_map;
                    const ctx_slice = wrapper.ctx_slice;
                    const x_slice = wrapper.x_slice;

                    // Aliases
                    const stencil_space = DofUtils.blockStencilSpace(mesh, 0, 0);
                    const cell_space = stencil_space.cellSpace();

                    // Fill base boundary and inner
                    {
                        var cells = stencil_space.cellSpace().cells();
                        var linear: usize = 0;

                        while (cells.next()) |cell| : (linear += 1) {
                            inline for (comptime std.enums.values(T.System)) |field| {
                                cell_space.setValue(cell, x_slice.field(field), input[linear]);
                            }
                        }

                        BoundaryUtils.fillBoundary(
                            O,
                            stencil_space,
                            DofUtils.operSystemBoundary(wrapper.oper),
                            x_slice,
                        );

                        DofUtils.fillWindow(
                            mesh,
                            block_map,
                            0,
                            0,
                            DofUtils.operContextBoundary(wrapper.oper),
                            ctx_slice,
                            wrapper.ctx,
                        );
                    }

                    // Apply operator and store in output.
                    {
                        var cells = stencil_space.cellSpace().cells();
                        var linear: usize = 0;

                        while (cells.next()) |cell| : (linear += 1) {
                            const engine = operator.EngineType(N, O, T){
                                .inner = .{
                                    .space = stencil_space,
                                    .cell = cell,
                                },
                                .ctx = ctx_slice,
                                .sys = x_slice,
                            };

                            const app = wrapper.oper.apply(engine);
                            output[linear] = @field(app, field_name);
                        }
                    }
                }

                // var iterations: usize = 0;

                pub fn callback(_: *const @This(), iteration: usize, residual: f64, _: []const f64) void {
                    std.debug.print("Iteration: {}, Residual: {}\n", .{ iteration, residual });

                    // const file_name = std.fmt.allocPrint(solver.self.mesh.gpa, "output/elliptic_iteration{}.vtu", .{iterations}) catch {
                    //     unreachable;
                    // };

                    // const file = std.fs.cwd().createFile(file_name, .{}) catch {
                    //     unreachable;
                    // };
                    // defer file.close();

                    // DofUtils.writeVtk(solver.self.mesh.gpa, solver.self.mesh, .{ .metric = x }, file.writer()) catch {
                    //     unreachable;
                    // };

                    // iterations += 1;
                }
            };
        }
    };
}
