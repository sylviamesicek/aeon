const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const basis = @import("../basis/basis.zig");
const dofs = @import("../dofs/dofs.zig");
const lac = @import("../lac/lac.zig");
const io = @import("../io/io.zig");
const system = @import("../system.zig");
const meshes = @import("../mesh/mesh.zig");
const geometry = @import("../geometry/geometry.zig");
const index = @import("../index.zig");

/// A multigrid based elliptic solver which uses the given base solver to approximate the solution
/// on the lowest level of the mesh.
pub fn MultigridMethod(comptime N: usize, comptime O: usize, comptime BaseSolver: type) type {
    if (comptime !lac.isLinearSolver(BaseSolver)) {
        @compileError("Base solver must satisfy the a linear solver requirement.");
    }

    return struct {
        base_solver: BaseSolver,
        max_iters: usize,
        tolerance: f64,

        const Self = @This();
        const Mesh = meshes.Mesh(N);
        const DataOut = io.DataOut(N);
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const BoundaryUtils = dofs.BoundaryUtils(N, O);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);
        const StencilSpace = basis.StencilSpace(N, O);
        const SystemSlice = system.SystemSlice;
        const SystemSliceConst = system.SystemSliceConst;

        pub fn new(max_iters: usize, tolerance: f64, base_solver: BaseSolver) Self {
            return .{
                .base_solver = base_solver,
                .max_iters = max_iters,
                .tolerance = tolerance,
            };
        }

        pub fn solve(
            self: *Self,
            allocator: Allocator,
            mesh: *const Mesh,
            dof_map: DofMap,
            oper: anytype,
            x: SystemSlice(@TypeOf(oper).System),
            context: SystemSliceConst(@TypeOf(oper).Context),
            b: SystemSliceConst(@TypeOf(oper).System),
        ) !void {
            // Alias
            const T = @TypeOf(oper);

            // Check trait constraints
            if (comptime !(dofs.isSystemOperator(N, O)(T))) {
                @compileError("Oper must satisfy isMeshOperator traits.");
            }

            if (comptime std.meta.fields(T.System).len != 1) {
                @compileError("The multigrid solver only supports systems with 1 field currently.");
            }

            std.debug.print("Running Multigrid Solver\n", .{});

            const sys_field = comptime std.enums.values(T.System)[0];

            // Get total dofs
            const total_dofs = dof_map.ndofs();

            // Stores the system in a dof vector
            const sys = try SystemSlice(T.System).init(allocator, total_dofs);
            defer sys.deinit(allocator);

            for (0..mesh.blocks.len) |block_id| {
                DofUtils.copyDofsFromCells(
                    T.System,
                    mesh,
                    dof_map,
                    block_id,
                    sys,
                    x.toConst(),
                );
            }

            for (0..mesh.blocks.len) |block_id| {
                DofUtils.fillBoundary(
                    mesh,
                    dof_map,
                    block_id,
                    DofUtils.operSystemBoundary(oper),
                    sys,
                );
            }

            // Stores a copy of the system from the down cycle in order to compute the correction
            const sys_old = try SystemSlice(T.System).init(allocator, total_dofs);
            defer sys_old.deinit(allocator);

            // The dof vector context (filled at start of iteration and then immutable);
            const ctx = try SystemSlice(T.Context).init(allocator, total_dofs);
            defer ctx.deinit(allocator);

            for (0..mesh.blocks.len) |block_id| {
                DofUtils.copyDofsFromCells(T.Context, mesh, dof_map, block_id, ctx, context.toConst());
            }

            for (0..mesh.blocks.len) |block_id| {
                DofUtils.fillBoundary(
                    mesh,
                    dof_map,
                    block_id,
                    DofUtils.operContextBoundary(oper),
                    ctx,
                );
            }

            // Stores the right hand side of the equations
            const rhs = try SystemSlice(T.System).init(allocator, mesh.cell_total);
            defer rhs.deinit(allocator);

            for (0..mesh.blocks.len) |block_id| {
                DofUtils.copyCells(T.System, mesh, block_id, rhs, b);
            }

            // Stores the residual (i.e. b - Ax).
            const res = try SystemSlice(T.System).init(allocator, mesh.cell_total);
            defer res.deinit(allocator);

            for (0..mesh.blocks.len) |block_id| {
                DofUtils.residual(
                    mesh,
                    dof_map,
                    block_id,
                    oper,
                    res,
                    rhs.toConst(),
                    sys.toConst(),
                    ctx.toConst(),
                );
            }

            // Use initial residual to set tolerance.
            const ires: f64 = norm(res.field(sys_field));
            const tol: f64 = self.tolerance * @fabs(ires);

            std.debug.print("Multigrid Tolerance {}\n", .{tol});

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(allocator);
            defer arena.deinit();

            const scratch: Allocator = arena.allocator();

            // Run iterations
            var iteration: usize = 0;

            while (iteration < self.max_iters) : (iteration += 1) {
                defer _ = arena.reset(.retain_capacity);

                // Recurse down the mesh to the base level
                for (1..mesh.levels.len) |reverse_level| {
                    const level_id: usize = mesh.levels.len - reverse_level;
                    const level = mesh.levels[level_id];
                    const coarse = mesh.levels[level_id - 1];

                    // Perform presmoothing
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        DofUtils.smooth(
                            mesh,
                            dof_map,
                            block_id,
                            oper,
                            x,
                            rhs.toConst(),
                            sys.toConst(),
                            ctx.toConst(),
                        );

                        DofUtils.copyDofsFromCells(
                            T.System,
                            mesh,
                            dof_map,
                            block_id,
                            sys,
                            x.toConst(),
                        );
                    }

                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        // Fill boundaries now that sys has been smoothed
                        DofUtils.fillBoundary(
                            mesh,
                            dof_map,
                            block_id,
                            DofUtils.operSystemBoundary(oper),
                            sys,
                        );
                    }

                    // Restrict data down a level now that we have smoothed everything
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        DofUtils.restrict(
                            T.System,
                            mesh,
                            dof_map,
                            block_id,
                            sys,
                        );
                    }

                    // Fill sys boundary on restricted level
                    for (coarse.block_offset..coarse.block_offset + coarse.block_total) |block_id| {
                        DofUtils.fillBoundary(
                            mesh,
                            dof_map,
                            block_id,
                            DofUtils.operSystemBoundary(oper),
                            sys,
                        );
                    }

                    // We now have a "guess" for the value of sys of the coarser level, which we will later use to
                    // correct sys.
                    for (coarse.block_offset..coarse.block_offset + coarse.block_total) |block_id| {
                        DofUtils.copyDofs(
                            T.System,
                            dof_map,
                            block_id,
                            sys_old,
                            sys.toConst(),
                        );
                    }

                    // Compute corrected rhs
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        DofUtils.residual(
                            mesh,
                            dof_map,
                            block_id,
                            oper,
                            res,
                            rhs.toConst(),
                            sys.toConst(),
                            ctx.toConst(),
                        );

                        DofUtils.restrictRhs(
                            mesh,
                            dof_map,
                            block_id,
                            oper,
                            rhs,
                            res.toConst(),
                            sys.toConst(),
                            ctx.toConst(),
                        );
                    }
                }

                std.debug.print("Solving Level {}\n", .{0});

                // Solve base
                const base = mesh.blocks[0];

                DofUtils.copyCellsFromDofs(
                    T.System,
                    mesh,
                    dof_map,
                    0,
                    x,
                    sys.toConst(),
                );

                const x_field: []f64 = x.field(sys_field)[0..base.cell_total];
                const rhs_field: []f64 = rhs.field(sys_field)[0..base.cell_total];

                // Solve system using the base solver
                const base_linear_map: BaseLinearMap(T) = .{
                    .mesh = mesh,
                    .dof_map = dof_map,
                    .oper = oper,
                    .sys = sys,
                    .ctx = ctx.toConst(),
                };

                try self.base_solver.solve(scratch, base_linear_map, x_field, rhs_field);

                // Copy back to sys
                DofUtils.copyDofsFromCells(
                    T.System,
                    mesh,
                    dof_map,
                    0,
                    sys,
                    x.toConst(),
                );

                DofUtils.fillBoundary(
                    mesh,
                    dof_map,
                    0,
                    DofUtils.operSystemBoundary(oper),
                    sys,
                );

                {
                    const DebugOutput = enum {
                        sys,
                        rhs,
                    };

                    const file_name = try std.fmt.allocPrint(allocator, "output/multigrid_base_{}.vtu", .{iteration});
                    defer allocator.free(file_name);

                    const file = try std.fs.cwd().createFile(file_name, .{});
                    defer file.close();

                    const debug_output = SystemSliceConst(DebugOutput).view(mesh.cell_total, .{
                        .sys = x.field(sys_field),
                        .rhs = rhs.field(sys_field),
                    });

                    try DataOut.writeVtkLevel(DebugOutput, allocator, mesh, 0, debug_output, file.writer());
                }

                // Iterate up, adding correction and performing post smoothing.
                for (1..mesh.levels.len) |level_id| {
                    const level = mesh.levels[level_id];

                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        DofUtils.prolongCorrection(
                            T.System,
                            mesh,
                            dof_map,
                            block_id,
                            sys,
                            sys_old.toConst(),
                        );
                    }

                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        DofUtils.fillBoundary(
                            mesh,
                            dof_map,
                            block_id,
                            DofUtils.operSystemBoundary(oper),
                            sys,
                        );
                    }

                    // Perform post smoothing
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        DofUtils.smooth(
                            mesh,
                            dof_map,
                            block_id,
                            oper,
                            x,
                            rhs.toConst(),
                            sys.toConst(),
                            ctx.toConst(),
                        );

                        DofUtils.copyDofsFromCells(
                            T.System,
                            mesh,
                            dof_map,
                            block_id,
                            sys,
                            x.toConst(),
                        );
                    }

                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        DofUtils.fillBoundary(
                            mesh,
                            dof_map,
                            block_id,
                            DofUtils.operSystemBoundary(oper),
                            sys,
                        );
                    }

                    // Restrict data down a level now that we have smoothed everything
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        DofUtils.restrict(
                            T.System,
                            mesh,
                            dof_map,
                            block_id,
                            sys,
                        );
                    }

                    // const file_name = try std.fmt.allocPrint(allocator, "output/multigrid_up_{}.vtu", .{level_id});
                    // defer allocator.free(file_name);

                    // const file = try std.fs.cwd().createFile(file_name, .{});
                    // defer file.close();

                    // const debug_output = SystemSliceConst(DebugOutput).view(total_dofs, .{
                    //     .sol = sys.field(sys_field),
                    // });

                    // try DofUtils.writeDofsToVtk(DebugOutput, allocator, mesh, level_id, dof_map, debug_output, file.writer());
                }

                for (0..mesh.blocks.len) |block_id| {
                    DofUtils.residual(
                        mesh,
                        dof_map,
                        block_id,
                        oper,
                        res,
                        rhs.toConst(),
                        sys.toConst(),
                        ctx.toConst(),
                    );
                }

                const res_norm = norm(res.field(sys_field));

                std.debug.print("Multigrid Iteration: {}, Residual: {}\n", .{ iteration, res_norm });

                {
                    const DebugOutput = enum {
                        sys,
                        rhs,
                        res,
                    };

                    const file_name = try std.fmt.allocPrint(allocator, "output/multigrid_iteration_{}.vtu", .{iteration});
                    defer allocator.free(file_name);

                    const file = try std.fs.cwd().createFile(file_name, .{});
                    defer file.close();

                    const debug_output = SystemSliceConst(DebugOutput).view(mesh.cell_total, .{
                        .sys = x.field(sys_field),
                        .rhs = rhs.field(sys_field),
                        .res = res.field(sys_field),
                    });

                    try DataOut.writeVtk(DebugOutput, allocator, mesh, debug_output, file.writer());
                }

                if (res_norm <= tol) {
                    break;
                }
            }

            // Copy solution back into x.
            for (0..mesh.blocks.len) |block_id| {
                DofUtils.copyCellsFromDofs(
                    T.System,
                    mesh,
                    dof_map,
                    block_id,
                    x,
                    sys.toConst(),
                );
            }
        }

        // fn Worker(comptime T: type) type {
        //     const sys_field = comptime std.enums.values(T.System)[0];

        //     return struct {
        //         base_solver: BaseSolver,
        //         mesh: *const Mesh,
        //         block_map: []const usize,
        //         dof_map: DofMap,
        //         oper: T,
        //         sys: SystemSlice(T.System),
        //         ctx: SystemSlice(T.Context),
        //         rhs: SystemSlice(T.System),
        //         x: SystemSlice(T.System),

        //         fn cycle(self: *const @This(), level_id: usize) !void {
        //             if (level_id == 0) {
        //                 // Solve base
        //                 const base = self.mesh.blocks[0];

        //                 DofUtils.copyCellsFromDofs(T.System, self.mesh, self.dof_map, 0, self.x, self.sys.toConst());

        //                 const x_field: []f64 = self.x.field(sys_field)[0..base.cell_total];
        //                 const rhs_field: []f64 = self.rhs.field(sys_field)[0..base.cell_total];

        //                 // Solve system using the base solver
        //                 const base_linear_map: BaseLinearMap(T) = .{ .mesh = self.mesh, .oper = self.oper, .ctx = self.ctx, .sys = self.sys };

        //                 self.base_solver.solve(base_linear_map, x_field, rhs_field);

        //                 DofUtils.copyDofsFromCells(T.System, self.mesh, self.dof_map, 0, self.sys, self.x.toConst());
        //                 DofUtils.fillBoundary(self.mesh, self.block_map, self.dof_map, 0, DofUtils.operSystemBoundary(self.oper), self.sys);
        //             } else {

        //             }
        //         }
        //     };
        // }

        fn BaseLinearMap(comptime T: type) type {
            const field_name: []const u8 = comptime std.meta.fieldNames(T.System)[0];

            return struct {
                mesh: *const Mesh,
                dof_map: DofMap,
                oper: T,
                sys: SystemSlice(T.System),
                ctx: SystemSliceConst(T.Context),

                pub fn apply(self: *const @This(), output: []f64, input: []const f64) void {
                    // Build systems slices which mirror input and output.
                    var input_sys: SystemSliceConst(T.System) = undefined;
                    input_sys.len = self.mesh.cell_total; // This is basically just asking for there to be a problem
                    @field(input_sys.ptrs, field_name) = input.ptr;

                    var output_sys: SystemSlice(T.System) = undefined;
                    output_sys.len = self.mesh.cell_total;
                    @field(output_sys.ptrs, field_name) = output.ptr;

                    DofUtils.copyDofsFromCells(
                        T.System,
                        self.mesh,
                        self.dof_map,
                        0,
                        self.sys,
                        input_sys,
                    );

                    DofUtils.fillBoundary(
                        self.mesh,
                        self.dof_map,
                        0,
                        DofUtils.operSystemBoundary(self.oper),
                        self.sys,
                    );

                    DofUtils.apply(
                        self.mesh,
                        self.dof_map,
                        0,
                        self.oper,
                        output_sys,
                        self.sys.toConst(),
                        self.ctx,
                    );
                }

                // pub fn callback(_: *const @This(), iteration: usize, residual: f64, _: []const f64) void {
                //     std.debug.print("Iteration: {}, Residual: {}\n", .{ iteration, residual });
                // }
            };
        }

        fn dot(u: []const f64, v: []const f64) f64 {
            var result: f64 = 0.0;
            for (u, v) |a, b| {
                result += a * b;
            }
            return result;
        }

        fn norm(slice: []const f64) f64 {
            return @sqrt(dot(slice, slice));
        }
    };
}
