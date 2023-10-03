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
        base_solver: *const BaseSolver,
        max_iters: usize,
        tolerance: f64,

        const Self = @This();
        const Mesh = meshes.Mesh(N);
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const BoundaryUtils = boundary.BoundaryUtils(N, O);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);
        const CellSpace = basis.CellSpace(N, O);
        const StencilSpace = basis.StencilSpace(N, O);
        const SystemSlice = system.SystemSlice;
        const SystemSliceConst = system.SystemSliceConst;

        pub fn init(max_iters: usize, tolerance: f64, base_solver: *BaseSolver) Self {
            return .{
                .base_solver = base_solver,
                .max_iters = max_iters,
                .tolerance = tolerance,
            };
        }

        pub fn deinit(_: *Self) void {}

        pub fn solve(
            self: *Self,
            allocator: Allocator,
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: DofMap,
            oper: anytype,
            x: SystemSlice(@TypeOf(oper).System),
            b: SystemSliceConst(@TypeOf(oper).System),
            context: SystemSliceConst(@TypeOf(oper).Context),
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

            const sys_field = comptime std.enums.values(T.System)[0];

            // Get total dofs
            const total_dofs = dof_map.total();

            const sys = try SystemSlice(T.System).init(allocator, total_dofs);
            defer sys.deinit(allocator);

            DofUtils.copyDofsFromCellsAll(
                T.System,
                mesh,
                dof_map,
                sys,
                x.toConst(),
            );

            DofUtils.fillBoundaryAll(
                mesh,
                block_map,
                dof_map,
                DofUtils.operSystemBoundary(oper),
                sys,
            );

            // Scratch dof vector
            const sys2 = try SystemSlice(T.System).init(allocator, total_dofs);
            defer sys2.deinit(allocator);

            const ctx = try SystemSlice(T.Context).init(allocator, total_dofs);
            defer ctx.deinit(allocator);

            DofUtils.copyDofsFromCellsAll(
                T.Context,
                mesh,
                dof_map,
                ctx,
                context.toConst(),
            );

            DofUtils.fillBoundaryFullAll(
                mesh,
                block_map,
                dof_map,
                DofUtils.operContextBoundary(oper),
                ctx,
            );

            const tau = try SystemSlice(T.System).init(allocator, mesh.cell_total);
            defer tau.deinit(allocator);

            // Reset tau correction to 0.0
            @memset(tau.field(sys_field), 0.0);

            const rhs = try SystemSlice(T.System).init(allocator, mesh.cell_total);
            defer rhs.deinit(allocator);

            // Run iterations
            var iteration: usize = 0;

            // TODO Might require full smoothing to achieve proper accuracy.

            while (iteration < self.max_iters) : (iteration += 1) {
                // Recurse down the mesh to the base level
                for (1..mesh.active_levels) |reverse_level| {
                    const level: usize = mesh.active_levels - 1 - reverse_level;
                    const target = mesh.getLevel(level);

                    // Compute rhs
                    for (0..target.blockTotal()) |block| {
                        // Apply tau correction
                        const cell_offset = mesh.blockCellOffset(level, block);
                        const cell_total = mesh.blockCellTotal(level, block);

                        for (cell_offset..cell_offset + cell_total) |idx| {
                            inline for (comptime std.enums.values(T.System)) |field| {
                                rhs.field(field)[idx] = b.field(field)[idx] - tau.field(field)[idx];
                            }
                        }
                    }

                    // Perform smoothing
                    for (0..target.blockTotal()) |block| {
                        // Smooth and store result in sys2
                        DofUtils.smooth(
                            mesh,
                            dof_map,
                            level,
                            block,
                            oper,
                            sys2,
                            sys.toConst(),
                            ctx.toConst(),
                            rhs.toConst(),
                        );
                        // Copy back to sys
                        DofUtils.copyDofs(
                            T.System,
                            dof_map,
                            level,
                            block,
                            sys,
                            sys2.toConst(),
                        );
                        // Fill boundaries now that sys has been smoothed
                        DofUtils.fillBoundary(
                            mesh,
                            block_map,
                            dof_map,
                            level,
                            block,
                            DofUtils.operSystemBoundary(oper),
                            sys,
                        );
                        // Restrict updated vector.
                        DofUtils.restrict(
                            T.System,
                            mesh,
                            block_map,
                            dof_map,
                            level,
                            block,
                            sys,
                        );
                    }

                    // Compute tau correction
                    for (0..target.blockTotal()) |block| {
                        // Fill boundaries of sys fully
                        DofUtils.fillBoundaryFull(
                            mesh,
                            block_map,
                            dof_map,
                            level,
                            block,
                            DofUtils.operSystemBoundary(oper),
                            sys,
                        );
                        // Apply operator fully and store in sys2
                        DofUtils.applyFull(
                            mesh,
                            dof_map,
                            level,
                            block,
                            oper,
                            sys2,
                            sys.toConst(),
                            ctx.toConst(),
                        );
                        // Restrict residual and store result in tau
                        DofUtils.restrictResidual(
                            mesh,
                            block_map,
                            dof_map,
                            level,
                            block,
                            oper,
                            tau,
                            sys.toConst(),
                            ctx.toConst(),
                            sys2.toConst(),
                        );
                    }

                    // Copy data from sys to sys2
                    for (0..target.blockTotal()) |block| {
                        DofUtils.copyDofs(
                            T.System,
                            dof_map,
                            level,
                            block,
                            sys2,
                            sys.toConst(),
                        );
                    }
                }

                // Solve base
                const base = mesh.getLevel(0);

                const field: T.System = comptime std.enums.values(T.System)[0];
                const x_field: []f64 = x.field(field)[0..base.cell_total];
                const rhs_field: []f64 = rhs.field(field)[0..base.cell_total];

                // Set x_field and rhs_field
                const cell_space = CellSpace.fromSize(DofUtils.blockCellSize(mesh, 0, 0));

                var cells = cell_space.cells();
                var linear: usize = 0;

                while (cells.next()) |cell| : (linear += 1) {
                    x_field[linear] = cell_space.value(cell, sys.field(sys_field)); // Value from sys
                    rhs_field[linear] = b.field(sys_field)[linear] - tau.field(sys_field)[linear];
                }

                // Solve system using the base solver
                const base_linear_map: BaseLinearMap(T) = .{
                    .mesh = mesh,
                    .oper = oper,
                    .ctx = ctx,
                    .sys = sys,
                };

                self.base_solver.solve(base_linear_map, x_field, rhs_field);

                // Copy back to sys
                DofUtils.copyDofsFromCells(
                    T.System,
                    mesh,
                    dof_map,
                    0,
                    0,
                    sys,
                    x.toConst(),
                );

                DofUtils.fillBoundary(
                    mesh,
                    block_map,
                    dof_map,
                    0,
                    0,
                    DofUtils.operSystemBoundary(oper),
                    sys,
                );

                // Iterate up, adding correction and performing post smoothing.
                for (1..mesh.active_levels) |level| {
                    const target = mesh.getLevel(level);

                    for (0..target.blockTotal()) |block| {
                        DofUtils.prolongCorrection(
                            T.System,
                            mesh,
                            block_map,
                            dof_map,
                            level,
                            block,
                            x,
                            sys.toConst(),
                            sys2.toConst(),
                        );

                        // Copy back to sys
                        DofUtils.copyDofsFromCells(
                            T.System,
                            mesh,
                            dof_map,
                            level,
                            block,
                            sys,
                            x.toConst(),
                        );

                        DofUtils.fillBoundary(
                            mesh,
                            block_map,
                            dof_map,
                            level,
                            block,
                            DofUtils.operSystemBoundary(oper),
                            sys,
                        );
                    }

                    // Perform smoothing
                    for (0..target.blockTotal()) |block| {
                        // Apply tau correction
                        const cell_offset = mesh.blockCellOffset(level, block);
                        const cell_total = mesh.blockCellTotal(level, block);

                        for (cell_offset..cell_offset + cell_total) |idx| {
                            inline for (comptime std.enums.values(T.System)) |f| {
                                rhs.field(f)[idx] = b.field(f)[idx] - tau.field(f)[idx];
                            }
                        }

                        // Smooth and store result in sys2
                        DofUtils.smooth(
                            mesh,
                            dof_map,
                            level,
                            block,
                            oper,
                            sys2,
                            sys.toConst(),
                            ctx.toConst(),
                            rhs.toConst(),
                        );
                        // Copy back to sys
                        DofUtils.copyDofs(
                            T.System,
                            dof_map,
                            level,
                            block,
                            sys,
                            sys2.toConst(),
                        );
                        // Fill boundaries now that sys has been smoothed
                        DofUtils.fillBoundary(
                            mesh,
                            block_map,
                            dof_map,
                            level,
                            block,
                            DofUtils.operSystemBoundary(oper),
                            sys,
                        );
                        // Restrict updated vector.
                        DofUtils.restrict(
                            T.System,
                            mesh,
                            block_map,
                            dof_map,
                            level,
                            block,
                            sys,
                        );
                    }
                }
            }

            // Copy final solution back into x.
            DofUtils.copyCellsFromDofsAll(
                T.System,
                mesh,
                dof_map,
                x,
                sys.toConst(),
            );
        }

        fn BaseLinearMap(comptime T: type) type {
            const field_name: []const u8 = comptime std.meta.fieldNames(T.System)[0];

            return struct {
                mesh: *const Mesh,
                oper: T,
                ctx: SystemSlice(T.Context),
                sys: SystemSlice(T.System),

                pub fn apply(self: *const @This(), output: []f64, input: []const f64) void {
                    const sys = self.sys;
                    const ctx = self.ctx;

                    // Aliases
                    const stencil_space = DofUtils.blockStencilSpace(self.mesh, 0, 0);
                    const cell_space = stencil_space.cellSpace();

                    const base_dofs = cell_space.total();

                    // Fill base boundary and inner
                    {
                        var cells = cell_space.cells();
                        var linear: usize = 0;

                        while (cells.next()) |cell| : (linear += 1) {
                            inline for (comptime std.enums.values(T.System)) |field| {
                                cell_space.setValue(cell, sys.field(field), input[linear]);
                            }
                        }

                        BoundaryUtils.fillBoundary(
                            O,
                            stencil_space,
                            DofUtils.operSystemBoundary(self.oper),
                            sys.slice(0, base_dofs),
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
                                .ctx = ctx.slice(0, base_dofs).toConst(),
                                .sys = sys.slice(0, base_dofs).toConst(),
                            };

                            const app = self.oper.apply(engine);
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
