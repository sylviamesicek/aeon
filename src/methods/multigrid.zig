const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const basis = @import("../basis/basis.zig");
const bsamr = @import("../bsamr/bsamr.zig");
const geometry = @import("../geometry/geometry.zig");
const lac = @import("../lac/lac.zig");
const mesh = @import("../mesh/mesh.zig");
const nodes = @import("../nodes/nodes.zig");

/// A multigrid based elliptic solver which uses the given base solver to approximate the solution
/// on the lowest level of the mesh.
pub fn MultigridMethod(comptime N: usize, comptime M: usize, comptime BaseSolver: type) type {
    if (comptime !lac.isLinearSolver(BaseSolver)) {
        @compileError("Base solver must satisfy the a linear solver requirement.");
    }

    return struct {
        base_solver: BaseSolver,
        max_iters: usize,
        tolerance: f64,

        const Self = @This();
        const Mesh = bsamr.Mesh(N);
        const DofManager = bsamr.DofManager(N, M);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexMixin = geometry.IndexMixin(N);

        pub fn new(max_iters: usize, tolerance: f64, base_solver: BaseSolver) Self {
            return .{
                .base_solver = base_solver,
                .max_iters = max_iters,
                .tolerance = tolerance,
            };
        }

        pub fn solve(
            self: Self,
            allocator: Allocator,
            grid: *const Mesh,
            dofs: *const DofManager,
            operator: anytype,
            boundary: anytype,
            x: []f64,
            b: []const f64,
        ) !void {
            // Trait bounds
            const Op = @TypeOf(operator);
            const Bound = @TypeOf(boundary);

            if (comptime !(mesh.isOperator(N, M)(Op))) {
                @compileError("operator must satisfy isOperator trait.");
            }

            if (comptime !(nodes.isBoundary(N)(Bound))) {
                @compileError("boundary must satisfy isBoundary trait.");
            }

            std.debug.print("Running Multigrid Solver\n", .{});

            // Allocates a node vector for storing the system
            // and cell vectors for the corrected right hand side
            // and residual.

            const sys = try allocator.alloc(f64, dofs.numNodes());
            defer allocator.free(sys);

            dofs.transfer(grid, boundary, sys, x);

            const res = try allocator.alloc(f64, dofs.numCells());
            defer allocator.free(res);

            const rhs = try allocator.alloc(f64, dofs.numCells());
            defer allocator.free(rhs);

            const old = try allocator.alloc(f64, dofs.numNodes());
            defer allocator.free(old);

            const err = try allocator.alloc(f64, dofs.numNodes());
            defer allocator.free(err);

            // Use initial right hand side to set tolerance.
            const irhs = norm(rhs);
            const tol: f64 = self.tolerance * @fabs(irhs);

            if (irhs <= 1e-60) {
                @memset(x, 0.0);
                return;
            }

            std.debug.print("Multigrid Tolerance {}\n", .{tol});

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(allocator);
            defer arena.deinit();

            const scratch: Allocator = arena.allocator();

            // Run iterations
            var iteration: usize = 0;

            const worker: Worker(Op, Bound) = .{
                .grid = grid,
                .dofs = dofs,
                .operator = operator,
                .boundary = boundary,
                .x = x,
                .rhs = rhs,
                .res = res,
                .sys = sys,
                .err = err,
                .old = old,
            };

            while (iteration < self.max_iters) : (iteration += 1) {
                defer _ = arena.reset(.retain_capacity);

                @memcpy(rhs, b);

                dofs.residual(grid, operator, res, sys, rhs);

                const nres = norm(res);

                if (nres <= tol) {
                    break;
                }

                // We are not at sufficient accuracy, so run another iteration of multigrid.

                worker.iterate(scratch, grid.levels.len - 1);

                // const res_norm = norm(res.field(sys_field));

                // std.debug.print("Multigrid Iteration: {}, Residual: {}\n", .{ iteration, res_norm });

                // {
                //     const DebugOutput = enum {
                //         sys,
                //         rhs,
                //         res,
                //     };

                //     const file_name = try std.fmt.allocPrint(allocator, "output/multigrid_iteration_{}.vtu", .{iteration});
                //     defer allocator.free(file_name);

                //     const file = try std.fs.cwd().createFile(file_name, .{});
                //     defer file.close();

                //     const debug_output = SystemSliceConst(DebugOutput).view(mesh.cell_total, .{
                //         .sys = x.field(sys_field),
                //         .rhs = rhs.field(sys_field),
                //         .res = res.field(sys_field),
                //     });

                //     try DataOut.writeVtk(DebugOutput, allocator, mesh, debug_output, file.writer());
                // }

                // if (res_norm <= tol) {
                //     break;
                // }
            }

            dofs.copyCellsFromNodes(grid, x, sys);
        }

        fn Worker(comptime Op: type, comptime Bound: type) type {
            return struct {
                grid: *const Mesh,
                dofs: *const DofManager,
                operator: Op,
                boundary: Bound,
                x: []f64,
                rhs: []f64,
                res: []f64,
                rhs: []f64,
                sys: []f64,
                old: []f64,
                err: []f64,

                // Runs a multigrid iteration on the level. This approximates the solution to
                // A(sys) = rhs on the given level. This leaves sys
                fn iterate(self: @This(), allocator: Allocator, level_id: usize) void {
                    // Aliases
                    const grid = self.grid;
                    const dofs = self.dofs;
                    const operator = self.operator;
                    const boundary = self.boundary;
                    const x = self.x;
                    const rhs = self.rhs;
                    const res = self.res;
                    const sys = self.sys;
                    const old = self.old;
                    const err = self.err;

                    if (level_id == 0) {
                        // Solve base
                        dofs.copyBlockCellsFromNodes(grid, 0, x, sys);

                        const cell_total = dofs.cell_map.total(0);

                        const x_base = x[0..cell_total];
                        const rhs_base = rhs[0..cell_total];

                        // Solve system using the base solver
                        const base_linear_map: BaseLinearMap(Op, Bound) = .{
                            .mesh = grid,
                            .dofs = dofs,
                            .operator = operator,
                            .boundary = boundary,
                            .sys = sys,
                        };

                        try self.base_solver.solve(allocator, base_linear_map, x_base, rhs_base);

                        // Copy back to system
                        dofs.copyBlockNodesFromCells(grid, 0, sys, x);
                        dofs.fillBlockBoundary(grid, 0, boundary, sys);

                        return;
                    }

                    const level = grid.levels[level_id];
                    const coarse = grid.levels[level_id - 1];

                    // Perform presmoothing
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        dofs.fillBlockBoundary(grid, block_id, boundary, sys);
                    }

                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        dofs.smoothBlock(grid, block_id, operator, x, sys, rhs);
                        dofs.copyBlockNodesFromCells(grid, block_id, sys, x);
                    }

                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        dofs.fillBlockBoundary(grid, block_id, boundary, sys);
                    }

                    // TODO restriction like this might be unnessasary.

                    // Restrict data down a level now that we have smoothed everything
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        dofs.restrictBlock(grid, block_id, sys);
                    }

                    // Fill sys boundary on coarse level
                    for (coarse.block_offset..coarse.block_offset + coarse.block_total) |block_id| {
                        dofs.fillBlockBoundary(grid, block_id, boundary, sys);
                    }

                    // Compute residual on coarse level
                    for (coarse.block_offset..coarse.block_offset + coarse.block_total) |block_id| {
                        dofs.residualBlock(grid, block_id, operator, res, sys, rhs);
                    }

                    // Restrict residual from current level
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        dofs.residualBlock(grid, block_id, operator, res, sys, rhs);
                        dofs.restrictBlockCells(grid, block_id, res);
                    }

                    // Fill right hand side on coarse level
                    for (coarse.block_offset..coarse.block_offset + coarse.block_total) |block_id| {
                        dofs.applyBlock(grid, block_id, operator, rhs, sys);

                        for (dofs.cell_map.offset(block_id)..dofs.cell_map.offset(block_id + 1)) |idx| {
                            rhs[idx] += res[idx];
                        }
                    }

                    // Cache current coarse solution
                    for (coarse.block_offset..coarse.block_offset + coarse.block_total) |block_id| {
                        dofs.copyBlockNodes(block_id, old, sys);
                    }

                    // Recurse
                    self.iterate(allocator, level_id - 1);

                    // Compute and prolong err
                    for (coarse.block_offset..coarse.block_offset + coarse.block_total) |block_id| {
                        for (dofs.node_map.offset(block_id)..dofs.node_map.offset(block_id + 1)) |idx| {
                            err[idx] = sys[idx] - old[idx];
                        }
                    }

                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        dofs.prolongBlock(grid, block_id, err);
                    }

                    // Correct system
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        for (dofs.node_map.offset(block_id)..dofs.node_map.offset(block_id + 1)) |idx| {
                            sys[idx] += err[idx];
                        }
                    }

                    // Fill new boundaries
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        dofs.fillBlockBoundary(grid, block_id, boundary, sys);
                    }

                    // Perform post smoothing
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        dofs.smoothBlock(grid, block_id, operator, x, sys, rhs);
                        dofs.copyBlockNodesFromCells(grid, block_id, sys, x);
                    }

                    // Fill boundaries in post
                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        dofs.fillBlockBoundary(grid, block_id, boundary, sys);
                    }
                }
            };
        }

        fn BaseLinearMap(comptime Op: type, comptime Bound: type) type {
            return struct {
                grid: *const Mesh,
                dofs: *const DofManager,
                operator: Op,
                boundary: Bound,
                sys: []f64,

                pub fn apply(self: *const @This(), out: []f64, in: []const f64) void {
                    var aout: []f64 = out;
                    var ain: []f64 = in;

                    aout.len = self.dofs.numCells();
                    ain.len = self.dofs.numCells();

                    // Cells -> Nodes
                    self.dofs.copyBlockNodesFromCells(self.grid, 0, self.sys, ain);
                    self.dofs.fillBlockBoundary(self.grid, 0, self.boundary, self.sys);
                    // (Apply) Nodes => Cells
                    self.dofs.applyBlock(self.grid, 0, self.operator, aout, self.sys);
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
