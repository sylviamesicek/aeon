const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const basis = @import("../basis/basis.zig");
const bsamr = @import("../bsamr/bsamr.zig");
const geometry = @import("../geometry/geometry.zig");
const io = @import("../io/io.zig");
const lac = @import("../lac/lac.zig");
const mesh = @import("../mesh/mesh.zig");
const common = @import("../common/common.zig");

const system = @import("system.zig");

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
        presmooth: usize,
        postsmooth: usize,

        const Self = @This();
        const Mesh = bsamr.Mesh(N);
        const DofManager = bsamr.DofManager(N, M);
        const DataOut = io.DataOut(N, M);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexMixin = geometry.IndexMixin(N);

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

            if (comptime !(common.isBoundary(N)(Bound))) {
                @compileError("boundary must satisfy isBoundary trait.");
            }

            // std.debug.print("Running Multigrid Solver\n", .{});

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
            const irhs = norm(b);
            const tol: f64 = self.tolerance * @abs(irhs);

            if (irhs <= 1e-60) {
                std.debug.print("Trivial Linear Problem\n", .{});

                @memset(x, 0.0);
                return;
            }

            // std.debug.print("Multigrid Tolerance {}\n", .{tol});

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(allocator);
            defer arena.deinit();

            const scratch: Allocator = arena.allocator();

            // Run iterations
            var iteration: usize = 0;

            const worker: Worker(Op, Bound) = .{
                .base_solver = self.base_solver,
                .presmooth = self.presmooth,
                .postsmooth = self.postsmooth,
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

                dofs.fillBoundary(grid, boundary, sys);
                dofs.residual(grid, operator, res, sys, rhs);

                const finest = grid.blocks.len - 1;

                const nres = norm(res[dofs.cell_map.offset(finest)..dofs.cell_map.offset(finest + 1)]);

                if (nres <= tol) {
                    break;
                }

                // We are not at sufficient accuracy, so run another iteration of multigrid.
                try worker.iterate(scratch, grid.levels.len - 1);
            }

            dofs.copyCellsFromNodes(grid, x, sys);
        }

        fn Worker(comptime Op: type, comptime Bound: type) type {
            return struct {
                base_solver: BaseSolver,
                presmooth: usize,
                postsmooth: usize,
                grid: *const Mesh,
                dofs: *const DofManager,
                operator: Op,
                boundary: Bound,
                x: []f64,
                res: []f64,
                rhs: []f64,
                sys: []f64,
                old: []f64,
                err: []f64,

                // Runs a multigrid iteration on the level. This approximates the solution to
                // A(sys) = rhs on the given level. This leaves sys
                fn iterate(self: @This(), allocator: Allocator, level_id: usize) !void {
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
                            .grid = grid,
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

                    // *****************************
                    // Presmoothing

                    for (0..self.presmooth) |_| {
                        dofs.fillLevelBoundary(grid, level_id, boundary, sys);
                        dofs.smoothLevel(grid, level_id, operator, x, sys, rhs);
                        dofs.copyLevelNodesFromCells(grid, level_id, sys, x);
                    }

                    // *****************************
                    // Restrict Solution

                    dofs.fillLevelBoundary(grid, level_id, boundary, sys);
                    dofs.restrictLevel(grid, level_id, sys);

                    dofs.fillLevelBoundary(grid, level_id - 1, boundary, sys);
                    dofs.copyLevelNodes(grid, level_id - 1, old, sys);

                    // *****************************
                    // Compute Residual

                    dofs.residualLevel(grid, level_id, operator, res, sys, rhs);
                    dofs.restrictLevelCells(grid, level_id, res);

                    // *****************************
                    // RHS computation

                    dofs.applyLevel(grid, level_id - 1, operator, rhs, sys);

                    for (coarse.block_offset..coarse.block_offset + coarse.block_total) |block_id| {
                        for (dofs.cell_map.offset(block_id)..dofs.cell_map.offset(block_id + 1)) |idx| {
                            rhs[idx] += res[idx];
                        }
                    }

                    // *****************************
                    // Recurse

                    try self.iterate(allocator, level_id - 1);

                    // ********************************
                    // Error Correction

                    dofs.fillLevelBoundary(grid, level_id - 1, boundary, sys);
                    dofs.fillLevelBoundary(grid, level_id - 1, boundary, old);

                    for (coarse.block_offset..coarse.block_offset + coarse.block_total) |block_id| {
                        for (dofs.node_map.offset(block_id)..dofs.node_map.offset(block_id + 1)) |idx| {
                            err[idx] = sys[idx] - old[idx];
                        }
                    }

                    dofs.prolongLevel(grid, level_id, err);

                    for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                        for (dofs.node_map.offset(block_id)..dofs.node_map.offset(block_id + 1)) |idx| {
                            sys[idx] += err[idx];
                        }
                    }

                    // **************************************
                    // Post smoothing

                    for (0..self.postsmooth) |_| {
                        dofs.fillLevelBoundary(grid, level_id, boundary, sys);
                        dofs.smoothLevel(grid, level_id, operator, x, sys, rhs);
                        dofs.copyLevelNodesFromCells(grid, level_id, sys, x);
                    }

                    dofs.fillLevelBoundary(grid, level_id, boundary, sys);
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
                    var ain: []const f64 = in;

                    aout.len = self.dofs.numCells();
                    ain.len = self.dofs.numCells();

                    // Cells -> Nodes
                    self.dofs.copyBlockNodesFromCells(self.grid, 0, self.sys, ain);
                    self.dofs.fillBlockBoundary(self.grid, 0, self.boundary, self.sys);
                    // (Apply) Nodes => Cells
                    self.dofs.applyBlock(self.grid, 0, self.operator, aout, self.sys);
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
