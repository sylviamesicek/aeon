const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const assert = std.debug.assert;

const tree = @import("tree.zig");

const basis = @import("../basis/basis.zig");
const common = @import("../common/common.zig");
const geometry = @import("../geometry/geometry.zig");
const lac = @import("../lac/lac.zig");

pub fn MultigridMethod(comptime N: usize, comptime M: usize, comptime O: usize, comptime BaseSolver: type) type {
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
        const Mesh = tree.TreeMesh(N);
        const NodeManager = tree.NodeManager(N);
        const NodeWorker = tree.NodeWorker(N, M);
        const isOperator = common.isOperator(N, M, O);
        const isBoundary = common.isBoundary(N);

        pub fn solve(
            self: Self,
            allocator: Allocator,
            worker: *const NodeWorker,
            operator: anytype,
            boundary: anytype,
            x: []f64,
            b: []const f64,
        ) !void {
            const Oper = @TypeOf(operator);
            const Bound = @TypeOf(boundary);

            if (comptime !(isOperator(Oper))) {
                @compileError("operator must satisfy isOperator trait.");
            }

            if (comptime !(isBoundary(Bound))) {
                @compileError("boundary must satisfy isBoundary trait.");
            }

            assert(x.len == worker.numNodes());
            assert(b.len == worker.numNodes());

            const levels = worker.mesh.numLevels();

            const old = try allocator.alloc(f64, worker.numNodes());
            defer allocator.free(old);

            const scr = try allocator.alloc(f64, worker.numNodes());
            defer allocator.free(scr);

            @memset(scr, 0.0);

            const rhs = try allocator.alloc(f64, worker.numNodes());
            defer allocator.free(rhs);

            @memcpy(rhs, b);

            // Use initial right hand side to set tolerance.
            const irhs = worker.normAll(rhs);
            const tol: f64 = self.tolerance * @abs(irhs);

            if (irhs <= 1e-60) {
                std.debug.print("Trivial Linear Problem\n", .{});

                @memset(x, 0.0);
                return;
            }

            // Build scratch allocator
            var arena: ArenaAllocator = ArenaAllocator.init(allocator);
            defer arena.deinit();

            const scratch: Allocator = arena.allocator();

            // Run iterations
            var iteration: usize = 0;

            const recursive: Recursive(Oper, Bound) = .{
                .method = self,
                .worker = worker,
                .oper = operator,
                .bound = boundary,

                .sys = x,
                .old = old,
                .scr = scr,
                .rhs = rhs,
            };

            while (iteration < self.max_iters) : (iteration += 1) {
                defer _ = arena.reset(.retain_capacity);
                // @memcpy(rhs, b);

                // Iterate
                try recursive.iterate(scratch, levels - 1);

                // Check residual
                for (0..levels) |level| {
                    worker.order(O).residual(level, scr, rhs, operator, x);
                }

                const nres = worker.normAll(scr);

                std.debug.print("Iteration {}, Residual {}\n", .{ iteration, nres });

                if (nres <= tol) {
                    break;
                }
            }
        }

        fn Recursive(comptime Oper: type, comptime Bound: type) type {
            return struct {
                method: Self,
                worker: *const NodeWorker,
                oper: Oper,
                bound: Bound,

                sys: []f64,
                old: []f64,
                scr: []f64,
                rhs: []f64,

                pub fn apply(self: *const @This(), out: []f64, in: []const f64) void {
                    self.worker.unpackBase(self.sys, in);
                    self.worker.order(O).fillGhostNodes(0, self.bound, self.sys);
                    self.worker.order(O).apply(0, self.scr, self.oper, self.sys);
                    self.worker.packBase(out, self.scr);
                }

                pub fn iterate(self: @This(), allocator: Allocator, level: usize) !void {
                    const worker = self.worker.order(O);
                    const worker0 = self.worker.order(0);

                    if (level == 0) {
                        const ndofs = self.worker.manager.numPackedBaseNodes();

                        const sys_base = try allocator.alloc(f64, ndofs);
                        defer allocator.free(sys_base);

                        const rhs_base = try allocator.alloc(f64, ndofs);
                        defer allocator.free(rhs_base);

                        self.worker.packBase(sys_base, self.sys);
                        self.worker.packBase(rhs_base, self.rhs);

                        try self.method.base_solver.solve(allocator, self, sys_base, rhs_base);

                        self.worker.unpackBase(self.sys, sys_base);
                        self.worker.unpackBase(self.rhs, rhs_base);

                        self.worker.order(M).fillGhostNodes(0, self.bound, self.sys);

                        return;
                    }

                    // ********************************
                    // Presmoothing

                    for (0..self.method.presmooth) |_| {
                        worker.fillGhostNodes(level, self.bound, self.sys);
                        worker.smooth(level, self.scr, self.oper, self.sys, self.rhs);
                        self.worker.copy(level, self.sys, self.scr);
                    }

                    // ********************************
                    // Restrict Solution

                    worker.fillGhostNodes(level, self.bound, self.sys);
                    worker.restrict(level, self.sys);

                    worker.fillGhostNodes(level - 1, self.bound, self.sys);
                    self.worker.copy(level - 1, self.old, self.sys);

                    // ********************************
                    // Right Hand Side (Tau Correction)

                    worker.residual(level, self.scr, self.rhs, self.oper, self.sys);
                    worker0.restrict(level, self.scr);
                    worker.tauCorrect(level - 1, self.rhs, self.scr, self.oper, self.sys);

                    // ********************************
                    // Recurese

                    try self.iterate(allocator, level - 1);

                    // ********************************
                    // Error Correction

                    // Sys and Old should both have boundaries filled
                    self.worker.copy(level - 1, self.scr, self.sys);
                    self.worker.subtract(level - 1, self.scr, self.old);

                    worker.prolong(level, self.scr);

                    self.worker.add(level, self.sys, self.scr);

                    // **********************************
                    // Post smooth

                    for (0..self.method.postsmooth) |_| {
                        worker.fillGhostNodes(level, self.bound, self.sys);
                        worker.smooth(level, self.scr, self.oper, self.sys, self.rhs);
                        self.worker.copy(level, self.sys, self.scr);
                    }

                    worker.fillGhostNodes(level, self.bound, self.sys);
                }
            };
        }
    };
}
