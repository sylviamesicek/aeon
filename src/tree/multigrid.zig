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
        gpa: Allocator,

        config: Config,
        base_solver: BaseSolver,
        base_buffer: ArenaAllocator,

        scr: []f64,
        old: []f64,
        rhs: []f64,

        const Self = @This();
        const Mesh = tree.TreeMesh(N);
        const NodeManager = tree.NodeManager(N, M);
        const NodeWorker = tree.NodeWorker(N, M);
        const isOperator = common.isOperator(N, M, O);
        const isBoundary = common.isBoundary(N);

        pub const Config = struct {
            max_iters: usize,
            tolerance: f64,
            presmooth: usize,
            postsmooth: usize,
        };

        pub fn init(allocator: Allocator, num_nodes: usize, base_solver: BaseSolver, config: Config) !Self {
            const scr = try allocator.alloc(f64, num_nodes);
            errdefer allocator.free(scr);

            const old = try allocator.alloc(f64, num_nodes);
            errdefer allocator.free(old);

            const rhs = try allocator.alloc(f64, num_nodes);
            errdefer allocator.free(rhs);

            return .{
                .gpa = allocator,

                .config = config,
                .base_solver = base_solver,
                .base_buffer = ArenaAllocator.init(allocator),

                .scr = scr,
                .old = old,
                .rhs = rhs,
            };
        }

        pub fn deinit(self: *Self) void {
            self.base_buffer.deinit();
            self.gpa.free(self.scr);
            self.gpa.free(self.old);
            self.gpa.free(self.rhs);
        }

        pub fn solve(
            self: *Self,
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

            assert(x.len == worker.manager.numNodes());
            assert(b.len == worker.manager.numNodes());
            assert(self.scr.len == worker.manager.numNodes());

            const levels = worker.mesh.numLevels();

            @memset(self.scr, 0.0);
            @memcpy(self.rhs, b);

            // Use initial right hand side to set tolerance.
            const irhs = worker.norm(self.rhs);
            const tol: f64 = self.config.tolerance * @abs(irhs);

            if (irhs <= 1e-60) {
                std.debug.print("Trivial Linear Problem\n", .{});

                @memset(x, 0.0);
                return;
            }

            // Run iterations
            var iteration: usize = 0;

            const recursive: Recursive(Oper, Bound) = .{
                .method = self,
                .worker = worker,
                .oper = operator,
                .bound = boundary,
            };

            while (iteration < self.config.max_iters) : (iteration += 1) {
                @memcpy(self.rhs, b);

                // Iterate
                try recursive.iterate(levels - 1, x);

                // Check if residual is less than tolerance.
                worker.order(O).residual(self.scr, self.rhs, operator, x);

                const nres = worker.norm(self.scr);

                if (nres <= tol) {
                    break;
                }

                // // Debugging code

                // const DataOut = @import("../aeon.zig").DataOut(N, M);

                // const file_name = try std.fmt.allocPrint(allocator, "output/multigrid{}.vtu", .{iteration});
                // defer allocator.free(file_name);

                // const file = try std.fs.cwd().createFile(file_name, .{});
                // defer file.close();

                // const Output = enum { residual, sys };

                // const output = common.SystemConst(Output).view(worker.numNodes(), .{
                //     .residual = scr,
                //     .sys = x,
                // });

                // var buf = std.io.bufferedWriter(file.writer());

                // try DataOut.writeVtk(
                //     Output,
                //     allocator,
                //     worker,
                //     output,
                //     buf.writer(),
                // );

                // try buf.flush();

                // std.debug.print("Iteration {}, Residual {}\n", .{ iteration, nres });
            }
        }

        fn Recursive(comptime Oper: type, comptime Bound: type) type {
            return struct {
                method: *Self,
                worker: *const NodeWorker,
                oper: Oper,
                bound: Bound,

                pub fn iterate(self: @This(), level: usize, sys: []f64) !void {
                    const rhs = self.method.rhs;
                    const old = self.method.old;
                    const scr = self.method.scr;

                    const worker = self.worker.order(O);
                    const worker0 = self.worker.order(0);

                    if (level == 0) {
                        defer _ = self.method.base_buffer.reset(.retain_capacity);
                        const allocator = self.method.base_buffer.allocator();

                        const num_base_nodes = self.worker.numBaseNodes();

                        const sys_base = try allocator.alloc(f64, num_base_nodes);
                        defer allocator.free(sys_base);

                        const rhs_base = try allocator.alloc(f64, num_base_nodes);
                        defer allocator.free(rhs_base);

                        self.worker.packBase(sys_base, sys);
                        self.worker.packBase(rhs_base, rhs);

                        const BaseOperator = struct {
                            worker: *const NodeWorker,
                            oper: Oper,
                            bound: Bound,
                            scr: []f64,
                            sys: []f64,

                            pub fn apply(base: *const @This(), out: []f64, in: []const f64) void {
                                base.worker.unpackBase(base.sys, in);
                                base.worker.order(O).fillLevelGhostNodes(0, base.bound, base.sys);
                                base.worker.order(O).applyLevel(0, base.scr, base.oper, base.sys);
                                base.worker.packBase(out, base.scr);
                            }
                        };

                        try self.method.base_solver.solve(allocator, BaseOperator{
                            .worker = self.worker,
                            .oper = self.oper,
                            .bound = self.bound,
                            .scr = scr,
                            .sys = sys,
                        }, sys_base, rhs_base);

                        self.worker.unpackBase(sys, sys_base);
                        self.worker.unpackBase(rhs, rhs_base);

                        self.worker.order(O).fillLevelGhostNodes(0, self.bound, sys);

                        return;
                    }

                    // ********************************
                    // Presmoothing

                    for (0..self.method.config.presmooth) |_| {
                        worker.fillLevelGhostNodes(level, self.bound, sys);
                        worker.smoothLevel(level, scr, self.oper, sys, rhs);
                        self.worker.copyLevel(level, sys, scr);
                    }

                    // ********************************
                    // Restrict Solution

                    worker.fillLevelGhostNodes(level, self.bound, sys);
                    worker.restrictLevel(level, sys);

                    worker.fillLevelGhostNodes(level - 1, self.bound, sys);
                    self.worker.copyLevel(level - 1, old, sys);

                    // ********************************
                    // Right Hand Side (Tau Correction)

                    worker.residualLevel(level, scr, rhs, self.oper, sys);
                    worker0.restrictLevel(level, scr);
                    worker.tauCorrectLevel(level - 1, rhs, scr, self.oper, sys);

                    // ********************************
                    // Recurse

                    try self.iterate(level - 1, sys);

                    // ********************************
                    // Error Correction

                    // Sys and Old should both have boundaries filled
                    self.worker.copyLevel(level - 1, scr, sys);
                    self.worker.subAssignLevel(level - 1, scr, old);

                    worker.prolongLevel(level, scr);

                    self.worker.addAssignLevel(level, sys, scr);

                    // **********************************
                    // Post smooth

                    for (0..self.method.config.postsmooth) |_| {
                        worker.fillLevelGhostNodes(level, self.bound, sys);
                        worker.smoothLevel(level, scr, self.oper, sys, rhs);
                        self.worker.copyLevel(level, sys, scr);
                    }

                    worker.fillLevelGhostNodes(level, self.bound, sys);
                }
            };
        }
    };
}
