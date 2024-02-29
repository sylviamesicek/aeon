const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const assert = std.debug.assert;

const mesh_ = @import("mesh.zig");

const basis = @import("../basis/basis.zig");
const common = @import("../common/common.zig");
const geometry = @import("../geometry/geometry.zig");
const lac = @import("../lac/lac.zig");

pub fn MultigridMethod(comptime N: usize, comptime M: usize, comptime BaseSolver: type) type {
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
        const Mesh = mesh_.Mesh(N);
        const NodeManager = mesh_.NodeManager(N, M);
        const NodeWorker = mesh_.NodeWorker(N, M);

        const checkBoundarySet = common.checkBoundarySet;
        const checkOperator = common.checkOperator;
        const checkFunction = common.checkFunction;

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
            set: anytype,
            x: []f64,
            b: []const f64,
        ) !void {
            const Oper = @TypeOf(operator);
            const Set = @TypeOf(set);

            checkBoundarySet(N, Set);
            checkOperator(N, M, Oper);

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

            const recursive: Recursive(Oper, Set) = .{
                .method = self,
                .worker = worker,
                .oper = operator,
                .set = set,
            };

            while (iteration < self.config.max_iters) : (iteration += 1) {
                @memcpy(self.rhs, b);

                // Iterate
                try recursive.iterate(levels - 1, x);

                // for (0..worker.mesh.numLevels()) |rev_id| {
                //     const id = worker.mesh.numLevels() - 1 - rev_id;
                //     worker.restrictLevel(0, id, x);
                // }

                // Check if residual is less than tolerance.
                worker.residual(self.scr, self.rhs, operator, x);

                const nres = worker.norm(self.scr);

                std.debug.print("Iteration {}, Residual {}\n", .{ iteration, nres });

                for (0..worker.mesh.numLevels()) |i| {
                    std.debug.print("    Level: {}, Residual {}\n", .{ i, worker.normLevel(i, self.scr) });
                }

                // worker.residual(self.scr, b, operator, x);

                // std.debug.print("Iteration {}, Residual {}\n", .{ iteration, worker.norm(self.scr) });

                // for (0..worker.mesh.numLevels()) |i| {
                //     std.debug.print("    Level: {}, Residual {}\n", .{ i, worker.normLevel(i, self.scr) });
                // }

                if (nres <= tol) {
                    break;
                }

                // Debugging code

                // const DataOut = @import("../aeon.zig").DataOut(N, M);

                // const file_name = try std.fmt.allocPrint(self.gpa, "output/multigrid{}.vtu", .{iteration});
                // defer self.gpa.free(file_name);

                // const file = try std.fs.cwd().createFile(file_name, .{});
                // defer file.close();

                // const Output = enum { residual, sys };

                // const output = common.SystemConst(Output).view(worker.manager.numNodes(), .{
                //     .residual = self.scr,
                //     .sys = x,
                // });

                // var buf = std.io.bufferedWriter(file.writer());

                // try DataOut.writeVtk(
                //     Output,
                //     self.gpa,
                //     worker,
                //     output,
                //     buf.writer(),
                // );

                // try buf.flush();

            }
        }

        fn Recursive(comptime Oper: type, comptime Set: type) type {
            const O: usize = Oper.order;

            return struct {
                method: *Self,
                worker: *const NodeWorker,
                oper: Oper,
                set: Set,

                pub fn iterate(self: @This(), level: usize, sys: []f64) !void {
                    const rhs = self.method.rhs;
                    const old = self.method.old;
                    const scr = self.method.scr;

                    const worker = self.worker;

                    if (level == 0) {
                        defer _ = self.method.base_buffer.reset(.retain_capacity);
                        const allocator = self.method.base_buffer.allocator();

                        const num_base_nodes = worker.numBaseNodes();

                        const sys_base = try allocator.alloc(f64, num_base_nodes);
                        defer allocator.free(sys_base);

                        const rhs_base = try allocator.alloc(f64, num_base_nodes);
                        defer allocator.free(rhs_base);

                        worker.packBase(sys_base, sys);
                        worker.packBase(rhs_base, rhs);

                        const BaseOperator = struct {
                            worker: *const NodeWorker,
                            oper: Oper,
                            set: Set,
                            scr: []f64,
                            sys: []f64,

                            pub fn apply(base: *const @This(), out: []f64, in: []const f64) void {
                                base.worker.unpackBase(base.sys, in);
                                base.worker.fillLevelGhostNodes(O, 0, base.set, base.sys);
                                base.worker.applyLevel(0, base.scr, base.oper, base.sys);
                                base.worker.packBase(out, base.scr);
                            }
                        };

                        try self.method.base_solver.solve(allocator, BaseOperator{
                            .worker = self.worker,
                            .oper = self.oper,
                            .set = self.set,
                            .scr = scr,
                            .sys = sys,
                        }, sys_base, rhs_base);

                        worker.unpackBase(sys, sys_base);
                        worker.unpackBase(rhs, rhs_base);

                        worker.fillLevelGhostNodes(O, 0, self.set, sys);

                        return;
                    }

                    // ********************************
                    // Presmoothing

                    for (0..self.method.config.presmooth) |_| {
                        worker.fillLevelGhostNodes(O, level, self.set, sys);
                        worker.smoothLevel(level, scr, self.oper, sys, rhs);
                        self.worker.copyLevel(level, sys, scr);
                    }

                    // ********************************
                    // Restrict Solution

                    worker.fillLevelGhostNodes(O, level, self.set, sys);
                    worker.restrictLevel(0, level, sys);

                    // worker.fillLevelGhostNodes(level, self.bound, sys);

                    // self.worker.copyLevel(level, scr, sys);
                    // self.worker.copyLevel(level - 1, scr, sys);

                    // worker.restrictLevel(level, scr);
                    // worker.fillLevelGhostNodes(level - 1, self.bound, scr);

                    // self.worker.copyLevel(level - 1, old, scr);

                    worker.fillLevelGhostNodes(O, level - 1, self.set, sys);
                    self.worker.copyLevel(level - 1, old, sys);

                    // ********************************
                    // Right Hand Side (Tau Correction)

                    worker.residualLevel(level, scr, rhs, self.oper, sys);
                    worker.fillLevelGhostNodes(O, level, self.set, scr);
                    worker.restrictLevel(1, level, scr);
                    worker.tauCorrectLevel(level, rhs, scr, self.oper, sys);

                    // ********************************
                    // Recurse

                    try self.iterate(level - 1, sys);

                    // ********************************
                    // Error Correction

                    // Sys and Old should both have boundaries filled by this point
                    self.worker.copyLevel(level - 1, scr, sys);
                    self.worker.subAssignLevel(level - 1, scr, old);

                    worker.prolongLevel(1, level, scr);

                    self.worker.addAssignLevel(level, sys, scr);

                    // **********************************
                    // Post smooth

                    for (0..self.method.config.postsmooth) |_| {
                        worker.fillLevelGhostNodes(O, level, self.set, sys);
                        worker.smoothLevel(level, scr, self.oper, sys, rhs);
                        self.worker.copyLevel(level, sys, scr);
                    }

                    worker.fillLevelGhostNodes(O, level, self.set, sys);
                }
            };
        }
    };
}
