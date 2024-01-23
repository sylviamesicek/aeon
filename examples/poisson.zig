// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");

const common = aeon.common;
const geometry = aeon.geometry;
const tree = aeon.tree;

pub fn PoissonEquation(comptime M: usize) type {
    const N = 2;
    return struct {
        const BoundaryKind = common.BoundaryKind;
        const Robin = common.Robin;

        const DataOut = aeon.DataOut(N, M);

        const SystemConst = common.SystemConst;

        const MultigridMethod = tree.MultigridMethod(N, M, M, BiCGStabSolver);
        const NodeManager = tree.NodeManager(N);
        const NodeWorker = tree.NodeWorker(N, M);
        const TreeMesh = tree.TreeMesh(N);

        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexMixin = geometry.IndexMixin(N);

        const BiCGStabSolver = aeon.lac.BiCGStabSolver;

        const Engine = common.Engine(N, M, M);

        pub const Source = struct {
            amplitude: f64,

            pub fn project(self: Source, engine: Engine) f64 {
                const pos = engine.position();

                var result: f64 = self.amplitude;

                inline for (0..N) |i| {
                    result *= std.math.sin(pos[i]);
                }

                return result;
            }
        };

        pub const Boundary = struct {
            pub fn kind(face: FaceIndex) BoundaryKind {
                _ = face;
                return .odd;
            }

            pub fn robin(self: Boundary, pos: [N]f64, face: FaceIndex) Robin {
                _ = face;
                _ = pos;
                _ = self;
                return Robin.diritchlet(0.0);
            }
        };

        pub const PoissonOperator = struct {
            pub fn apply(
                _: PoissonOperator,
                engine: Engine,
                field: []const f64,
            ) f64 {
                return -engine.laplacian(field);
            }

            pub fn applyDiag(
                _: PoissonOperator,
                engine: Engine,
            ) f64 {
                return -engine.laplacianDiag();
            }
        };

        // Run

        fn run(allocator: Allocator) !void {
            std.debug.print("Running Poisson Elliptic Solver\n", .{});

            var mesh = try TreeMesh.init(allocator, .{
                .origin = [2]f64{ 0.0, 0.0 },
                .size = [2]f64{ 2.0 * std.math.pi, 2.0 * std.math.pi },
            });
            defer mesh.deinit();

            // Globally refine three times
            for (0..3) |r| {
                std.debug.print("Running Refinement {}\n", .{r});
                std.debug.print("Mesh Neighbors {any}\n", .{mesh.cells.items(.neighbors)});
                std.debug.print("Mesh Parent {any}\n", .{mesh.cells.items(.parent)});
                std.debug.print("Mesh Children {any}\n", .{mesh.cells.items(.children)});

                @memset(mesh.cells.items(.flag), true);
                try mesh.refine(allocator);
            }

            var manager = try NodeManager.init(allocator, [1]usize{16} ** N, 8);
            defer manager.deinit();

            try manager.build(allocator, &mesh);

            // for (manager.blocks.items) |block| {
            //     std.debug.print("Block {}\n", .{block});
            // }

            std.debug.print("Num packed nodes: {}\n", .{manager.numPackedNodes()});

            // Create worker
            var worker = try NodeWorker.init(allocator, &mesh, &manager);
            defer worker.deinit();

            // Project source values
            const source = try allocator.alloc(f64, worker.numNodes());
            defer allocator.free(source);

            worker.order(M).projectAll(Source{ .amplitude = 2.0 }, source);

            // Project solution
            const solution = try allocator.alloc(f64, worker.numNodes());
            defer allocator.free(solution);

            worker.order(M).projectAll(Source{ .amplitude = 1.0 }, solution);

            // Allocate numerical cell vector

            const numerical = try allocator.alloc(f64, worker.numNodes());
            defer allocator.free(numerical);

            @memset(numerical, 0.0);

            const solver: MultigridMethod = .{
                .base_solver = BiCGStabSolver.new(10000, 10e-12),
                .max_iters = 20,
                .tolerance = 10e-10,
                .presmooth = 5,
                .postsmooth = 5,
            };

            try solver.solve(
                allocator,
                &worker,
                PoissonOperator{},
                Boundary{},
                numerical,
                source,
            );

            // Compute error

            const err = try allocator.alloc(f64, worker.numNodes());
            defer allocator.free(err);

            for (0..worker.numNodes()) |i| {
                err[i] = numerical[i] - solution[i];
            }

            // Output result

            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/poisson.vtu", .{});
            defer file.close();

            const Output = enum {
                numerical,
                exact,
                err,
                source,
            };

            const output = SystemConst(Output).view(worker.numNodes(), .{
                .numerical = numerical,
                .exact = solution,
                .err = err,
                .source = source,
            });

            try DataOut.writeVtk(
                Output,
                allocator,
                &worker,
                output,
                file.writer(),
            );
        }
    };
}

/// Actual main function (with allocator and leak detection boilerplate)
pub fn main() !void {
    // Setup Allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const deinit_status = gpa.deinit();

        if (deinit_status == .leak) {
            std.debug.print("Runtime data leak detected\n", .{});
        }
    }

    // Run main
    try PoissonEquation(2).run(gpa.allocator());
}
