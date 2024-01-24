// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");

const common = aeon.common;
const geometry = aeon.geometry;
const tree = aeon.tree;

pub fn BrillInitialData(comptime M: usize) type {
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

        const RealBox = geometry.RealBox(N);
        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexMixin = geometry.IndexMixin(N);

        const BiCGStabSolver = aeon.lac.BiCGStabSolver;

        const Engine = common.Engine(N, M, M);

        pub const Seed = struct {
            amplitude: f64,
            sigma: f64,

            pub fn project(self: Seed, engine: Engine) f64 {
                const pos = engine.position();

                const rho = pos[0];
                const z = pos[1];

                const rho2 = rho * rho;
                const z2 = z * z;
                const sigma2 = self.sigma * self.sigma;

                const term1: f64 = 2.0 * self.amplitude / 4.0;
                const term2 = 2.0 * rho2 * rho2 - 6.0 * rho2 * sigma2 + sigma2 * sigma2 + 2.0 * rho2 * z2;
                const term3 = @exp(-(rho2 + z2) / sigma2) / (sigma2 * sigma2 * sigma2);

                return term1 * term2 * term3;
            }
        };

        pub const Boundary = struct {
            pub fn kind(face: FaceIndex) BoundaryKind {
                if (face.side == false) {
                    return .even;
                } else {
                    return .robin;
                }
            }

            pub fn robin(_: Boundary, pos: [N]f64, face: FaceIndex) Robin {
                if (face.side == false) {
                    return Robin.nuemann(0.0);
                } else {
                    const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);

                    // return Robin.nuemann(0.0);

                    return Robin{
                        .value = 1.0 / r,
                        .flux = 1.0,
                        .rhs = 0.0,
                    };
                }
            }
        };

        pub const MetricOperator = struct {
            seed: []const f64,

            pub fn apply(
                self: MetricOperator,
                engine: Engine,
                metric: []const f64,
            ) f64 {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessian(metric);
                const gradient: [N]f64 = engine.gradient(metric);
                const value: f64 = engine.value(metric);

                const seed: f64 = engine.value(self.seed);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                return -lap - seed * value;
            }

            pub fn applyDiag(
                self: MetricOperator,
                engine: Engine,
            ) f64 {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianDiag();
                const gradient: [N]f64 = engine.gradientDiag();
                const value: f64 = engine.valueDiag();

                const seed: f64 = engine.value(self.seed);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                return -lap - seed * value;
            }
        };

        // Run

        fn run(allocator: Allocator) !void {
            std.debug.print("Running Poisson Elliptic Solver\n", .{});

            var mesh = try TreeMesh.init(allocator, .{
                .origin = [2]f64{ 0.0, 0.0 },
                .size = [2]f64{ 10.0, 10.0 },
            });
            defer mesh.deinit();

            // Globally refine two times
            for (0..3) |r| {
                std.debug.print("Running Refinement {}\n", .{r});

                @memset(mesh.cells.items(.flag), false);

                for (0..mesh.cells.len) |cell_id| {
                    const bounds: RealBox = mesh.cells.items(.bounds)[cell_id];
                    if (bounds.origin[0] < 0.1 and bounds.origin[1] < 0.1) {
                        mesh.cells.items(.flag)[cell_id] = true;
                    }

                    try mesh.refine(allocator);
                }

                try mesh.refine(allocator);
            }

            // std.debug.print("Mesh Neighbors {any}\n", .{mesh.cells.items(.neighbors)});
            // std.debug.print("Mesh Parent {any}\n", .{mesh.cells.items(.parent)});
            // std.debug.print("Mesh Children {any}\n", .{mesh.cells.items(.children)});

            var manager = try NodeManager.init(allocator, [1]usize{16} ** N, 8);
            defer manager.deinit();

            try manager.build(allocator, &mesh);

            std.debug.print("Num packed nodes: {}\n", .{manager.numPackedNodes()});

            // Create worker
            var worker = try NodeWorker.init(allocator, &mesh, &manager);
            defer worker.deinit();

            // Project seed values
            const seed = try allocator.alloc(f64, worker.numNodes());
            defer allocator.free(seed);

            worker.order(M).projectAll(Seed{ .amplitude = 1.0, .sigma = 1.0 }, seed);
            worker.order(M).fillGhostNodesAll(Boundary{}, seed);

            // Allocate psi
            const psi = try allocator.alloc(f64, worker.numNodes());
            defer allocator.free(psi);

            @memset(psi, 0.0);

            // Solve with multigrid
            std.debug.print("Running Multigrid Solver\n", .{});

            const solver: MultigridMethod = .{
                .base_solver = BiCGStabSolver.new(10000, 10e-12),
                .max_iters = 30,
                .tolerance = 10e-10,
                .presmooth = 5,
                .postsmooth = 5,
            };

            try solver.solve(
                allocator,
                &worker,
                MetricOperator{
                    .seed = seed,
                },
                Boundary{},
                psi,
                seed,
            );

            // Output result

            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/brill.vtu", .{});
            defer file.close();

            const Output = enum {
                psi,
                seed,
            };

            const output = SystemConst(Output).view(worker.numNodes(), .{
                .psi = psi,
                .seed = seed,
            });

            var buf = std.io.bufferedWriter(file.writer());

            try DataOut.writeVtk(
                Output,
                allocator,
                &worker,
                output,
                buf.writer(),
            );

            try buf.flush();
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
    try BrillInitialData(2).run(gpa.allocator());
}
