// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");

const bsamr = aeon.bsamr;
const geometry = aeon.geometry;
const nodes = aeon.nodes;

pub fn PoissonEquation(comptime M: usize) type {
    const N = 2;
    return struct {
        const BoundaryKind = nodes.BoundaryKind;
        const Robin = nodes.Robin;

        const DataOut = aeon.DataOut(N, M);
        const MultigridMethod = aeon.methods.MultigridMethod(N, M, BiCGStabSolver);
        const LinearMapMethod = aeon.methods.LinearMapMethod(N, M, BiCGStabSolver);
        const SystemConst = aeon.methods.SystemConst;

        const Mesh = bsamr.Mesh(N);
        const DofManager = bsamr.DofManager(N, M);
        const RegridManager = bsamr.RegridManager(N);

        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexMixin = geometry.IndexMixin(N);

        const BiCGStabSolver = aeon.lac.BiCGStabSolver;

        const Engine = aeon.mesh.Engine(N, M);

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

                    return .{
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

            var mesh = try Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, 0.0 },
                    .size = [2]f64{ 2.0 * std.math.pi, 2.0 * std.math.pi },
                },
                .tile_width = 8,
                .index_size = [2]usize{ 1, 1 },
            });
            defer mesh.deinit();

            var dofs = DofManager.init(allocator);
            defer dofs.deinit();

            try dofs.build(&mesh);

            // Globally refine three times

            for (0..10) |_| {
                const amr: RegridManager = .{
                    .max_levels = 16,
                    .patch_efficiency = 0.1,
                    .block_efficiency = 0.7,
                };

                var tags = try allocator.alloc(bool, dofs.numTiles());
                defer allocator.free(tags);

                @memset(tags, true);

                try amr.regrid(allocator, tags, &mesh, dofs.tile_map);
                try dofs.build(&mesh);
            }

            std.debug.print("NDofs: {}\n", .{dofs.numNodes()});

            // Project seed values
            const seed = try allocator.alloc(f64, dofs.numCells());
            defer allocator.free(seed);

            dofs.project(&mesh, Seed{ .amplitude = 1.0, .sigma = 1.0 }, seed);

            const nseed = try allocator.alloc(f64, dofs.numNodes());
            defer allocator.free(nseed);

            dofs.transfer(&mesh, Boundary{}, nseed, seed);

            // Allocate metric vector

            const metric = try allocator.alloc(f64, dofs.numCells());
            defer allocator.free(metric);

            @memset(metric, 0.0);

            // Solve using multigrid method

            const solver: MultigridMethod = .{
                .base_solver = BiCGStabSolver.new(20000, 10e-14),
                .max_iters = 100,
                .tolerance = 10e-10,
                .presmooth = 5,
                .postsmooth = 5,
            };

            try solver.solve(
                allocator,
                &mesh,
                &dofs,
                MetricOperator{ .seed = nseed },
                Boundary{},
                metric,
                seed,
            );

            // Output result

            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/brill.vtu", .{});
            defer file.close();

            const Output = enum {
                metric,
                seed,
            };

            const output = SystemConst(Output).view(dofs.numCells(), .{
                .metric = metric,
                .seed = seed,
            });

            try DataOut.writeVtk(
                Output,
                allocator,
                &mesh,
                &dofs,
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
