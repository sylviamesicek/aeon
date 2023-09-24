// Subdirectories
pub const basis = @import("basis/basis.zig");
pub const dofs = @import("dofs/dofs.zig");
pub const geometry = @import("geometry/geometry.zig");
pub const index = @import("index.zig");
pub const lac = @import("lac/lac.zig");
pub const mesh = @import("mesh/mesh.zig");
pub const system = @import("system.zig");
pub const vtkio = @import("vtkio.zig");

// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

// ***********************
// Temp Main Code ********
// ***********************

pub fn ScalarFieldProblem(comptime O: usize) type {
    const N = 2;
    return struct {
        const BoundaryCondition = dofs.BoundaryCondition;
        const DofHandler = dofs.DofHandler(N, O);
        const MultigridSolver = dofs.MultigridSolver(N, O, BiCGStabSolver);

        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);

        const BiCGStabSolver = lac.BiCGStabSolver(2);

        const Mesh = mesh.Mesh(N, O);

        pub const Seed = enum {
            seed,
        };

        pub const Metric = enum {
            factor,
        };

        // Seed Function
        pub const SeedFunction = struct {
            amplitude: f64,
            sigma: f64,

            pub const Input = enum {};
            pub const Output = Seed;

            pub fn value(self: SeedFunction, engine: dofs.FunctionEngine(N, O, Input)) system.SystemValue(Output) {
                const pos = engine.position();
                const rho = pos[0];
                const z = pos[1];

                const rho2 = rho * rho;
                const z2 = z * z;
                const sigma2 = self.sigma * self.sigma;

                const val = self.amplitude * rho2 / sigma2 * @exp(-(rho2 + z2) / sigma2);

                return .{
                    .seed = val,
                };
            }
        };

        pub const SeedLaplacianFunction = struct {
            amplitude: f64,
            sigma: f64,

            pub const Input = enum {};
            pub const Output = Seed;

            pub fn value(self: SeedLaplacianFunction, engine: dofs.FunctionEngine(N, O, Input)) system.SystemValue(Output) {
                const pos = engine.position();
                const rho = pos[0];
                const z = pos[1];

                const rho2 = rho * rho;
                const z2 = z * z;
                const sigma2 = self.sigma * self.sigma;

                const term1: f64 = 2 * self.amplitude;
                const term2 = 2 * rho2 * rho2 - 6 * rho2 * sigma2 + sigma2 * sigma2 + 2 * rho2 * z2;
                const term3 = @exp(-(rho2 + z2) / sigma2) / (sigma2 * sigma2 * sigma2);

                return .{
                    .seed = term1 * term2 * term3,
                };
            }
        };

        pub const MetricOperator = struct {
            pub const Context = Seed;
            pub const System = Metric;

            pub fn apply(_: MetricOperator, engine: dofs.OperatorEngine(N, O, Context, System)) system.SystemValue(System) {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianOp(.factor);
                const gradient: [N]f64 = engine.gradientOp(.factor);
                const value: f64 = engine.valueOp(.factor);

                const seed: f64 = engine.value(.seed);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                return .{
                    .factor = -lap - seed * value,
                };
            }

            pub fn applyDiagonal(_: MetricOperator, engine: dofs.OperatorEngine(N, O, Context, System)) system.SystemValue(System) {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianDiagonal();
                const gradient: [N]f64 = engine.gradientDiagonal();
                const value: f64 = engine.valueDiagonal();

                const seed: f64 = engine.value(.seed);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                return .{
                    .factor = -lap - seed * value,
                };
            }

            pub fn condition(_: MetricOperator, pos: [N]f64, face: Face) dofs.SystemBoundaryCondition(System) {
                if (face.side) {
                    const r = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                    _ = r;
                } else {
                    return .{
                        .factor = BoundaryCondition.nuemann(0.0),
                    };
                }

                if (face.side == false) {
                    return .{ .factor = BoundaryCondition.nuemann(0.0) };
                } else {
                    const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                    return .{ .factor = BoundaryCondition.robin(1.0 / r, 1.0, 0.0) };
                }
            }
        };

        // Run

        fn run(allocator: Allocator) !void {
            var grid = Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, 0.0 },
                    .size = [2]f64{ 10.0, 10.0 },
                },
                .tile_width = 16,
                .index_size = [2]usize{ 1, 1 },
            });
            defer grid.deinit();

            var dof_handler = DofHandler.init(allocator, &grid);
            defer dof_handler.deinit();

            const ndofs: usize = dof_handler.ndofs();
            const ndofs_reduced = IndexSpace.fromSize(Index.scaled(grid.base.index_size, grid.tile_width)).total();

            var seed: []f64 = try allocator.alloc(f64, ndofs);
            defer allocator.free(seed);

            const seed_func: SeedLaplacianFunction = .{
                .amplitude = 1.0,
                .sigma = 1.0,
            };

            dof_handler.apply(seed_func, .{ .seed = seed }, .{});

            var metric: []f64 = try allocator.alloc(f64, ndofs);
            defer allocator.free(metric);

            const oper = MetricOperator{};

            var base_solver = try BiCGStabSolver.init(allocator, ndofs_reduced, 1000, 10e-6);
            defer base_solver.deinit();

            var solver = try MultigridSolver.init(allocator, &dof_handler, &base_solver);
            defer solver.deinit();

            solver.solve(
                oper,
                .{ .factor = metric },
                .{ .factor = seed },
                .{ .seed = seed },
            );

            const file = try std.fs.cwd().createFile("output/seed.vtu", .{});
            defer file.close();

            try dof_handler.writeVtk(.{ .seed = seed, .metric = metric }, file.writer());
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
    try ScalarFieldProblem(2).run(gpa.allocator());
}

test {
    _ = basis;
    _ = geometry;
    _ = mesh;
    _ = vtkio;
    _ = lac;
}
