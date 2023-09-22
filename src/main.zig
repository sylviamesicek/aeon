// Subdirectories
pub const basis = @import("basis/basis.zig");
pub const geometry = @import("geometry/geometry.zig");
pub const index = @import("index.zig");
pub const mesh = @import("mesh/mesh.zig");
pub const solver = @import("solver/solver.zig");
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
        const Mesh = mesh.Mesh(N, O);
        const Face = geometry.Face(N);
        const BoundaryCondition = basis.BoundaryCondition;

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

            pub fn value(self: SeedFunction, engine: mesh.FunctionEngine(N, O, Input)) system.SystemValue(Output) {
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

            pub fn value(self: SeedLaplacianFunction, engine: mesh.FunctionEngine(N, O, Input)) system.SystemValue(Output) {
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

            pub fn apply(_: MetricOperator, engine: mesh.OperatorEngine(N, O, Context, System)) system.SystemValue(System) {
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

            pub fn applyDiagonal(_: MetricOperator, engine: mesh.OperatorEngine(N, O, Context, System)) system.SystemValue(System) {
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
        };

        pub const MetricBoundaryConditions = struct {
            pub fn condition(_: @This(), pos: [N]f64, face: Face) BoundaryCondition {
                if (face.side == false) {
                    return BoundaryCondition.nuemann(0.0);
                } else {
                    const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                    return BoundaryCondition.robin(1.0 / r, 1.0, 0.0);
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

            const ndofs: usize = grid.cellTotal();

            var seed: []f64 = try allocator.alloc(f64, ndofs);
            defer allocator.free(seed);

            const seed_func: SeedLaplacianFunction = .{
                .amplitude = 1.0,
                .sigma = 1.0,
            };

            grid.apply(seed_func, .{ .seed = seed }, .{});

            var metric: []f64 = try allocator.alloc(f64, ndofs);
            defer allocator.free(metric);

            const oper = MetricOperator{};
            const boundary = MetricBoundaryConditions{};

            // Solve
            try grid.solveBase(oper, .{ .factor = metric }, .{ .factor = seed }, .{ .seed = seed }, boundary);

            const file = try std.fs.cwd().createFile("output/seed.vtu", .{});
            defer file.close();

            try grid.writeVtk(.{ .seed = seed, .metric = metric }, file.writer());
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
    _ = solver;
    _ = vtkio;
}
