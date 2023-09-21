// Subdirectories
pub const basis = @import("basis/basis.zig");
pub const geometry = @import("geometry/geometry.zig");
pub const index = @import("index.zig");
pub const mesh = @import("mesh/mesh.zig");
pub const solver = @import("solver/solver.zig");
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

            pub const Context = enum {};
            pub const Output = Seed;

            pub fn value(self: SeedFunction, engine: mesh.FunctionEngine(N, O, Context)) mesh.system.SystemValue(Output) {
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

            pub const Context = enum {};
            pub const Output = Seed;

            pub fn value(self: SeedLaplacianFunction, engine: mesh.FunctionEngine(N, O, Context)) mesh.system.SystemValue(Output) {
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

            pub fn apply(_: MetricOperator, engine: mesh.OperatorEngine(N, O, Context, System)) mesh.system.SystemValue(System) {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianOp(.metric);
                const gradient: [N]f64 = engine.gradientOp(.metric);
                const value: f64 = engine.valueOp(.metric);

                const seed: f64 = engine.value(.seed);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                return .{
                    .seed = -lap - seed * value,
                };
            }

            pub fn applyDiagonal(_: MetricOperator, engine: mesh.OperatorEngine(N, O, Context, System)) mesh.system.SystemValue(System) {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianDiagonal();
                const gradient: [N]f64 = engine.gradientDiagonal();
                const value: f64 = engine.valueDiagonal();

                const seed: f64 = engine.value(.seed);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                return .{
                    .seed = -lap - seed * value,
                };
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
                .index_size = [2]usize{ 8, 8 },
            });
            defer grid.deinit();

            const ndofs: usize = grid.cellTotal();

            var seed: []f64 = try allocator.alloc(f64, ndofs);
            defer allocator.free(seed);

            const seed_func: SeedLaplacianFunction = .{
                .amplitude = 1.0,
                .sigma = 1.0,
            };

            grid.project(seed_func, .{ .seed = seed }, .{});

            var psi: []f64 = try allocator.alloc(f64, ndofs);
            defer allocator.free(psi);

            const file = try std.fs.cwd().createFile("output/seed.vtu", .{});
            defer file.close();

            try grid.writeVtk(.{ .seed = seed }, file.writer());
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
