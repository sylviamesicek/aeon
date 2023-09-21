// Subdirectories
pub const array = @import("array.zig");
pub const basis = @import("basis/basis.zig");
pub const geometry = @import("geometry/geometry.zig");
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

        // Seed Function
        pub const SeedFunction = struct {
            amplitude: f64,
            sigma: f64,

            pub const Context = enum {};
            pub const Output = enum {
                seed,
            };

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

        // Run

        fn run(allocator: Allocator) !void {
            var grid = Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, 0.0 },
                    .size = [2]f64{ 1.0, 1.0 },
                },
                .tile_width = 10,
                .index_size = [2]usize{ 1, 1 },
                .global_refinement = 0,
            });
            defer grid.deinit();

            const ndofs: usize = grid.cellTotal();

            var seed: []f64 = try allocator.alloc(f64, ndofs);
            defer allocator.free(seed);

            const func: SeedFunction = .{
                .amplitude = 1.0,
                .sigma = 1.0,
            };

            grid.project(func, .{ .seed = seed }, .{});

            const file = try std.fs.cwd().createFile("seed.vtu", .{});
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
    _ = array;
    _ = basis;
    _ = geometry;
    _ = mesh;
    _ = solver;
    _ = vtkio;
}
