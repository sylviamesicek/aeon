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
        };

        // Run

        fn run(allocator: Allocator) !void {
            std.debug.print("Running Poisson Elliptic Solver\n", .{});

            var mesh = try Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, 0.0 },
                    .size = [2]f64{ 2.0 * std.math.pi, 2.0 * std.math.pi },
                },
                .tile_width = 32,
                .index_size = [2]usize{ 1, 1 },
            });
            defer mesh.deinit();

            var dofs = DofManager.init(allocator);
            defer dofs.deinit();

            try dofs.build(&mesh);

            // Globally refine three times

            for (0..1) |_| {
                const amr: RegridManager = .{
                    .max_levels = 4,
                    .patch_efficiency = 0.1,
                    .block_efficiency = 0.7,
                };

                var tags = try allocator.alloc(bool, dofs.numTiles());
                defer allocator.free(tags);

                @memset(tags, true);

                try amr.regrid(allocator, tags, &mesh, dofs.tile_map);
                try dofs.build(&mesh);
            }

            for (mesh.patches) |patch| {
                std.debug.print("Patch {}\n", .{patch});
            }

            for (mesh.blocks) |block| {
                std.debug.print("Block {}\n", .{block});
            }

            std.debug.print("NDofs: {}\n", .{dofs.numNodes()});

            // Project source values

            const source = try allocator.alloc(f64, dofs.numCells());
            defer allocator.free(source);

            dofs.project(&mesh, Source{ .amplitude = 2.0 }, source);

            const solution = try allocator.alloc(f64, dofs.numCells());
            defer allocator.free(solution);

            dofs.project(&mesh, Source{ .amplitude = 1.0 }, solution);

            // Allocate numerical cell vector

            const numerical = try allocator.alloc(f64, dofs.numCells());
            defer allocator.free(numerical);

            @memset(numerical, 0.0);

            const solver = MultigridMethod.new(
                20,
                10e-10,
                BiCGStabSolver.new(10000, 10e-10),
            );
            // const solver = LinearMapMethod.new(BiCGStabSolver.new(10000, 10e-10));

            try solver.solve(
                allocator,
                &mesh,
                &dofs,
                PoissonOperator{},
                Boundary{},
                numerical,
                source,
            );

            // Compute error

            const err = try allocator.alloc(f64, dofs.numCells());
            defer allocator.free(err);

            for (0..dofs.numCells()) |i| {
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

            const output = SystemConst(Output).view(dofs.numCells(), .{
                .numerical = numerical,
                .exact = solution,
                .err = err,
                .source = source,
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
