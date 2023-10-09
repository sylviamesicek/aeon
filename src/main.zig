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
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const MultigridSolver = dofs.MultigridSolver(N, O, BiCGStabSolver);
        const SystemSlice = system.SystemSlice;
        const SystemSliceConst = system.SystemSliceConst;

        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);

        const BiCGStabSolver = lac.BiCGStablSolver(2);

        const Mesh = mesh.Mesh(N);

        pub const Seed = enum {
            seed,
        };

        pub const Metric = enum {
            factor,
        };

        // Seed Function
        pub const SeedProjection = struct {
            amplitude: f64,
            sigma: f64,

            pub const System = Seed;

            pub fn project(self: SeedProjection, pos: [N]f64) system.SystemValue(System) {
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

        pub const SeedLaplacianProjection = struct {
            amplitude: f64,
            sigma: f64,

            pub const System = Seed;

            pub fn project(self: SeedLaplacianProjection, pos: [N]f64) system.SystemValue(System) {
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

                // return .{
                //     .seed = 1.0,
                // };
            }
        };

        pub const MetricOperator = struct {
            pub const Context = Seed;
            pub const System = Metric;

            pub fn apply(_: MetricOperator, engine: dofs.OperatorEngine(N, O, Context, System)) system.SystemValue(System) {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianSys(.factor);
                const gradient: [N]f64 = engine.gradientSys(.factor);
                const value: f64 = engine.valueSys(.factor);

                const seed: f64 = engine.valueCtx(.seed);

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

                const seed: f64 = engine.valueCtx(.seed);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                return .{
                    .factor = -lap - seed * value,
                };
            }

            pub fn boundarySys(_: MetricOperator, pos: [N]f64, face: Face) dofs.SystemBoundaryCondition(System) {
                if (face.side == false) {
                    return .{ .factor = BoundaryCondition.nuemann(0.0) };
                } else {
                    const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                    return .{ .factor = BoundaryCondition.robin(1.0 / r, 1.0, 0.0) };
                }
            }

            pub fn boundaryCtx(_: MetricOperator, pos: [N]f64, face: Face) dofs.SystemBoundaryCondition(Context) {
                if (face.side == false) {
                    return .{ .seed = BoundaryCondition.nuemann(0.0) };
                } else {
                    const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                    return .{ .seed = BoundaryCondition.robin(1.0 / r, 1.0, 0.0) };
                }
            }
        };

        // Run

        fn run(allocator: Allocator) !void {
            std.debug.print("Running Elliptic Solver\n", .{});

            var grid = try Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, 0.0 },
                    .size = [2]f64{ 10.0, 10.0 },
                },
                .tile_width = 16,
                .index_size = [2]usize{ 1, 1 },
            });
            defer grid.deinit();

            // Globally refine three times

            for (0..3) |_| {
                var tags = try allocator.alloc(bool, grid.tile_total);
                defer allocator.free(tags);

                @memset(tags, true);

                try grid.regrid(allocator, tags, .{
                    .max_levels = 4,
                    .patch_efficiency = 0.1,
                    .patch_max_tiles = 100,
                    .block_efficiency = 0.7,
                    .block_max_tiles = 100,
                });
            }

            // Build maps

            const block_map: []usize = try allocator.alloc(usize, grid.tile_total);
            defer allocator.free(block_map);

            grid.buildBlockMap(block_map);

            const dof_map: DofMap = try DofMap.init(allocator, &grid);
            defer dof_map.deinit(allocator);

            std.debug.print("NDofs: {}\n", .{grid.cell_total});

            // Build functions

            var seed = try SystemSlice(Seed).init(allocator, grid.cell_total);
            defer seed.deinit(allocator);

            const seed_proj: SeedLaplacianProjection = .{
                .amplitude = 1.0,
                .sigma = 1.0,
            };

            DofUtils.projectCells(&grid, seed_proj, seed);

            var metric = try SystemSlice(Metric).init(allocator, grid.cell_total);
            defer metric.deinit(allocator);

            const rhs = SystemSlice(Metric).view(grid.cell_total, .{ .factor = seed.field(.seed) });

            const oper = MetricOperator{};

            var base_solver = try BiCGStabSolver.init(allocator, grid.blocks[0].cell_total, 10000, 10e-10);
            defer base_solver.deinit();

            var solver = MultigridSolver.init(1, 10e-10, &base_solver);
            defer solver.deinit();

            try solver.solve(
                allocator,
                &grid,
                block_map,
                dof_map,
                oper,
                metric,
                rhs.toConst(),
                seed.toConst(),
            );

            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/seed.vtu", .{});
            defer file.close();

            const Output = enum {
                seed,
                metric,
            };

            const output = SystemSliceConst(Output).view(grid.cell_total, .{
                .seed = seed.field(.seed),
                .metric = metric.field(.factor),
            });

            try DofUtils.writeCellsToVtk(Output, allocator, &grid, output, file.writer());
        }
    };
}

pub fn PoissonEquation(comptime O: usize) type {
    const N = 2;
    return struct {
        const BoundaryCondition = dofs.BoundaryCondition;
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const MultigridSolver = dofs.MultigridSolver(N, O, BiCGStabSolver);
        const SystemSlice = system.SystemSlice;
        const SystemSliceConst = system.SystemSliceConst;

        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);

        const BiCGStabSolver = lac.BiCGStablSolver(2);

        const Mesh = mesh.Mesh(N);

        pub const Empty = enum {};

        pub const Function = enum {
            func,
        };

        // Seed Function
        pub const RhsProjection = struct {
            amplitude: f64,

            pub const System = Function;

            pub fn project(self: RhsProjection, pos: [N]f64) system.SystemValue(Function) {
                return .{
                    .func = self.amplitude * std.math.sin(pos[0]) * std.math.sin(pos[1]),
                };
            }
        };

        pub const PoissonOperator = struct {
            pub const Context = Empty;
            pub const System = Function;

            pub fn apply(_: PoissonOperator, engine: dofs.OperatorEngine(N, O, Context, System)) system.SystemValue(System) {
                const lap = engine.laplacianSys(.func);

                return .{
                    .func = -lap,
                };
            }

            pub fn applyDiagonal(_: PoissonOperator, engine: dofs.OperatorEngine(N, O, Context, System)) system.SystemValue(System) {
                return .{
                    .func = -engine.laplacianDiagonal(),
                };
            }

            pub fn boundarySys(_: PoissonOperator, _: [N]f64, _: Face) dofs.SystemBoundaryCondition(System) {
                return .{
                    .func = BoundaryCondition.diritchlet(0.0),
                };
            }

            pub fn boundaryCtx(_: PoissonOperator, _: [N]f64, _: Face) dofs.SystemBoundaryCondition(Context) {
                return .{};
            }
        };

        // Run

        fn run(allocator: Allocator) !void {
            std.debug.print("Running Poisson Elliptic Solver\n", .{});

            var grid = try Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, 0.0 },
                    .size = [2]f64{ std.math.pi, std.math.pi },
                },
                .tile_width = 16,
                .index_size = [2]usize{ 1, 1 },
            });
            defer grid.deinit();

            // Globally refine three times

            for (0..1) |_| {
                var tags = try allocator.alloc(bool, grid.tile_total);
                defer allocator.free(tags);

                @memset(tags, true);

                try grid.regrid(allocator, tags, .{
                    .max_levels = 4,
                    .patch_efficiency = 0.1,
                    .patch_max_tiles = 100,
                    .block_efficiency = 0.7,
                    .block_max_tiles = 100,
                });
            }

            for (grid.blocks) |block| {
                std.debug.print("Block {}\n", .{block});
            }

            // Build maps

            const block_map: []usize = try allocator.alloc(usize, grid.tile_total);
            defer allocator.free(block_map);

            grid.buildBlockMap(block_map);

            const dof_map: DofMap = try DofMap.init(allocator, &grid);
            defer dof_map.deinit(allocator);

            std.debug.print("NDofs: {}\n", .{grid.cell_total});

            // Build functions

            var rhs = try SystemSlice(Function).init(allocator, grid.cell_total);
            defer rhs.deinit(allocator);

            var rhs_proj: RhsProjection = .{ .amplitude = 2.0 };

            DofUtils.projectCells(&grid, rhs_proj, rhs);

            var sol = try SystemSlice(Function).init(allocator, grid.cell_total);
            defer sol.deinit(allocator);

            rhs_proj.amplitude = 1.0;

            DofUtils.projectCells(&grid, rhs_proj, sol);

            var err = try SystemSlice(Function).init(allocator, grid.cell_total);
            defer err.deinit(allocator);

            var numerical = try SystemSlice(Function).init(allocator, grid.cell_total);
            defer numerical.deinit(allocator);

            @memset(numerical.field(.func), 0.0);

            const oper = PoissonOperator{};

            var base_solver = try BiCGStabSolver.init(allocator, grid.blocks[0].cell_total, 10000, 10e-10);
            defer base_solver.deinit();

            var solver = MultigridSolver.init(1, 10e-10, &base_solver);
            defer solver.deinit();

            try solver.solve(
                allocator,
                &grid,
                block_map,
                dof_map,
                oper,
                numerical,
                rhs.toConst(),
                SystemSliceConst(Empty).view(grid.cell_total, .{}),
            );

            for (0..grid.cell_total) |i| {
                err.field(.func)[i] = numerical.field(.func)[i] - sol.field(.func)[i];
            }

            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/poisson.vtu", .{});
            defer file.close();

            const Output = enum {
                num,
                exact,
                err,
                rhs,
            };

            const output = SystemSliceConst(Output).view(grid.cell_total, .{
                .num = numerical.field(.func),
                .exact = sol.field(.func),
                .err = err.field(.func),
                .rhs = rhs.field(.func),
            });

            try DofUtils.writeCellsToVtk(Output, allocator, &grid, output, file.writer());
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
    // try ScalarFieldProblem(2).run(gpa.allocator());

    try PoissonEquation(1).run(gpa.allocator());
}

test {
    _ = basis;
    _ = geometry;
    _ = mesh;
    _ = vtkio;
    _ = lac;
}
