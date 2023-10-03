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
                .index_size = [2]usize{ 8, 8 },
            });
            defer grid.deinit();

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

            var base_solver = try BiCGStabSolver.init(allocator, grid.getLevel(0).cell_total, 10000, 10e-10);
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

            try DofUtils.writeVtk(Output, allocator, &grid, output, file.writer());
        }
    };
}

// pub fn PoissonEquation(comptime O: usize) type {
//     const N = 2;
//     return struct {
//         const math = std.math;

//         const BoundaryCondition = dofs.BoundaryCondition;
//         const DofHandler = dofs.DofHandler(N, O);
//         const MultigridSolver = dofs.MultigridSolver(N, O, BiCGStabSolver);

//         const Face = geometry.Face(N);
//         const IndexSpace = geometry.IndexSpace(N);
//         const Index = index.Index(N);

//         const BiCGStabSolver = lac.BiCGStablSolver(2);

//         const Mesh = mesh.Mesh(N, O);

//         pub const MySystem = enum {
//             sys,
//         };

//         // Solution Function
//         pub const SolutionFunction = struct {
//             pub const Input = enum {};
//             pub const Output = MySystem;

//             pub fn value(_: SolutionFunction, engine: dofs.FunctionEngine(N, O, Input)) system.SystemValue(Output) {
//                 const pos = engine.position();

//                 return .{
//                     .sys = math.sin(pos[0]) * math.sin(pos[1]),
//                     // .sys = (1.0 - pos[0] * pos[0]) * (1.0 - pos[1] * pos[1]),
//                     // .sys = 0.0,
//                 };
//             }
//         };

//         // Function
//         pub const RhsFunction = struct {
//             pub const Input = enum {};
//             pub const Output = MySystem;

//             pub fn value(_: RhsFunction, engine: dofs.FunctionEngine(N, O, Input)) system.SystemValue(Output) {
//                 const pos = engine.position();

//                 return .{
//                     .sys = -2 * math.sin(pos[0]) * math.sin(pos[1]),

//                     // .sys = 1.0,
//                 };
//             }
//         };

//         pub const PoissonOperator = struct {
//             pub const Context = enum {};
//             pub const System = MySystem;

//             pub fn apply(_: PoissonOperator, engine: dofs.OperatorEngine(N, O, Context, System)) system.SystemValue(System) {
//                 return .{
//                     .sys = engine.laplacianSys(.sys),
//                 };
//             }

//             pub fn applyDiagonal(_: PoissonOperator, engine: dofs.OperatorEngine(N, O, Context, System)) system.SystemValue(System) {
//                 return .{
//                     .sys = engine.laplacianDiagonal(),
//                 };
//             }

//             pub fn condition(_: PoissonOperator, _: [N]f64, _: Face) dofs.SystemBoundaryCondition(System) {
//                 return .{
//                     .sys = BoundaryCondition.diritchlet(0.0),
//                 };
//             }
//         };

//         // Run

//         fn run(allocator: Allocator) !void {
//             var grid = Mesh.init(allocator, .{
//                 .physical_bounds = .{
//                     .origin = [1]f64{0.0} ** N,
//                     .size = [1]f64{2.0 * math.pi} ** N,
//                 },
//                 .tile_width = 16,
//                 .index_size = [1]usize{2} ** N,
//             });
//             defer grid.deinit();

//             var dof_handler = DofHandler.init(allocator, &grid);
//             defer dof_handler.deinit();

//             const ndofs: usize = dof_handler.ndofs();
//             const ndofs_reduced = IndexSpace.fromSize(Index.scaled(grid.base.index_size, grid.tile_width)).total();
//             _ = ndofs_reduced;

//             // Projection solution
//             var solution: []f64 = try allocator.alloc(f64, ndofs);
//             defer allocator.free(solution);

//             dof_handler.project(SolutionFunction{}, .{ .sys = solution }, .{});

//             // Right hand side
//             var rhs: []f64 = try allocator.alloc(f64, ndofs);
//             defer allocator.free(rhs);

//             dof_handler.project(RhsFunction{}, .{ .sys = rhs }, .{});

//             // Operator
//             const oper = PoissonOperator{};

//             dof_handler.fillBaseBoundary(oper, .{ .sys = solution });
//             dof_handler.fillBaseBoundary(oper, .{ .sys = rhs });

//             // Solution vectors
//             var numerical_from_apply: []f64 = try allocator.alloc(f64, ndofs);
//             defer allocator.free(numerical_from_apply);

//             var numerical_from_project: []f64 = try allocator.alloc(f64, ndofs);
//             defer allocator.free(numerical_from_project);

//             dof_handler.project(RhsFunction{}, .{ .sys = numerical_from_project }, .{});
//             dof_handler.apply(oper, .{ .sys = numerical_from_apply }, .{ .sys = solution }, .{});
//             // dof_handler.fillBaseBoundary(oper, .{ .sys = numerical });

//             var err: []f64 = try allocator.alloc(f64, ndofs);
//             defer allocator.free(err);

//             for (0..ndofs) |i| {
//                 err[i] = numerical_from_apply[i] - rhs[i];
//             }

//             // var apply: []f64 = try allocator.alloc(f64, ndofs);
//             // defer allocator.free(apply);

//             // dof_handler.fillBaseBoundary(oper, .{ .sys = numerical });
//             // dof_handler.apply(oper, .{ .sys = apply }, .{ .sys = numerical }, .{});

//             // Solver
//             // var base_solver = try BiCGStabSolver.init(allocator, ndofs_reduced, 1000, 1e-10);
//             // defer base_solver.deinit();

//             // var solver = try MultigridSolver.init(allocator, &dof_handler, &base_solver);
//             // defer solver.deinit();

//             // solver.solve(
//             //     oper,
//             //     .{ .sys = numerical },
//             //     .{ .sys = rhs },
//             //     .{},
//             // );

//             const file = try std.fs.cwd().createFile("output/apply.vtu", .{});
//             defer file.close();

//             try dof_handler.writeVtk(true, .{
//                 .numerical_from_apply = numerical_from_apply,
//                 .numerical_from_project = numerical_from_project,
//                 .rhs = rhs,
//                 .solution = solution,
//                 .err = err,
//             }, file.writer());
//         }
//     };
// }

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

    // try ApplyTest(1).run(gpa.allocator());
}

test {
    _ = basis;
    _ = geometry;
    _ = mesh;
    _ = vtkio;
    _ = lac;
}
