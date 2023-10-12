// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");
const dofs = aeon.dofs;
const geometry = aeon.geometry;
const index = aeon.index;
const lac = aeon.lac;
const mesh = aeon.mesh;
const methods = aeon.methods;

// ***********************
// Temp Main Code ********
// ***********************

pub fn BrillInitialData(comptime O: usize) type {
    const N = 2;
    return struct {
        const BoundaryCondition = dofs.BoundaryCondition;
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const LinearMapMethod = methods.LinearMapMethod(N, O, BiCGStabSolver);
        const SystemSlice = aeon.SystemSlice;
        const SystemSliceConst = aeon.SystemSliceConst;
        const SystemValue = aeon.SystemValue;
        const SystemBoundaryCondition = dofs.SystemBoundaryCondition;

        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);

        const BiCGStabSolver = lac.BiCGStabSolver;

        const Mesh = mesh.Mesh(N);

        pub const Seed = enum {
            seed,
        };

        pub const Metric = enum {
            factor,
        };

        pub const SeedProjection = struct {
            amplitude: f64,
            sigma: f64,

            pub const System = Seed;

            pub fn project(self: SeedProjection, pos: [N]f64) SystemValue(System) {
                const rho = pos[0];
                const z = pos[1];

                const rho2 = rho * rho;
                const z2 = z * z;
                const sigma2 = self.sigma * self.sigma;

                const term1: f64 = 0.5 * self.amplitude;
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

            pub fn apply(_: MetricOperator, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
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

            pub fn applyDiagonal(_: MetricOperator, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
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

            pub fn boundarySys(_: MetricOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                if (face.side == false) {
                    return .{ .factor = BoundaryCondition.nuemann(0.0) };
                } else {
                    const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                    return .{ .factor = BoundaryCondition.robin(1.0 / r, 1.0, 0.0) };
                }
            }

            pub fn boundaryCtx(_: MetricOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(Context) {
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
            std.debug.print("Running Brill Initial Data Solver\n", .{});

            var grid = try Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, 0.0 },
                    .size = [2]f64{ 10.0, 10.0 },
                },
                .tile_width = 128,
                .index_size = [2]usize{ 1, 1 },
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

            const seed_proj: SeedProjection = .{
                .amplitude = 1.0,
                .sigma = 1.0,
            };

            DofUtils.projectCells(&grid, seed_proj, seed);

            var metric = try SystemSlice(Metric).init(allocator, grid.cell_total);
            defer metric.deinit(allocator);

            const rhs = SystemSlice(Metric).view(grid.cell_total, .{ .factor = seed.field(.seed) });

            const oper = MetricOperator{};

            var solver = LinearMapMethod.new(BiCGStabSolver.new(1000000, 10e-12));

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

// pub fn LetsFindABug(comptime O: usize) type {
//     const N = 2;
//     return struct {
//         const BoundaryCondition = dofs.BoundaryCondition;
//         const DofMap = dofs.DofMap(N, O);
//         const DofUtils = dofs.DofUtils(N, O);
//         const SystemSlice = system.SystemSlice;
//         const SystemSliceConst = system.SystemSliceConst;

//         const Face = geometry.Face(N);
//         const IndexSpace = geometry.IndexSpace(N);
//         const Index = index.Index(N);

//         const Mesh = mesh.Mesh(N);

//         pub const Empty = enum {};

//         pub const Function = enum {
//             func,
//         };

//         // Seed Function
//         pub const Projection = struct {
//             pub const System = Function;

//             pub fn project(_: Projection, pos: [N]f64) system.SystemValue(Function) {
//                 return .{
//                     .func = std.math.sin(pos[0]) * std.math.sin(pos[1]),
//                 };
//             }
//         };

//         const Diritchlet = struct {
//             pub const System = Function;

//             pub fn boundary(_: @This(), _: [2]f64, _: Face) dofs.SystemBoundaryCondition(Function) {
//                 return .{
//                     .func = BoundaryCondition.diritchlet(0.0),
//                 };
//             }
//         };

//         // Run

//         fn run(allocator: Allocator) !void {
//             std.debug.print("Running Poisson Elliptic Solver\n", .{});

//             var grid = try Mesh.init(allocator, .{
//                 .physical_bounds = .{
//                     .origin = [2]f64{ 0.0, 0.0 },
//                     .size = [2]f64{ 2.0 * std.math.pi, 2.0 * std.math.pi },
//                 },
//                 .tile_width = 16,
//                 .index_size = [2]usize{ 1, 1 },
//             });
//             defer grid.deinit();

//             for (grid.blocks) |block| {
//                 std.debug.print("Block {}\n", .{block});
//             }

//             // Build maps

//             const block_map: []usize = try allocator.alloc(usize, grid.tile_total);
//             defer allocator.free(block_map);

//             grid.buildBlockMap(block_map);

//             const dof_map: DofMap = try DofMap.init(allocator, &grid);
//             defer dof_map.deinit(allocator);

//             std.debug.print("NDofs: {}\n", .{grid.cell_total});

//             // Build functions

//             var func = try SystemSlice(Function).init(allocator, grid.cell_total);
//             defer func.deinit(allocator);

//             DofUtils.projectCells(&grid, Projection{}, func);

//             var func_dofs = try SystemSlice(Function).init(allocator, dof_map.ndofs());
//             defer func_dofs.deinit(allocator);

//             DofUtils.copyDofsFromCells(Function, &grid, dof_map, 0, func_dofs, func.toConst());

//             DofUtils.fillBoundaryFull(&grid, block_map, dof_map, 0, Diritchlet{}, func_dofs);

//             std.debug.print("Writing Solution To File\n", .{});

//             // const stencil_space = DofUtils.blockStencilSpace(&grid, 0);
//             // _ = stencil_space;

//             // const field = func_dofs.slice(dof_map.offset(0), dof_map.total(0)).field(.func);
//             // _ = field;

//             // std.debug.print("Extent 1: {}\n", .{
//             //     stencil_space.boundaryValue([2]isize{ -1, -1 }, 2 * O + 1, [2]isize{ 0, 0 }, field),
//             // });
//             // std.debug.print("Extent 2: {}\n", .{
//             //     stencil_space.boundaryValue([2]isize{ -2, -2 }, 2 * O + 1, [2]isize{ 0, 0 }, field),
//             // });

//             // std.debug.print("Extent Coef 1: {}\n", .{
//             //     stencil_space.boundaryValueCoef([2]isize{ -1, -1 }, 2 * O + 1),
//             // });
//             // std.debug.print("Extent Coef 2: {}\n", .{
//             //     stencil_space.boundaryValueCoef([2]isize{ -2, -2 }, 2 * O + 1),
//             // });

//             const file = try std.fs.cwd().createFile("output/error.vtu", .{});
//             defer file.close();

//             try DofUtils.writeDofsToVtk(Function, allocator, &grid, 0, dof_map, func_dofs.toConst(), file.writer());
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
    try BrillInitialData(2).run(gpa.allocator());
}
