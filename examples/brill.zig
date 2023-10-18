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
        const DataOut = aeon.DataOut(N);
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const DofUtilsTotal = dofs.DofUtilsTotal(N, O);
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
            pub const Context = aeon.EmptySystem;

            pub fn project(self: SeedProjection, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const pos = engine.position();

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
            pub const System = Metric;
            pub const Context = Seed;

            pub fn apply(_: MetricOperator, comptime Setting: dofs.EngineSetting, engine: dofs.Engine(N, O, Setting, System, Context)) SystemValue(System) {
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
                .tile_width = 64,
                .index_size = [2]usize{ 1, 1 },
            });
            defer grid.deinit();

            // Build maps

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

            DofUtilsTotal.project(&grid, dof_map, seed_proj, seed, aeon.EmptySystem.sliceConst());

            var metric = try SystemSlice(Metric).init(allocator, grid.cell_total);
            defer metric.deinit(allocator);

            const rhs = SystemSlice(Metric).view(grid.cell_total, .{ .factor = seed.field(.seed) });

            const oper = MetricOperator{};

            var solver = LinearMapMethod.new(BiCGStabSolver.new(1000000, 10e-12));

            try solver.solve(
                allocator,
                &grid,
                dof_map,
                oper,
                metric,
                seed.toConst(),
                rhs.toConst(),
            );

            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/brillinitial.vtu", .{});
            defer file.close();

            const Output = enum {
                seed,
                metric,
            };

            const output = SystemSliceConst(Output).view(grid.cell_total, .{
                .seed = seed.field(.seed),
                .metric = metric.field(.factor),
            });

            try DataOut.writeVtk(Output, allocator, &grid, output, file.writer());
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
