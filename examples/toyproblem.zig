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

        pub const Metric = enum {
            factor,
        };

        pub const RhsProjection = struct {
            pub const System = Metric;
            pub const Context = aeon.EmptySystem;

            pub fn project(_: RhsProjection, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const pos = engine.position();

                const r = pos[0];
                const z = pos[1];

                return .{
                    .factor = @cos(z) * (4 * @cos(r) - 5 * @sin(r) * r),
                };
            }
        };

        pub const ExactProjection = struct {
            pub const System = Metric;
            pub const Context = aeon.EmptySystem;

            pub fn project(_: ExactProjection, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const pos = engine.position();

                const r = pos[0];
                const z = pos[1];

                return .{
                    .factor = @cos(z) * @cos(r) * r * r,
                };
            }
        };

        pub const MetricOperator = struct {
            pub const System = Metric;
            pub const Context = aeon.EmptySystem;

            pub fn apply(_: MetricOperator, comptime Setting: dofs.EngineSetting, engine: dofs.Engine(N, O, Setting, System, Context)) SystemValue(System) {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianSys(.factor);
                const gradient: [N]f64 = engine.gradientSys(.factor);
                const value: f64 = engine.valueSys(.factor);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                return .{
                    .factor = lap + 2 * value,
                };
            }

            pub fn boundarySys(_: MetricOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                const r = pos[0];
                const z = pos[1];

                const r_deriv = -@cos(z) * r * (r * @sin(r) - 2 * @sin(r));
                _ = r_deriv;
                const z_deriv = -r * r * @cos(r) * @sin(z);
                _ = z_deriv;

                if (face.side == false) {
                    return .{
                        .factor = BoundaryCondition.nuemann(0.0),
                    };
                } else {
                    return .{
                        .factor = BoundaryCondition.diritchlet(0.0),
                    };
                }

                // if (face.axis == 0) {
                //     return .{
                //         .factor = BoundaryCondition.nuemann(r_deriv),
                //     };
                // } else {
                //     return .{
                //         .factor = BoundaryCondition.nuemann(z_deriv),
                //     };
                // }
            }

            pub fn boundaryCtx(_: MetricOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(Context) {
                _ = face;
                _ = pos;
                return .{};
            }
        };

        // Run

        fn run(allocator: Allocator) !void {
            std.debug.print("Running Brill Initial Data Solver\n", .{});

            var grid = try Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, 0.0 },
                    .size = [2]f64{ std.math.pi / 2.0, std.math.pi / 2.0 },
                },
                .tile_width = 1024,
                .index_size = [2]usize{ 1, 1 },
            });
            defer grid.deinit();

            // Build maps

            const dof_map: DofMap = try DofMap.init(allocator, &grid);
            defer dof_map.deinit(allocator);

            std.debug.print("NDofs: {}\n", .{grid.cell_total});

            // Build functions

            var rhs = try SystemSlice(Metric).init(allocator, grid.cell_total);
            defer rhs.deinit(allocator);

            DofUtilsTotal.project(&grid, dof_map, RhsProjection{}, rhs, aeon.EmptySystem.sliceConst());

            var metric = try SystemSlice(Metric).init(allocator, grid.cell_total);
            defer metric.deinit(allocator);

            const oper = MetricOperator{};

            var solver = LinearMapMethod.new(BiCGStabSolver.new(1000000, 10e-15));

            try solver.solve(
                allocator,
                &grid,
                dof_map,
                oper,
                metric,
                aeon.EmptySystem.sliceConst(),
                rhs.toConst(),
            );

            var exact = try SystemSlice(Metric).init(allocator, grid.cell_total);
            defer exact.deinit(allocator);

            DofUtilsTotal.project(&grid, dof_map, ExactProjection{}, exact, aeon.EmptySystem.sliceConst());

            var err = try SystemSlice(Metric).init(allocator, grid.cell_total);
            defer err.deinit(allocator);

            for (0..grid.cell_total) |i| {
                err.field(.factor)[i] = exact.field(.factor)[i] - metric.field(.factor)[i];
            }

            var residual: f64 = 0.0;

            for (0..grid.cell_total) |i| {
                const f = err.field(.factor)[i];
                // residual += f * f;
                residual = @max(residual, f * f);
            }

            std.debug.print("Residual: {}\n", .{@sqrt(residual)});

            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/toyproblem.vtu", .{});
            defer file.close();

            const Output = enum {
                exact,
                metric,
                err,
                rhs,
            };

            const output = SystemSliceConst(Output).view(grid.cell_total, .{
                .exact = exact.field(.factor),
                .metric = metric.field(.factor),
                .err = err.field(.factor),
                .rhs = rhs.field(.factor),
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
