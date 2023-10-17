// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");
const dofs = aeon.dofs;
const geometry = aeon.geometry;

pub fn PoissonEquation(comptime O: usize) type {
    const N = 2;
    return struct {
        const BoundaryCondition = dofs.BoundaryCondition;
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const DofUtilsTotal = dofs.DofUtilsTotal(N, O);
        const DataOut = aeon.DataOut(N);
        const MultigridMethod = aeon.methods.MultigridMethod(N, O, BiCGStabSolver);
        const LinearMapMethod = aeon.methods.LinearMapMethod(N, O, BiCGStabSolver);
        const SystemSlice = aeon.SystemSlice;
        const SystemSliceConst = aeon.SystemSliceConst;
        const SystemValue = aeon.SystemValue;
        const SystemBoundaryCondition = dofs.SystemBoundaryCondition;

        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = aeon.Index(N);

        const BiCGStabSolver = aeon.lac.BiCGStabSolver;

        const Mesh = aeon.mesh.Mesh(N);

        pub const Function = enum {
            func,
        };

        // Seed Function
        pub const RhsProjection = struct {
            amplitude: f64,

            pub const System = Function;
            pub const Context = aeon.EmptySystem;

            pub fn project(self: RhsProjection, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const pos = engine.position();

                return .{
                    .func = self.amplitude * std.math.sin(pos[0]) * std.math.sin(pos[1]),
                    // .func = self.amplitude,
                };
            }
        };

        pub const PoissonOperator = struct {
            pub const System = Function;
            pub const Context = aeon.EmptySystem;

            pub fn apply(
                _: PoissonOperator,
                comptime Setting: dofs.EngineSetting,
                engine: dofs.Engine(N, O, Setting, System, Context),
            ) SystemValue(System) {
                return .{
                    .func = -engine.laplacianSys(.func),
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

            var mesh = try Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, 0.0 },
                    .size = [2]f64{ 2.0 * std.math.pi, 2.0 * std.math.pi },
                },
                .tile_width = 4,
                .index_size = [2]usize{ 1, 1 },
            });
            defer mesh.deinit();

            // Globally refine three times

            for (0..2) |_| {
                var tags = try allocator.alloc(bool, mesh.tile_total);
                defer allocator.free(tags);

                @memset(tags, true);

                try mesh.regrid(allocator, tags, .{
                    .max_levels = 4,
                    .patch_efficiency = 0.1,
                    .patch_max_tiles = 100,
                    .block_efficiency = 0.7,
                    .block_max_tiles = 100,
                });
            }

            for (mesh.blocks) |block| {
                std.debug.print("Block {}\n", .{block});
            }

            // Build maps

            const dof_map: DofMap = try DofMap.init(allocator, &mesh);
            defer dof_map.deinit(allocator);

            std.debug.print("NDofs: {}\n", .{mesh.cell_total});

            // Build functions

            // Project right hand side function

            const rhs = try SystemSlice(Function).init(allocator, mesh.cell_total);
            defer rhs.deinit(allocator);

            DofUtilsTotal.project(
                &mesh,
                dof_map,
                RhsProjection{ .amplitude = 2.0 },
                rhs,
                aeon.EmptySystem.sliceConst(),
            );

            const sol = try SystemSlice(Function).init(allocator, mesh.cell_total);
            defer sol.deinit(allocator);

            // As well as the solution function

            DofUtilsTotal.project(
                &mesh,
                dof_map,
                RhsProjection{ .amplitude = 1.0 },
                sol,
                aeon.EmptySystem.sliceConst(),
            );

            // Allocate memory for the numerical solution, and set initial guess.

            var numerical = try SystemSlice(Function).init(allocator, mesh.cell_total);
            defer numerical.deinit(allocator);

            @memset(numerical.field(.func), 0.0);

            // Run multigrid method

            var solver = MultigridMethod.new(20, 10e-10, BiCGStabSolver.new(10000, 10e-10));

            // var solver = LinearMapMethod.new(BiCGStabSolver.new(10000, 10e-10));

            try solver.solve(
                allocator,
                &mesh,
                dof_map,
                PoissonOperator{},
                numerical,
                aeon.EmptySystem.sliceConst(),
                rhs.toConst(),
            );

            // Compute error

            var err = try SystemSlice(Function).init(allocator, mesh.cell_total);
            defer err.deinit(allocator);

            for (0..mesh.cell_total) |i| {
                err.field(.func)[i] = numerical.field(.func)[i] - sol.field(.func)[i];
            }

            // Output results

            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/poisson.vtu", .{});
            defer file.close();

            const Output = enum {
                num,
                exact,
                err,
                rhs,
            };

            const output = SystemSliceConst(Output).view(mesh.cell_total, .{
                .num = numerical.field(.func),
                .exact = sol.field(.func),
                .err = err.field(.func),
                .rhs = rhs.field(.func),
            });

            try DataOut.writeVtk(Output, allocator, &mesh, output, file.writer());
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
    try PoissonEquation(1).run(gpa.allocator());
}
