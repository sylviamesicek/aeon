// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");
const dofs = aeon.dofs;
const geometry = aeon.geometry;
const lac = aeon.lac;
const mesh = aeon.mesh;
const methods = aeon.methods;

pub fn PoissonEquation(comptime O: usize) type {
    const N = 2;
    return struct {
        const BoundaryCondition = dofs.BoundaryCondition;
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const DofUtilsTotal = dofs.DofUtilsTotal(N, O);
        const MultigridMethod = methods.MultigridMethod(N, O, BiCGStabSolver);
        const SystemSlice = aeon.SystemSlice;
        const SystemSliceConst = aeon.SystemSliceConst;
        const SystemValue = aeon.SystemValue;
        const SystemBoundaryCondition = dofs.SystemBoundaryCondition;

        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = aeon.Index(N);

        const BiCGStabSolver = lac.BiCGStabSolver;

        const Mesh = mesh.Mesh(N);

        pub const Function = enum {
            func,
        };

        // Seed Function
        pub const RhsProjection = struct {
            amplitude: f64,

            pub const System = Function;
            pub const Context = aeon.EmptySystem;

            pub fn project(self: RhsProjection, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(Function) {
                const pos = engine.position();

                return .{
                    .func = self.amplitude * std.math.sin(pos[0]) * std.math.sin(pos[1]),
                    // .func = self.amplitude,
                };
            }

            pub fn boundaryCtx(_: RhsProjection, _: [N]f64, _: Face) SystemBoundaryCondition(Context) {
                return .{};
            }
        };

        pub const PoissonOperator = struct {
            pub const System = Function;
            pub const Context = aeon.EmptySystem;

            pub fn apply(_: PoissonOperator, engine: dofs.Engine(N, O, System, Context)) aeon.SystemValue(System) {
                return .{
                    .func = -engine.laplacianSys(.func),
                };
            }

            pub fn applyDiagonal(_: PoissonOperator, engine: dofs.Engine(N, O, System, Context)) aeon.SystemValue(System) {
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
                    .size = [2]f64{ 2.0 * std.math.pi, 2.0 * std.math.pi },
                },
                .tile_width = 32,
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

            const dof_map: DofMap = try DofMap.init(allocator, &grid);
            defer dof_map.deinit(allocator);

            std.debug.print("NDofs: {}\n", .{grid.cell_total});

            // Build functions

            var rhs = try SystemSlice(Function).init(allocator, grid.cell_total);
            defer rhs.deinit(allocator);

            var rhs_proj: RhsProjection = .{ .amplitude = 2.0 };

            DofUtilsTotal.projectCells(&grid, dof_map, rhs_proj, rhs, aeon.EmptySystem.sliceConst());

            var sol = try SystemSlice(Function).init(allocator, grid.cell_total);
            defer sol.deinit(allocator);

            rhs_proj.amplitude = 1.0;

            DofUtilsTotal.projectCells(&grid, dof_map, rhs_proj, sol, aeon.EmptySystem.sliceConst());

            var err = try SystemSlice(Function).init(allocator, grid.cell_total);
            defer err.deinit(allocator);

            var numerical = try SystemSlice(Function).init(allocator, grid.cell_total);
            defer numerical.deinit(allocator);

            @memset(numerical.field(.func), 0.0);

            const oper = PoissonOperator{};

            var solver = MultigridMethod.new(20, 10e-10, BiCGStabSolver.new(10000, 10e-10));

            try solver.solve(
                allocator,
                &grid,
                dof_map,
                oper,
                numerical,
                rhs.toConst(),
                aeon.EmptySystem.sliceConst(),
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
    try PoissonEquation(2).run(gpa.allocator());
}
