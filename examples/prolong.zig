// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");
const dofs = aeon.dofs;
const geometry = aeon.geometry;

pub fn Prolongation(comptime O: usize) type {
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
        pub const Projection = struct {
            amplitude: f64,

            pub const System = Function;
            pub const Context = aeon.EmptySystem;

            pub fn project(self: Projection, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const pos = engine.position();

                return .{
                    .func = self.amplitude * std.math.sin(pos[0]) * std.math.sin(pos[1]),
                };
            }
        };

        pub const Boundary = struct {
            pub const System = Function;

            pub fn boundary(self: Boundary, pos: [N]f64, face: Face) dofs.SystemBoundaryCondition(System) {
                _ = face;
                _ = pos;
                _ = self;
                return .{
                    .func = BoundaryCondition.diritchlet(0.0),
                };
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
                .tile_width = 16,
                .index_size = [2]usize{ 1, 1 },
            });
            defer mesh.deinit();

            // Globally refine three times

            for (0..1) |_| {
                var tags = try allocator.alloc(bool, mesh.tile_total);
                defer allocator.free(tags);

                @memset(tags, true);

                try mesh.regrid(allocator, tags, .{
                    .max_levels = 4,
                    .patch_efficiency = 0.1,
                    .patch_max_tiles = 1000,
                    .block_efficiency = 0.7,
                    .block_max_tiles = 1000,
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

            const exact = try SystemSlice(Function).init(allocator, mesh.cell_total);
            defer exact.deinit(allocator);

            DofUtilsTotal.project(
                &mesh,
                dof_map,
                Projection{ .amplitude = 1.0 },
                exact,
                aeon.EmptySystem.sliceConst(),
            );

            const func = try SystemSlice(Function).init(allocator, mesh.cell_total);
            defer func.deinit(allocator);

            DofUtilsTotal.project(
                &mesh,
                dof_map,
                Projection{ .amplitude = 1.0 },
                func,
                aeon.EmptySystem.sliceConst(),
            );

            const func_dofs = try SystemSlice(Function).init(allocator, dof_map.ndofs());
            defer func_dofs.deinit(allocator);

            for (0..mesh.blocks.len) |block_id| {
                DofUtils.copyDofsFromCells(
                    Function,
                    &mesh,
                    dof_map,
                    block_id,
                    func_dofs,
                    func.toConst(),
                );
            }

            for (1..mesh.levels.len) |level_id| {
                const coarse = mesh.levels[level_id - 1];
                const level = mesh.levels[level_id];

                for (coarse.block_offset..coarse.block_offset + coarse.block_total) |block_id| {
                    DofUtils.fillBoundary(
                        &mesh,
                        dof_map,
                        block_id,
                        Boundary{},
                        func_dofs,
                    );
                }

                for (level.block_offset..level.block_offset + level.block_total) |block_id| {
                    DofUtils.prolong(
                        Function,
                        &mesh,
                        dof_map,
                        block_id,
                        func_dofs,
                    );
                }
            }

            for (0..mesh.blocks.len) |block_id| {
                DofUtils.copyCellsFromDofs(
                    Function,
                    &mesh,
                    dof_map,
                    block_id,
                    func,
                    func_dofs.toConst(),
                );
            }

            const err = try SystemSlice(Function).init(allocator, mesh.cell_total);
            defer err.deinit(allocator);

            for (0..mesh.cell_total) |i| {
                err.field(.func)[i] = func.field(.func)[i] - exact.field(.func)[i];
            }

            // Output results

            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/prolong.vtu", .{});
            defer file.close();

            const Output = enum {
                exact,
                func,
                err,
            };

            const output = SystemSliceConst(Output).view(mesh.cell_total, .{
                .func = func.field(.func),
                .exact = exact.field(.func),
                .err = err.field(.func),
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
    try Prolongation(2).run(gpa.allocator());
}
