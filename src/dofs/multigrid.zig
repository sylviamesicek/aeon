const std = @import("std");
const Allocator = std.mem.Allocator;

const basis = @import("../basis/basis.zig");
const lac = @import("../lac/lac.zig");
const system = @import("../system.zig");
const meshes = @import("../mesh/mesh.zig");
const geometry = @import("../geometry/geometry.zig");
const index = @import("../index.zig");

const boundary = @import("boundary.zig");
const chunks = @import("chunks.zig");
const dofs = @import("dofs.zig");
const operator = @import("operator.zig");

/// A multigrid based elliptic solver which uses the given base solver to approximate the solution
/// on the lowest level of the mesh.
pub fn MultigridSolver(comptime N: usize, comptime O: usize, comptime BaseSolver: type) type {
    if (comptime !lac.isLinearSolver(BaseSolver)) {
        @compileError("Base solver must satisfy the a linear solver requirement.");
    }

    return struct {
        mesh: *const Mesh,
        block_map: []const usize,
        base_solver: *const BaseSolver,

        const Self = @This();
        const Mesh = meshes.Mesh(N);
        const DofUtils = dofs.DofUtils(N, O);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);
        const CellSpace = basis.CellSpace(N, O, O);
        const StencilSpace = basis.StencilSpace(N, O, O);
        const SystemChunk = chunks.SystemChunk;

        pub fn init(mesh: *const Mesh, block_map: []const usize, base_solver: *BaseSolver) Self {
            return .{
                .mesh = mesh,
                .block_map = block_map,
                .base_solver = base_solver,
            };
        }

        pub fn deinit(_: *Self) void {}

        pub fn solve(
            self: *Self,
            oper: anytype,
            x: system.SystemSlice(@TypeOf(oper).System),
            b: system.SystemSliceConst(@TypeOf(oper).System),
            ctx: system.SystemSliceConst(@TypeOf(oper).Context),
            x_chunk: SystemChunk(@TypeOf(oper).System),
            b_chunk: SystemChunk(@TypeOf(oper).System),
            ctx_chunk: SystemChunk(@TypeOf(oper).Context),
        ) void {
            _ = b_chunk;
            // Alias
            const T = @TypeOf(oper);

            if (comptime !(operator.isMeshOperator(N, O)(T))) {
                @compileError("Oper must satisfy isMeshOperator traits.");
            }

            if (comptime system.systemFieldCount(T.System) != 1) {
                @compileError("The multigrid solver only supports systems with 1 field currently.");
            }

            const field_name: []const u8 = comptime system.systemFieldNames(T.System)[0];
            const x_field: []f64 = @field(x, field_name);
            const b_field: []const f64 = @field(b, field_name);

            // Solve system using the base solver
            const base_linear_map: BaseLinearMap(T) = .{
                .self = self,
                .oper = oper,
                .ctx = ctx,
                .ctx_chunk = ctx_chunk,
                .x_chunk = x_chunk,
            };

            self.base_solver.solve(base_linear_map, x_field, b_field);
        }

        fn BaseLinearMap(comptime T: type) type {
            const field_name: []const u8 = comptime system.systemFieldNames(T.System)[0];

            return struct {
                self: *const Self,
                oper: T,
                ctx: system.SystemSliceConst(T.Context),
                ctx_chunk: SystemChunk(T.Context),
                x_chunk: SystemChunk(T.System),

                pub fn apply(wrapper: *const @This(), output: []f64, input: []const f64) void {
                    const mesh = wrapper.self.mesh;
                    const block_map = wrapper.self.block_map;
                    const ctx_chunk = wrapper.ctx_chunk;
                    const x_chunk = wrapper.x_chunk;

                    // Aliases
                    const stencil_space = DofUtils.blockStencilSpace(mesh, 0, 0);

                    // Fill base boundary and inner
                    {
                        var sys: system.SystemSliceConst(T.System) = undefined;
                        @field(sys, field_name) = input;

                        DofUtils.fillInterior(
                            T.System,
                            mesh,
                            0,
                            0,
                            x_chunk,
                            sys,
                        );
                        DofUtils.fillBoundary(
                            mesh,
                            0,
                            0,
                            block_map,
                            DofUtils.operSystemBoundary(wrapper.oper),
                            x_chunk,
                            sys,
                        );

                        DofUtils.fillInterior(
                            T.Context,
                            mesh,
                            0,
                            0,
                            ctx_chunk,
                            wrapper.ctx,
                        );
                        DofUtils.fillBoundary(
                            mesh,
                            0,
                            0,
                            block_map,
                            DofUtils.operContextBoundary(wrapper.oper),
                            ctx_chunk,
                            wrapper.ctx,
                        );
                    }

                    // Apply operator
                    {
                        var cells = stencil_space.cellSpace().cells();
                        var linear: usize = 0;

                        while (cells.next()) |cell| : (linear += 1) {
                            const engine = operator.EngineType(N, O, T){
                                .inner = .{
                                    .space = stencil_space,
                                    .cell = cell,
                                },
                                .ctx = ctx_chunk.sliceConst(stencil_space.cellSpace().total()),
                                .sys = x_chunk.sliceConst(stencil_space.cellSpace().total()),
                            };

                            output[linear] = @field(wrapper.oper.apply(engine), field_name);
                        }
                    }
                }

                // var iterations: usize = 0;

                pub fn callback(_: *const @This(), iteration: usize, residual: f64, _: []const f64) void {
                    std.debug.print("Iteration: {}, Residual: {}\n", .{ iteration, residual });

                    // const file_name = std.fmt.allocPrint(solver.self.mesh.gpa, "output/elliptic_iteration{}.vtu", .{iterations}) catch {
                    //     unreachable;
                    // };

                    // const file = std.fs.cwd().createFile(file_name, .{}) catch {
                    //     unreachable;
                    // };
                    // defer file.close();

                    // DofUtils.writeVtk(solver.self.mesh.gpa, solver.self.mesh, .{ .metric = x }, file.writer()) catch {
                    //     unreachable;
                    // };

                    // iterations += 1;
                }
            };
        }
    };
}
