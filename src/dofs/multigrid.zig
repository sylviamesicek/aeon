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

/// A multigrid based elliptic solver.
pub fn MultigridSolver(comptime N: usize, comptime O: usize, comptime BaseSolver: type) type {
    if (comptime !lac.isLinearSolver(BaseSolver)) {
        @compileError("Base solver must satisfy the a linear solver requirement.");
    }

    return struct {
        mesh: *const Mesh,
        base_solver: *const BaseSolver,

        const Self = @This();
        const Mesh = meshes.Mesh(N);
        const DofUtils = dofs.DofUtils(N, O);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);
        const CellSpace = basis.CellSpace(N, O, O);
        const StencilSpace = basis.StencilSpace(N, O, O);
        const SystemChunk = chunks.SystemChunk;

        pub fn init(mesh: *const Mesh, base_solver: *BaseSolver) Self {
            return .{
                .mesh = mesh,
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

            const base_total = self.mesh.base.cell_total;

            const field_name: []const u8 = comptime system.systemFieldNames(T.System)[0];

            const x_field: []f64 = @field(x, field_name)[0..base_total];
            const b_field: []const f64 = @field(b, field_name)[0..base_total];

            const stencil_space = DofUtils.baseStencilSpace(self.mesh);

            // Fill ctx scratch

            DofUtils.fillBaseBoundary(self.mesh, DofUtils.operContextBoundary(oper), ctx_chunk, ctx);

            const ctx_scratch = ctx_chunk.sliceConst(stencil_space.cellSpace().total());

            // Solve system using the base solver
            const base_linear_map: BaseLinearMap(T) = .{
                .self = self,
                .oper = oper,
                .ctx = ctx,
                .ctx_scratch = ctx_scratch,
                .x_scratch = x_chunk,
            };

            self.base_solver.solve(base_linear_map, x_field, b_field);
        }

        fn BaseLinearMap(comptime T: type) type {
            const field_name: []const u8 = comptime system.systemFieldNames(T.System)[0];

            return struct {
                self: *const Self,
                oper: T,
                ctx: system.SystemSliceConst(T.Context),
                ctx_scratch: system.SystemSliceConst(T.Context),
                x_scratch: SystemChunk(T.System),

                pub fn apply(wrapper: *const @This(), output: []f64, input: []const f64) void {
                    // Aliases
                    const stencil_space = DofUtils.baseStencilSpace(wrapper.self.mesh);

                    // Fill base boundary and inner
                    {
                        var sys: system.SystemSliceConst(T.System) = undefined;
                        @field(sys, field_name) = input;

                        DofUtils.fillBaseBoundary(wrapper.self.mesh, DofUtils.operSystemBoundary(wrapper.oper), wrapper.x_scratch, sys);
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
                                .ctx = wrapper.ctx_scratch,
                                .sys = wrapper.x_scratch.sliceConst(stencil_space.cellSpace().total()),
                            };

                            output[linear] = @field(wrapper.oper.apply(engine), field_name);
                        }
                    }
                }

                var iterations: usize = 0;

                pub fn callback(solver: *const @This(), iteration: usize, residual: f64, x: []const f64) void {
                    std.debug.print("Iteration: {}, Residual: {}\n", .{ iteration, residual });

                    const file_name = std.fmt.allocPrint(solver.self.mesh.gpa, "output/elliptic_iteration{}.vtu", .{iterations}) catch {
                        unreachable;
                    };

                    const file = std.fs.cwd().createFile(file_name, .{}) catch {
                        unreachable;
                    };
                    defer file.close();

                    DofUtils.writeVtk(solver.self.mesh.gpa, solver.self.mesh, .{ .metric = x }, file.writer()) catch {
                        unreachable;
                    };

                    iterations += 1;
                }
            };
        }
    };
}
