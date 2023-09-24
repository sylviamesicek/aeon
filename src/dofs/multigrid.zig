const std = @import("std");
const lac = @import("../lac/lac.zig");
const system = @import("../system.zig");
const mesh = @import("../mesh/mesh.zig");
const geometry = @import("../geometry/geometry.zig");
const index = @import("../index.zig");

const operator = @import("operator.zig");

/// A multigrid based elliptic solver.
pub fn MultigridSolver(comptime N: usize, comptime O: usize, comptime BaseSolver: type) type {
    if (comptime !lac.isLinearSolver(BaseSolver)) {
        @compileError("Base solver must satisfy the a linear solver requirement.");
    }

    return struct {
        mesh: *const Mesh,
        base_solver: *const BaseSolver,
        base_x: []f64,
        base_b: []f64,
        scratch: []f64,

        const Self = @This();
        const Mesh = mesh.Mesh(N, O);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);

        pub fn init(grid: *const Mesh, base_solver: *BaseSolver) !Self {
            const ndofs = grid.cellTotal();
            const ndofs_base = IndexSpace.fromSize(Index.scaled(grid.base.index_size, grid.config.tile_width)).total();

            const base_x: []f64 = try grid.gpa.alloc(f64, ndofs_base);
            errdefer grid.gpa.free(base_x);

            const base_b: []f64 = try grid.gpa.alloc(f64, ndofs_base);
            errdefer grid.gpa.free(base_b);

            const scratch = try grid.gpa.alloc(f64, ndofs);
            errdefer grid.gpa.free(scratch);

            return .{
                .mesh = grid,
                .base_solver = base_solver,
                .base_x = base_x,
                .base_b = base_b,
                .scratch = scratch,
            };
        }

        pub fn deinit(self: *Self) void {
            const gpa = self.mesh.gpa;

            gpa.free(self.base_x);
            gpa.free(self.base_b);
            gpa.free(self.scratch);
        }

        pub fn solve(
            self: *Self,
            oper: anytype,
            x: system.SystemSlice(@TypeOf(oper).System),
            b: system.SystemSliceConst(@TypeOf(oper).System),
            context: system.SystemSliceConst(@TypeOf(oper).Context),
        ) void {
            // Alias
            const T = @TypeOf(oper);

            if (comptime !(operator.isMeshOperator(N, O)(T))) {
                @compileError("Oper must satisfy isMeshOperator trait.");
            }

            if (comptime system.systemFieldCount(T.System) != 1) {
                @compileError("The multigrid solver only supports systems with 1 field currently.");
            }

            const field_name: []const u8 = comptime system.systemFieldNames(T)[0];
            const x_field: []f64 = @field(x, field_name);
            const b_field: []const f64 = @field(b, field_name);

            const stencil_space = self.baseStencilSpace();

            // Fill base
            {
                var cells = stencil_space.cellSpace().cells();
                var linear: usize = 0;

                while (cells.next()) |cell| : (linear += 1) {
                    self.base_x[linear] = stencil_space.value(cell, x_field);
                    self.base_b[linear] = stencil_space.value(cell, b_field);
                }
            }

            // Solve system using the base solver
            const base_linear_map: BaseLinearMap(T) = .{
                .self = self,
                .oper = oper,
                .context = context,
            };

            self.base_solver.solve(base_linear_map, self.base_x, self.base_b);

            // Copy solution to x
            {
                var cells = stencil_space.cellSpace().cells();
                var linear: usize = 0;

                while (cells.next()) |cell| : (linear += 1) {
                    stencil_space.setValue(cell, x_field, self.base_x[linear]);
                }
            }
        }

        fn BaseLinearMap(comptime T: type) type {
            const field_name: []const u8 = comptime system.systemFieldNames(T)[0];

            return struct {
                self: *const Self,
                oper: T,
                context: system.SystemSliceConst(T.Context),

                pub fn apply(wrapper: BaseLinearMap, output: []f64, input: []const f64) void {
                    // Aliases
                    const self = wrapper.self;
                    const stencil_space = self.mesh.baseStencilSpace();

                    // Fill scratch spell
                    {
                        var cells = stencil_space.cellSpace().cells();
                        var linear: usize = 0;

                        while (cells.next()) |cell| : (linear += 1) {
                            stencil_space.cellSpace().setValue(cell, self.scratch, input[linear]);
                        }
                    }

                    // TODO fill boundaries of scratch.

                    // Apply operator
                    {
                        var cells = stencil_space.cellSpace().cells();
                        var linear: usize = 0;

                        var operated: system.SystemSliceConst(T.System) = undefined;
                        @field(operated, field_name) = self.scratch;

                        while (cells.next()) |cell| : (linear += 1) {
                            stencil_space.cellSpace().setValue(cell, self.scratch, input[linear]);

                            const engine = system.EngineType(N, O, T){
                                .inner = .{
                                    .space = self.stencil_space,
                                    .cell = cell,
                                },
                                .context = self.context,
                                .operated = operated,
                            };

                            output[linear] = @field(self.oper.apply(engine), field_name);
                        }
                    }
                }
            };
        }
    };
}
