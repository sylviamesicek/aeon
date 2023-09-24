const std = @import("std");
const Allocator = std.mem.Allocator;

const lac = @import("../lac/lac.zig");
const system = @import("../system.zig");
const mesh = @import("../mesh/mesh.zig");
const geometry = @import("../geometry/geometry.zig");
const index = @import("../index.zig");

const dofs = @import("dofs.zig");
const operator = @import("operator.zig");

/// A multigrid based elliptic solver.
pub fn MultigridSolver(comptime N: usize, comptime O: usize, comptime BaseSolver: type) type {
    if (comptime !lac.isLinearSolver(BaseSolver)) {
        @compileError("Base solver must satisfy the a linear solver requirement.");
    }

    return struct {
        gpa: Allocator,
        dof_handler: *const DofHandler,
        base_size: [N]usize,
        base_solver: *const BaseSolver,
        base_x: []f64,
        base_b: []f64,
        scratch: []f64,

        const Self = @This();
        const DofHandler = dofs.DofHandler(N, O);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);

        pub fn init(allocator: Allocator, dof_handler: *const DofHandler, base_solver: *BaseSolver) !Self {
            const ndofs = dof_handler.ndofs();

            const base_size = dof_handler.mesh.base.index_size;

            const ndofs_base = IndexSpace.fromSize(Index.scaled(base_size, dof_handler.mesh.tile_width)).total();

            const base_x: []f64 = try allocator.alloc(f64, ndofs_base);
            errdefer allocator.free(base_x);

            const base_b: []f64 = try allocator.alloc(f64, ndofs_base);
            errdefer allocator.free(base_b);

            const scratch = try allocator.alloc(f64, ndofs);
            errdefer allocator.free(scratch);

            return .{
                .gpa = allocator,
                .dof_handler = dof_handler,
                .base_size = base_size,
                .base_solver = base_solver,
                .base_x = base_x,
                .base_b = base_b,
                .scratch = scratch,
            };
        }

        pub fn deinit(self: *Self) void {
            self.gpa.free(self.base_x);
            self.gpa.free(self.base_b);
            self.gpa.free(self.scratch);
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

            if (comptime !(operator.isMeshOperator(N, O)(T) and operator.isMeshBoundary(N)(T))) {
                @compileError("Oper must satisfy isMeshOperator and isMeshBoundary traits.");
            }

            if (comptime system.systemFieldCount(T.System) != 1) {
                @compileError("The multigrid solver only supports systems with 1 field currently.");
            }

            const field_name: []const u8 = comptime system.systemFieldNames(T.System)[0];
            const x_field: []f64 = @field(x, field_name);
            const b_field: []const f64 = @field(b, field_name);

            const stencil_space = self.dof_handler.mesh.baseStencilSpace();

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
                    stencil_space.cellSpace().setValue(cell, x_field, self.base_x[linear]);
                }
            }
        }

        fn BaseLinearMap(comptime T: type) type {
            const field_name: []const u8 = comptime system.systemFieldNames(T.System)[0];

            return struct {
                self: *const Self,
                oper: T,
                context: system.SystemSliceConst(T.Context),

                pub fn apply(wrapper: @This(), output: []f64, input: []const f64) void {
                    // Aliases
                    const self: *const Self = wrapper.self;
                    const stencil_space = self.dof_handler.mesh.baseStencilSpace();

                    // Fill scratch spell
                    {
                        var cells = stencil_space.cellSpace().cells();
                        var linear: usize = 0;

                        while (cells.next()) |cell| : (linear += 1) {
                            stencil_space.cellSpace().setValue(cell, self.scratch, input[linear]);
                        }
                    }

                    // Fill base boundary
                    {
                        var sys: system.SystemSlice(T.System) = undefined;
                        @field(sys, field_name) = self.scratch;

                        self.dof_handler.fillBaseBoundary(wrapper.oper, sys);
                    }

                    // Apply operator
                    {
                        var cells = stencil_space.cellSpace().cells();
                        var linear: usize = 0;

                        var sys: system.SystemSliceConst(T.System) = undefined;
                        @field(sys, field_name) = self.scratch;

                        while (cells.next()) |cell| : (linear += 1) {
                            const engine = operator.EngineType(N, O, T){
                                .inner = .{
                                    .space = stencil_space,
                                    .cell = cell,
                                },
                                .context = wrapper.context,
                                .operated = sys,
                            };

                            output[linear] = @field(wrapper.oper.apply(engine), field_name);
                        }
                    }
                }
            };
        }
    };
}
