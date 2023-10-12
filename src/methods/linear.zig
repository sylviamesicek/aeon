const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const basis = @import("../basis/basis.zig");
const dofs = @import("../dofs/dofs.zig");
const lac = @import("../lac/lac.zig");
const system = @import("../system.zig");
const meshes = @import("../mesh/mesh.zig");
const geometry = @import("../geometry/geometry.zig");

/// An elliptic method which simply wraps an underlying linear solver.
pub fn LinearMapMethod(comptime N: usize, comptime O: usize, comptime InnerSolver: type) type {
    if (comptime !lac.isLinearSolver(InnerSolver)) {
        @compileError("Base solver must satisfy the a linear solver requirement.");
    }

    return struct {
        inner: InnerSolver,

        const Self = @This();

        pub fn new(inner: InnerSolver) Self {
            return .{
                .inner = inner,
            };
        }

        const Mesh = meshes.Mesh(N);
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const BoundaryUtils = dofs.BoundaryUtils(N, O);
        const IndexSpace = geometry.IndexSpace(N);
        const StencilSpace = basis.StencilSpace(N, O);
        const SystemSlice = system.SystemSlice;
        const SystemSliceConst = system.SystemSliceConst;

        pub fn solve(
            self: Self,
            allocator: Allocator,
            mesh: *const Mesh,
            block_map: []const usize,
            dof_map: DofMap,
            oper: anytype,
            x: SystemSlice(@TypeOf(oper).System),
            b: SystemSliceConst(@TypeOf(oper).System),
            context: SystemSliceConst(@TypeOf(oper).Context),
        ) !void {
            const T = @TypeOf(oper);

            if (comptime !(dofs.isSystemOperator(N, O)(T))) {
                @compileError("oper must satisfy isMeshOperator traits.");
            }

            if (comptime std.enums.values(T.System).len != 1) {
                @compileError("The linear map method only supports systems with 1 field currently.");
            }

            const sys_field: T.System = comptime std.enums.values(T.System)[0];

            // Allocate and fill ctx

            const ctx = try SystemSlice(T.Context).init(allocator, dof_map.ndofs());
            defer ctx.deinit(allocator);

            for (0..mesh.blocks.len) |block_id| {
                DofUtils.copyDofsFromCells(
                    T.Context,
                    mesh,
                    dof_map,
                    block_id,
                    ctx,
                    context,
                );
            }

            for (0..mesh.blocks.len) |block_id| {
                DofUtils.fillBoundary(
                    mesh,
                    block_map,
                    dof_map,
                    block_id,
                    DofUtils.operContextBoundary(oper),
                    ctx,
                );
            }

            // Allocate sys vector

            const sys = try SystemSlice(T.System).init(allocator, dof_map.ndofs());
            defer sys.deinit(allocator);

            const map = LinearMap(T){
                .mesh = mesh,
                .block_map = block_map,
                .dof_map = dof_map,
                .oper = oper,
                .sys = sys,
                .ctx = ctx,
            };

            // Solve using inner

            try self.inner.solve(allocator, map, x.field(sys_field), b.field(sys_field));
        }

        fn LinearMap(comptime T: type) type {
            const field_name: []const u8 = comptime std.meta.fieldNames(T.System)[0];
            const sys_field: T.System = comptime std.enums.values(T.System)[0];

            return struct {
                mesh: *const Mesh,
                block_map: []const usize,
                dof_map: DofMap,
                oper: T,
                ctx: SystemSlice(T.Context),
                sys: SystemSlice(T.System),

                pub fn apply(self: *const @This(), output: []f64, input: []const f64) void {
                    // Build systems slices which mirror input and output.
                    var input_sys: SystemSliceConst(T.System) = undefined;
                    input_sys.len = input.len;
                    @field(input_sys.ptrs, field_name) = input.ptr;

                    var output_sys: SystemSlice(T.System) = undefined;
                    output_sys.len = output.len;
                    @field(output_sys.ptrs, field_name) = output.ptr;

                    for (0..self.mesh.blocks.len) |block_id| {
                        DofUtils.copyDofsFromCells(
                            T.System,
                            self.mesh,
                            self.dof_map,
                            block_id,
                            self.sys,
                            input_sys,
                        );
                    }

                    for (0..self.mesh.blocks.len) |block_id| {
                        DofUtils.fillBoundary(
                            self.mesh,
                            self.block_map,
                            self.dof_map,
                            block_id,
                            DofUtils.operSystemBoundary(self.oper),
                            self.sys,
                        );
                    }

                    for (0..self.mesh.blocks.len) |reverse_block_id| {
                        const block_id = self.mesh.blocks.len - 1 - reverse_block_id;
                        DofUtils.restrict(
                            T.System,
                            self.mesh,
                            self.block_map,
                            self.dof_map,
                            block_id,
                            self.sys,
                        );
                    }

                    for (0..self.mesh.blocks.len) |block_id| {
                        const dof_offset = self.dof_map.offset(block_id);
                        const dof_total = self.dof_map.total(block_id);

                        const cell_offset = self.mesh.blocks[block_id].cell_offset;
                        const cell_total = self.mesh.blocks[block_id].cell_total;

                        const block_sys = self.sys.slice(dof_offset, dof_total);
                        const block_ctx = self.ctx.slice(dof_offset, dof_total);
                        const block_output = output_sys.slice(cell_offset, cell_total);

                        const stencil_space = DofUtils.blockStencilSpace(self.mesh, block_id);

                        var cells = stencil_space.nodeSpace().nodes();
                        var linear: usize = 0;

                        while (cells.next()) |cell| : (linear += 1) {
                            const engine = dofs.Engine(N, O, T.System, T.Context).new(
                                stencil_space,
                                cell,
                                block_sys.toConst(),
                                block_ctx.toConst(),
                            );

                            const app = self.oper.apply(engine);
                            block_output.field(sys_field)[linear] = @field(app, field_name);
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
