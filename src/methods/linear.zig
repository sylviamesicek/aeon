const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const basis = @import("../basis/basis.zig");
const bsamr = @import("../bsamr/bsamr.zig");
const geometry = @import("../geometry/geometry.zig");
const lac = @import("../lac/lac.zig");
const mesh = @import("../mesh/mesh.zig");
const nodes = @import("../nodes/nodes.zig");

/// An elliptic method which simply wraps an underlying linear solver.
pub fn LinearMapMethod(comptime N: usize, comptime M: usize, comptime InnerSolver: type) type {
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

        const Mesh = bsamr.Mesh(N);
        const DofManager = bsamr.DofManager(N, M);
        const IndexSpace = geometry.IndexSpace(N);

        pub fn solve(
            self: Self,
            allocator: Allocator,
            grid: *const Mesh,
            dofs: DofManager,
            operator: anytype,
            boundary: anytype,
            x: []f64,
            rhs: []const f64,
        ) !void {
            const Op = @TypeOf(operator);
            const Bound = @TypeOf(boundary);

            if (comptime !(mesh.isOperator(N, M)(Op))) {
                @compileError("operator must satisfy isOperator trait.");
            }

            if (comptime !(nodes.isBoundary(N)(Bound))) {
                @compileError("boundary must satisfy isBoundary trait.");
            }

            // Allocate and fill x nodes

            const nodes_ = try allocator.alloc(f64, dofs.numNodes());
            defer allocator.free(nodes_);

            // Allocate sys vector

            const map = LinearMap(Op, Bound){
                .grid = grid,
                .dofs = dofs,
                .operator = operator,
                .boundary = boundary,
                .nodes_ = nodes_,
            };

            // Solve using inner
            try self.inner.solve(allocator, map, x, rhs);
        }

        fn LinearMap(comptime Op: type, comptime Bound: type) type {
            return struct {
                grid: *const Mesh,
                dofs: DofManager,
                operator: Op,
                boundary: Bound,
                nodes_: []f64,

                pub fn apply(self: *const @This(), out: []f64, in: []const f64) void {
                    // Cells -> Nodes
                    self.dofs.transfer(self.grid, self.boundary, self.nodes_, in);

                    for (0..self.mesh.blocks.len) |block_id| {
                        self.dofs.applyCells(self.grid, block_id, self.operator, out, self.nodes_);
                    }

                    for (0..self.mesh.blocks.len) |rev_block_id| {
                        const block_id = self.mesh.blocks.len - 1 - rev_block_id;

                        self.dofs.restrictCells(self.grid, block_id, out);
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
