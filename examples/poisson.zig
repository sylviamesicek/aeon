// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");

const common = aeon.common;
const geometry = aeon.geometry;
const tree = aeon.tree;

const PoissonEquation = struct {
    const N = 2;
    const M = 2;

    const BoundaryKind = common.BoundaryKind;
    const Robin = common.Robin;

    const DataOut = aeon.DataOut(N);

    const SystemConst = common.SystemConst;

    const MultigridMethod = tree.MultigridMethod(N, M, M, BiCGStabSolver);
    const NodeManager = tree.NodeManager(N);
    const NodeWorker = tree.NodeWorker(N, M);
    const TreeMesh = tree.TreeMesh(N);

    const RealBox = geometry.RealBox(N);
    const FaceIndex = geometry.FaceIndex(N);
    const IndexSpace = geometry.IndexSpace(N);
    const IndexMixin = geometry.IndexMixin(N);

    const BiCGStabSolver = aeon.lac.BiCGStabSolver;

    const Engine = common.Engine(N, M, M);

    const pi = std.math.pi;

    pub const Source = struct {
        amplitude: f64,

        pub fn project(self: Source, engine: Engine) f64 {
            const pos = engine.position();
            const x = pos[0];
            const y = pos[1];

            // return self.amplitude * 2 * pi * pi * @sin(pi * x) * @sin(pi * y);

            const term1 = (y * @sin(pi * y)) * (x * pi * pi * @sin(pi * x) - 2 * pi * @cos(pi * x));
            const term2 = (x * @sin(pi * x)) * (y * pi * pi * @sin(pi * y) - 2 * pi * @cos(pi * y));

            return self.amplitude * (term1 + term2);
        }
    };

    pub const Solution = struct {
        amplitude: f64,

        pub fn project(self: Solution, engine: Engine) f64 {
            const pos = engine.position();
            const x = pos[0];
            const y = pos[1];

            return self.amplitude * x * y * @sin(pi * x) * @sin(pi * y);

            // return self.amplitude * @sin(pi * x) * @sin(pi * y);
        }
    };

    pub const Boundary = struct {
        pub fn kind(face: FaceIndex) BoundaryKind {
            _ = face;
            return .robin;
        }

        pub fn robin(self: Boundary, pos: [N]f64, face: FaceIndex) Robin {
            _ = face;
            _ = pos;
            _ = self;
            return Robin.diritchlet(0.0);
        }
    };

    pub const PoissonOperator = struct {
        pub fn apply(
            _: PoissonOperator,
            engine: Engine,
            field: []const f64,
        ) f64 {
            return -engine.laplacian(field);
        }

        pub fn applyDiag(
            _: PoissonOperator,
            engine: Engine,
        ) f64 {
            return -engine.laplacianDiag();
        }
    };

    // Run

    fn run(allocator: Allocator) !void {
        std.debug.print("Running Poisson Elliptic Solver\n", .{});

        var mesh = try TreeMesh.init(allocator, .{
            .origin = [2]f64{ 0.0, 0.0 },
            .size = [2]f64{ 1.0, 1.0 },
        });
        defer mesh.deinit();

        // Globally refine two times
        for (0..2) |r| {
            std.debug.print("Running Global Refinement {}\n", .{r});

            @memset(mesh.cells.items(.flag), true);

            try mesh.refine(allocator);
        }

        // Locally refine once
        for (0..1) |r| {
            std.debug.print("Running Refinement {}\n", .{r});

            @memset(mesh.cells.items(.flag), false);

            for (0..mesh.cells.len) |cell_id| {
                const bounds: RealBox = mesh.cells.items(.bounds)[cell_id];
                const center = bounds.center();

                const radius = @sqrt((center[0] - 0.5) * (center[0] - 0.5) + (center[1] - 0.5) * (center[1] - 0.5));

                if (radius < 0.33) {
                    mesh.cells.items(.flag)[cell_id] = true;
                }
            }

            try mesh.refine(allocator);
        }

        var manager = try NodeManager.init(allocator, [1]usize{16} ** N, 8);
        defer manager.deinit();

        try manager.build(allocator, &mesh);

        std.debug.print("Num Cells: {}\n", .{mesh.numCells()});
        std.debug.print("Num Packed Nodes: {}\n", .{manager.numPackedNodes()});
        std.debug.print("Writing Mesh To File\n", .{});

        {
            const file = try std.fs.cwd().createFile("output/poisson-mesh.txt", .{});
            defer file.close();

            var buf = std.io.bufferedWriter(file.writer());

            try DataOut.writeMesh(&mesh, &manager, buf.writer());

            try buf.flush();
        }

        // Create worker
        var worker = try NodeWorker.init(allocator, &mesh, &manager);
        defer worker.deinit();

        // Project source values
        const source = try allocator.alloc(f64, worker.numNodes());
        defer allocator.free(source);

        worker.order(M).projectAll(Source{ .amplitude = 1.0 }, source);

        // Project solution
        const solution = try allocator.alloc(f64, worker.numNodes());
        defer allocator.free(solution);

        worker.order(M).projectAll(Solution{ .amplitude = 1.0 }, solution);
        worker.order(M).fillGhostNodesAll(Boundary{}, solution);

        // Allocate numerical cell vector

        const numerical = try allocator.alloc(f64, worker.numNodes());
        defer allocator.free(numerical);

        @memset(numerical, 0.0);

        const solver: MultigridMethod = .{
            .base_solver = BiCGStabSolver.new(10000, 10e-12),
            .max_iters = 20,
            .tolerance = 10e-10,
            .presmooth = 5,
            .postsmooth = 5,
        };

        try solver.solve(
            allocator,
            &worker,
            PoissonOperator{},
            Boundary{},
            numerical,
            source,
        );

        // Compute error

        const err = try allocator.alloc(f64, worker.numNodes());
        defer allocator.free(err);

        for (0..worker.numNodes()) |i| {
            err[i] = numerical[i] - solution[i];
        }

        const apply = try allocator.alloc(f64, worker.numNodes());
        defer allocator.free(apply);

        for (0..mesh.numLevels()) |level_id| {
            worker.order(M).apply(level_id, apply, PoissonOperator{}, solution);
        }

        // Output result

        std.debug.print("Writing Solution To File\n", .{});

        const file = try std.fs.cwd().createFile("output/poisson.vtu", .{});
        defer file.close();

        const Output = enum {
            numerical,
            exact,
            err,
            apply,
            source,
        };

        const output = SystemConst(Output).view(worker.numNodes(), .{
            .numerical = numerical,
            .exact = solution,
            .source = source,
            .apply = apply,
            .err = err,
        });

        var buf = std.io.bufferedWriter(file.writer());

        try DataOut.Ghost(M).writeVtk(
            Output,
            allocator,
            &worker,
            output,
            buf.writer(),
        );

        try buf.flush();
    }
};

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
    try PoissonEquation.run(gpa.allocator());
}
