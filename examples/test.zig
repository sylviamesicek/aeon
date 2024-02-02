// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");

const common = aeon.common;
const geometry = aeon.geometry;
const lac = aeon.lac;
const tree = aeon.tree;

const N = 2;
const M = 2;

const AdaptivePoissonEquation = struct {
    const BoundaryKind = common.BoundaryKind;
    const Robin = common.Robin;

    const DataOut = aeon.DataOut(N, M);

    const Engine = common.Engine(N, M, M);
    const System = common.System(Tag);
    const SystemConst = common.SystemConst(Tag);

    const MultigridMethod = tree.MultigridMethod(N, M, M, BiCGStabSolver);
    const NodeManager = tree.NodeManager(N, M);
    const NodeWorker = tree.NodeWorker(N, M);
    const TreeMesh = tree.TreeMesh(N);

    const RealBox = geometry.RealBox(N);
    const FaceIndex = geometry.FaceIndex(N);
    const IndexSpace = geometry.IndexSpace(N);
    const IndexMixin = geometry.IndexMixin(N);

    const BiCGStabSolver = lac.BiCGStabSolver;

    const pi = std.math.pi;

    pub const Source = struct {
        amplitude: f64,

        pub fn project(self: Source, engine: Engine) f64 {
            const pos = engine.position();
            const x = 1.0 - pos[0];
            const y = 1.0 - pos[1];

            const term1 = (y * @sin(pi * y)) * (x * pi * pi * @sin(pi * x) - 2 * pi * @cos(pi * x));
            const term2 = (x * @sin(pi * x)) * (y * pi * pi * @sin(pi * y) - 2 * pi * @cos(pi * y));

            return self.amplitude * (term1 + term2);
        }
    };

    pub const Solution = struct {
        amplitude: f64,

        pub fn project(self: Solution, engine: Engine) f64 {
            const pos = engine.position();
            const x = 1.0 - pos[0];
            const y = 1.0 - pos[1];

            return self.amplitude * x * y * @sin(pi * x) * @sin(pi * y);
        }
    };

    pub const Boundary = struct {
        pub fn kind(_: FaceIndex) BoundaryKind {
            return .robin;
        }

        pub fn robin(_: Boundary, _: [N]f64, _: FaceIndex) Robin {
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

    pub const Tag = enum {
        approx,
        exact,
        err,
        source,
    };

    // Run

    fn run(allocator: Allocator, max_adaptive: usize, tolerance: f64) !void {
        _ = max_adaptive; // autofix
        _ = tolerance; // autofix

        std.debug.print("Running Adaptive Poisson Solver\n", .{});

        var mesh = try TreeMesh.init(allocator, .{
            .origin = [2]f64{ 0.0, 0.0 },
            .size = [2]f64{ 1.0, 1.0 },
        });
        defer mesh.deinit();

        var manager = try NodeManager.init(allocator, [1]usize{256} ** N, 8);
        defer manager.deinit();

        try mesh.refineGlobal(allocator);

        const flags = try allocator.alloc(bool, mesh.numCells());
        defer allocator.free(flags);

        flags[1] = true;

        try mesh.refine(allocator, flags);

        // Build node
        try manager.build(allocator, &mesh);

        std.debug.print("Num Cells: {}\n", .{mesh.numCells()});
        std.debug.print("Num Nodes: {}\n", .{manager.numNodes()});

        var worker = try NodeWorker.init(&mesh, &manager);
        defer worker.deinit();

        // Allocate system
        const sys = try System.init(allocator, manager.numNodes());
        defer sys.deinit(allocator);

        // Project Source
        worker.order(M).project(Source{ .amplitude = 1.0 }, sys.field(.source));
        // Project Solution
        worker.order(M).project(Solution{ .amplitude = 1.0 }, sys.field(.exact));
        // Set initial guess
        @memset(sys.field(.approx), 0.0);

        // *****************************
        // Solve

        var method = try MultigridMethod.init(
            allocator,
            manager.numNodes(),
            BiCGStabSolver.new(10000, 10e-12),
            .{
                .max_iters = 20,
                .tolerance = 10e-10,
                .presmooth = 5,
                .postsmooth = 5,
            },
        );
        defer method.deinit();

        try method.solve(
            &worker,
            PoissonOperator{},
            Boundary{},
            sys.field(.approx),
            sys.field(.source),
        );

        // *****************************
        // Compute error

        for (0..manager.numNodes()) |i| {
            sys.field(.err)[i] = sys.field(.approx)[i] - sys.field(.exact)[i];
        }

        const norm = worker.normScaled(sys.field(.err));

        std.debug.print("Global Error {}\n", .{norm});

        {
            const file = try std.fs.cwd().createFile("output/test-mesh.txt", .{});
            defer file.close();

            var buf = std.io.bufferedWriter(file.writer());

            try DataOut.writeMesh(&mesh, &manager, buf.writer());

            try buf.flush();
        }

        const file_name = try std.fmt.allocPrint(allocator, "output/test.vtu", .{});
        defer allocator.free(file_name);

        // Output result
        std.debug.print("Writing Solution To File\n", .{});

        const file = try std.fs.cwd().createFile(file_name, .{});
        defer file.close();

        var buf = std.io.bufferedWriter(file.writer());

        try DataOut.writeVtk(
            Tag,
            allocator,
            &worker,
            sys.toConst(),
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
    try AdaptivePoissonEquation.run(gpa.allocator(), 6, 10e-12);
}
