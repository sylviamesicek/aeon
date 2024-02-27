// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");

const common = aeon.common;
const geometry = aeon.geometry;
const lac = aeon.lac;
const mesh_ = aeon.mesh;

const PoissonEquation = struct {
    const N = 2;
    const M = 2;

    const BoundaryKind = common.BoundaryKind;

    const DataOut = aeon.DataOut(N, M);

    const Engine = common.Engine(N, M, M);
    const System = common.System(Tag);
    const SystemConst = common.SystemConst(Tag);

    const MultigridMethod = mesh_.MultigridMethod(N, M, BiCGStabSolver);
    const NodeManager = mesh_.NodeManager(N, M);
    const NodeWorker = mesh_.NodeWorker(N, M);
    const Mesh = mesh_.Mesh(N);

    const RealBox = geometry.RealBox(N);
    const FaceIndex = geometry.FaceIndex(N);
    const IndexSpace = geometry.IndexSpace(N);
    const IndexMixin = geometry.IndexMixin(N);

    const BiCGStabSolver = lac.BiCGStabSolver;

    const pi = std.math.pi;

    pub const Source = struct {
        amplitude: f64,

        pub const order: usize = M;

        pub fn eval(self: Source, engine: Engine) f64 {
            const pos = engine.position();
            const x = pos[0];
            const y = pos[1];

            return self.amplitude * 2 * pi * pi * @sin(pi * x) * @sin(pi * y);

            // const term1 = (y * @sin(pi * y)) * (x * pi * pi * @sin(pi * x) - 2 * pi * @cos(pi * x));
            // const term2 = (x * @sin(pi * x)) * (y * pi * pi * @sin(pi * y) - 2 * pi * @cos(pi * y));

            // return self.amplitude * (term1 + term2);
        }
    };

    pub const Solution = struct {
        amplitude: f64,

        pub const order: usize = M;

        pub fn eval(self: Solution, engine: Engine) f64 {
            const pos = engine.position();
            const x = pos[0];
            const y = pos[1];

            // return self.amplitude * x * y * @sin(pi * x) * @sin(pi * y);

            return self.amplitude * @sin(pi * x) * @sin(pi * y);
        }
    };

    pub const BoundarySet = struct {
        pub const card: usize = 1;

        pub fn boundaryIdFromFace(_: FaceIndex) usize {
            return 0;
        }

        pub const BoundaryType0: type = common.OddBoundary;

        pub fn boundary0(_: @This()) BoundaryType0 {
            return .{};
        }
    };

    pub const PoissonOperator = struct {
        pub const order: usize = M;

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

    fn run(allocator: Allocator) !void {
        std.debug.print("Running Poisson Elliptic Solver\n", .{});

        var mesh = try Mesh.init(allocator, .{
            .origin = [2]f64{ 0.0, 0.0 },
            .size = [2]f64{ 1.0, 1.0 },
        });
        defer mesh.deinit();

        var manager = try NodeManager.init(allocator, [1]usize{16} ** N, 8);
        defer manager.deinit();

        // Globally refine two times
        for (0..2) |r| {
            std.debug.print("Running Global Refinement {}\n", .{r});
            try mesh.refineGlobal(allocator);
        }

        // // Locally refine once
        // for (0..1) |r| {
        //     std.debug.print("Running Local Refinement {}\n", .{r});

        //     const flags = try allocator.alloc(bool, mesh.numCells());
        //     defer allocator.free(flags);

        //     @memset(flags, false);

        //     for (0..mesh.cells.len) |cell_id| {
        //         const bounds: RealBox = mesh.cells.items(.bounds)[cell_id];
        //         const center = bounds.center();

        //         const radius = @sqrt((center[0] - 0.5) * (center[0] - 0.5) + (center[1] - 0.5) * (center[1] - 0.5));

        //         if (radius < 0.33) {
        //             flags[cell_id] = true;
        //         }
        //     }

        //     mesh.smoothRefineFlags(flags);

        //     try mesh.refine(allocator, flags);
        // }

        try manager.build(allocator, &mesh);

        std.debug.print("Num Cells: {}\n", .{mesh.numCells()});
        std.debug.print("Num Nodes: {}\n", .{manager.numNodes()});
        std.debug.print("Writing Mesh To File\n", .{});

        {
            const file = try std.fs.cwd().createFile("output/poisson-mesh.txt", .{});
            defer file.close();

            var buf = std.io.bufferedWriter(file.writer());

            try DataOut.writeMesh(&mesh, &manager, buf.writer());

            try buf.flush();
        }

        // Allocate system
        const sys = try System.init(allocator, manager.numNodes());
        defer sys.deinit(allocator);

        // Create worker
        var worker = try NodeWorker.init(&mesh, &manager);
        defer worker.deinit();

        // Project Source
        worker.project(Source{ .amplitude = 1.0 }, sys.field(.source));
        // Project Solution
        worker.project(Solution{ .amplitude = 1.0 }, sys.field(.exact));
        // Set initial guess
        @memset(sys.field(.approx), 0.0);

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
            BoundarySet{},
            sys.field(.approx),
            sys.field(.source),
        );

        // Compute error
        for (0..manager.numNodes()) |i| {
            sys.field(.err)[i] = sys.field(.approx)[i] - sys.field(.exact)[i];
        }

        // Output result
        std.debug.print("Writing Solution To File\n", .{});

        const file = try std.fs.cwd().createFile("output/poisson.vtu", .{});
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
    try PoissonEquation.run(gpa.allocator());
}
