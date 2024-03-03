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

            // if (is_symmetric) {
            //     return self.amplitude * 2 * pi * pi * @sin(pi * x) * @sin(pi * y);
            // } else {

            // }

            const term1 = -2 * pi * (@sin(pi * x) * y * @cos(pi * y) + @sin(pi * y) * x * @cos(pi * x));
            const term2 = (-1 - 2 * pi * pi) * x * y * @cos(pi * x) * @cos(pi * y);
            return self.amplitude * (term1 + term2);
        }
    };

    pub const Solution = struct {
        amplitude: f64,

        pub const order: usize = M;

        pub fn eval(self: Solution, engine: Engine) f64 {
            const pos = engine.position();
            const x = pos[0];
            const y = pos[1];

            // if (is_symmetric) {
            //     return self.amplitude * @sin(pi * x) * @sin(pi * y);
            // } else {
            //
            // }

            return self.amplitude * x * y * @cos(pi * x) * @cos(pi * y);
        }
    };

    pub const XDerivative = struct {
        amplitude: f64,

        pub fn eval(self: @This(), pos: [N]f64) f64 {
            const x = pos[0];
            const y = pos[1];

            return self.amplitude * y * @cos(pi * y) * (@cos(pi * x) - pi * x * @sin(pi * x));
        }
    };

    pub const YDerivative = struct {
        amplitude: f64,

        pub fn eval(self: @This(), pos: [N]f64) f64 {
            const x = pos[0];
            const y = pos[1];

            return self.amplitude * x * @cos(pi * x) * (@cos(pi * y) - pi * y * @sin(pi * y));
        }
    };

    pub const XBoundary = struct {
        robin_rhs: XDerivative,
        comptime robin_value: common.ZeroField(N) = .{},

        pub const kind: BoundaryKind = .robin;
        pub const priority: usize = 0;

        pub fn new(amplitude: f64) @This() {
            return .{ .robin_rhs = .{ .amplitude = amplitude } };
        }
    };

    pub const YBoundary = struct {
        robin_rhs: YDerivative,
        comptime robin_value: common.ZeroField(N) = .{},

        pub const kind: BoundaryKind = .robin;
        pub const priority: usize = 0;

        pub fn new(amplitude: f64) @This() {
            return .{ .robin_rhs = .{ .amplitude = amplitude } };
        }
    };

    pub const BoundarySet = struct {
        amplitude: f64,

        pub const card: usize = 3;

        pub fn boundaryIdFromFace(face: FaceIndex) usize {
            if (face.side == true and face.axis == 1) {
                return 2;
            } else if (face.side == true and face.axis == 0) {
                return 1;
            } else {
                return 0;
            }
        }

        pub const BoundaryType0: type = common.OddBoundary;
        pub const BoundaryType1: type = XBoundary;
        pub const BoundaryType2: type = YBoundary;

        pub fn boundary0(_: @This()) BoundaryType0 {
            return .{};
        }

        pub fn boundary1(self: @This()) BoundaryType1 {
            return XBoundary.new(self.amplitude);
        }

        pub fn boundary2(self: @This()) BoundaryType2 {
            return YBoundary.new(self.amplitude);
        }
    };

    pub const HemholtzOperator = struct {
        pub const order: usize = M;

        pub fn apply(
            _: HemholtzOperator,
            engine: Engine,
            field: []const f64,
        ) f64 {
            return engine.laplacian(field) - engine.value(field);
        }

        pub fn applyDiag(
            _: HemholtzOperator,
            engine: Engine,
        ) f64 {
            return engine.laplacianDiag() - engine.valueDiag();
        }
    };

    pub const Tag = enum {
        approx,
        exact,
        err,
        residual,
        source,
    };

    // Run

    fn run(allocator: Allocator) !void {
        std.debug.print("Running Poisson Elliptic Solver\n", .{});

        const source = Source{ .amplitude = 1.0 };
        const solution = Solution{ .amplitude = 1.0 };
        const set = BoundarySet{ .amplitude = 1.0 };
        const op = HemholtzOperator{};

        var mesh = try Mesh.init(allocator, .{
            .origin = [2]f64{ 0.0, 0.0 },
            .size = [2]f64{ 0.5, 0.5 },
        });
        defer mesh.deinit();

        // Globally refine two times
        for (0..0) |r| {
            std.debug.print("Running Global Refinement {}\n", .{r});
            try mesh.refineGlobal(allocator);
        }

        var manager = try NodeManager.init(allocator, &mesh, .{ 16, 16 }, 8);
        defer manager.deinit();

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

        std.debug.print("Num Cells: {}\n", .{manager.numCells()});
        std.debug.print("Num Nodes: {}\n", .{manager.numNodes()});
        std.debug.print("Writing Mesh To File\n", .{});

        {
            const file = try std.fs.cwd().createFile("output/poisson-mesh.txt", .{});
            defer file.close();

            var buf = std.io.bufferedWriter(file.writer());

            try DataOut.writeMesh(&manager, buf.writer());

            try buf.flush();
        }

        // Allocate system
        const sys = try System.init(allocator, manager.numNodes());
        defer sys.deinit(allocator);

        // Project Source
        manager.project(source, sys.field(.source));
        // Project Solution
        manager.project(solution, sys.field(.exact));
        // Fill Boundary of exact
        manager.fillGhostNodes(M, set, sys.field(.exact));
        // Set initial guess
        @memset(sys.field(.approx), 0.0);

        // Solve

        var method = try MultigridMethod.init(
            allocator,
            manager.numNodes(),
            BiCGStabSolver.new(1000, 10e-15),
            .{
                .max_iters = 1,
                .tolerance = 10e-11,
                .presmooth = 5,
                .postsmooth = 5,
            },
        );
        defer method.deinit();

        try method.solve(
            &manager,
            op,
            set,
            sys.field(.approx),
            sys.field(.source),
        );

        // Compute error
        for (0..manager.numNodes()) |i| {
            sys.field(.err)[i] = sys.field(.approx)[i] - sys.field(.exact)[i];
        }

        manager.residual(sys.field(.residual), sys.field(.source), op, sys.field(.exact));

        {
            // Output result
            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/poisson.vtu", .{});
            defer file.close();

            var buf = std.io.bufferedWriter(file.writer());

            try DataOut.writeVtk(
                Tag,
                allocator,
                &manager,
                sys.toConst(),
                .{ .ghost = true },
                buf.writer(),
            );

            try buf.flush();
        }

        {
            // Output result
            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/poisson-cover.vtu", .{});
            defer file.close();

            var buf = std.io.bufferedWriter(file.writer());

            try DataOut.writeVtk(
                Tag,
                allocator,
                &manager,
                sys.toConst(),
                .{
                    .levels = mesh.numLevels() - 1,
                    .ghost = true,
                },
                buf.writer(),
            );

            try buf.flush();
        }
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
