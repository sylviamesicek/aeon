//! Uses old inexact equations and boundary conditions.

// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const aeon = @import("aeon");

const common = aeon.common;
const geometry = aeon.geometry;
const lac = aeon.lac;
const tree = aeon.tree;

pub const N = 2;
pub const M = 3;
pub const O = 2;

const cfl: f64 = 0.1;

pub const BrillEvolution = struct {
    const DataOut = aeon.DataOut(N, M);

    const BoundaryKind = common.BoundaryKind;
    const Engine = common.Engine(N, M, O);
    const Robin = common.Robin;
    const Rk4Integrator = common.RungeKutta4Integrator;
    const System = common.System;
    const SystemConst = common.SystemConst;

    const FaceIndex = geometry.FaceIndex(N);
    const IndexSpace = geometry.IndexSpace(N);
    const IndexMixin = geometry.IndexMixin(N);
    const RealBox = geometry.RealBox(N);

    const BiCGStabSolver = lac.BiCGStabSolver;

    const TreeMesh = tree.TreeMesh(N);
    const MultigridMethod = tree.MultigridMethod(N, M, O, BiCGStabSolver);
    const NodeManager = tree.NodeManager(N, M);
    const NodeWorker = tree.NodeWorker(N, M);

    pub const EvenBoundary = struct {
        pub fn kind(face: FaceIndex) BoundaryKind {
            if (face.side == false and face.axis == 0) {
                return .even;
            } else {
                return .robin;
            }
        }

        pub fn robin(_: EvenBoundary, pos: [N]f64, face: FaceIndex) Robin {
            const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);

            return .{
                .value = @abs(pos[face.axis]) / (r * r),
                .flux = 1.0,
                .rhs = 0.0,
            };
        }
    };

    pub const OddBoundary = struct {
        pub fn kind(face: FaceIndex) BoundaryKind {
            if (face.side == false and face.axis == 0) {
                return .odd;
            } else {
                return .robin;
            }
        }

        pub fn robin(_: OddBoundary, pos: [N]f64, face: FaceIndex) Robin {
            const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);

            return .{
                .value = @abs(pos[face.axis]) / (r * r),
                .flux = 1.0,
                .rhs = 0.0,
            };
        }
    };

    const lapse_boundary = EvenBoundary{};
    const shiftr_boundary = OddBoundary{};
    const shiftz_boundary = EvenBoundary{};
    const psi_boundary = EvenBoundary{};
    const eta_boundary = EvenBoundary{};
    const seed_boundary = OddBoundary{};
    const y_boundary = OddBoundary{};
    const u_boundary = EvenBoundary{};
    const w_boundary = OddBoundary{};
    const constraint_boundary = EvenBoundary{};

    pub const Seed = struct {
        amplitude: f64,
        sigma: f64,

        pub fn project(self: Seed, engine: Engine) f64 {
            const pos = engine.position();

            const rho = pos[0];
            const z = pos[1];

            const rho2 = rho * rho;
            const z2 = z * z;
            const sigma2 = self.sigma * self.sigma;

            return self.amplitude * rho * @exp(-(rho2 + z2) / sigma2);
        }
    };

    pub const InitialEtaRhs = struct {
        seed: []const f64,

        pub fn project(self: InitialEtaRhs, engine: Engine) f64 {
            const pos = engine.position();
            const r = pos[0];

            const s_hessian: [N][N]f64 = engine.hessian(self.seed);
            const s_grad: [N]f64 = engine.gradient(self.seed);

            const s_rr = s_hessian[0][0];
            const s_zz = s_hessian[1][1];
            const s_r = s_grad[0];

            return -1.0 / 4.0 * (r * s_rr + 2.0 * s_r + r * s_zz);
        }
    };

    pub const InitialEtaOp = struct {
        seed: []const f64,

        pub fn apply(self: InitialEtaOp, engine: Engine, eta: []const f64) f64 {
            const position: [N]f64 = engine.position();
            const r = position[0];

            const hessian: [N][N]f64 = engine.hessian(eta);
            const grad: [N]f64 = engine.gradient(eta);
            const val: f64 = engine.value(eta);

            const shessian: [N][N]f64 = engine.hessian(self.seed);
            const sgrad: [N]f64 = engine.gradient(self.seed);

            const term1 = hessian[0][0] + grad[0] / r + hessian[1][1];
            const term2 = val / 4.0 * (r * shessian[0][0] + 2.0 * sgrad[0] + r * shessian[1][1]);

            return term1 + term2;
        }

        pub fn applyDiag(self: InitialEtaOp, engine: Engine) f64 {
            const position: [N]f64 = engine.position();
            const r = position[0];

            const hessian: [N][N]f64 = engine.hessianDiag();
            const grad: [N]f64 = engine.gradientDiag();
            const val: f64 = engine.valueDiag();

            const shessian: [N][N]f64 = engine.hessian(self.seed);
            const sgrad: [N]f64 = engine.gradient(self.seed);

            const term1 = hessian[0][0] + grad[0] / r + hessian[1][1];
            const term2 = val / 4.0 * (r * shessian[0][0] + 2.0 * sgrad[0] + r * shessian[1][1]);

            return term1 + term2;
        }
    };

    pub const LapseRhs = struct {
        psi: []const f64,
        seed: []const f64,
        u: []const f64,
        y: []const f64,
        w: []const f64,

        pub fn project(self: LapseRhs, engine: Engine) f64 {
            const pos: [N]f64 = engine.position();
            const rho = pos[0];

            const psi = engine.value(self.psi);
            const seed = engine.value(self.seed);
            const u = engine.value(self.u);
            const y = engine.value(self.y);
            const w = engine.value(self.w);

            const scale = @exp(2.0 * rho * seed + 2.0 * psi);

            const term1 = 2.0 / 3.0 * (rho * rho * y * y + rho * u * y + u * u);
            const term2 = 2.0 * w * w;

            return scale * (term1 + term2);
        }
    };

    pub const LapseOp = struct {
        psi: []const f64,
        seed: []const f64,
        u: []const f64,
        y: []const f64,
        w: []const f64,

        pub fn apply(self: LapseOp, engine: Engine, lapse: []const f64) f64 {
            const pos = engine.position();
            const rho = pos[0];

            const hessian: [N][N]f64 = engine.hessian(lapse);
            const grad: [N]f64 = engine.gradient(lapse);
            const val: f64 = engine.value(lapse);

            const psi = engine.value(self.psi);
            const pgrad = engine.gradient(self.psi);

            const seed = engine.value(self.seed);
            const u = engine.value(self.u);
            const y = engine.value(self.y);
            const w = engine.value(self.w);

            const term1 = hessian[0][0] + grad[0] / rho + hessian[1][1];
            const term2 = (pgrad[0] * grad[0] + pgrad[1] * grad[1]);

            const scale = -val * @exp(2.0 * rho * seed + 2.0 * psi);

            const term3 = 2.0 / 3.0 * (rho * rho * y * y + rho * u * y + u * u);
            const term4 = 2 * w * w;

            return term1 + term2 + scale * (term3 + term4);
        }

        pub fn applyDiag(self: LapseOp, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];

            const hessian: [N][N]f64 = engine.hessianDiag();
            const grad: [N]f64 = engine.gradientDiag();
            const val: f64 = engine.valueDiag();

            const psi = engine.value(self.psi);
            const pgrad = engine.gradient(self.psi);

            const seed = engine.value(self.seed);
            const u = engine.value(self.u);
            const y = engine.value(self.y);
            const w = engine.value(self.w);

            const term1 = hessian[0][0] + grad[0] / rho + hessian[1][1];
            const term2 = (pgrad[0] * grad[0] + pgrad[1] * grad[1]);

            const scale = -val * @exp(2.0 * rho * seed + 2.0 * psi);

            const term3 = 2.0 / 3.0 * (rho * rho * y * y + rho * u * y + u * u);
            const term4 = 2.0 * w * w;

            return term1 + term2 + scale * (term3 + term4);
        }
    };

    pub const ShiftRRhs = struct {
        lapse: []const f64,
        w: []const f64,
        u: []const f64,

        pub fn project(self: ShiftRRhs, engine: Engine) f64 {
            const lapse = engine.value(self.lapse);
            const lgrad = engine.gradient(self.lapse);

            const w = engine.value(self.w);
            const wgrad = engine.gradient(self.w);

            const u = engine.value(self.u);
            const ugrad = engine.gradient(self.u);

            const term1 = 2.0 * (w * lgrad[1] + lapse * wgrad[1]);
            const term2 = -u * lgrad[0] - lapse * ugrad[0];

            return term1 + term2;
        }
    };

    pub const ShiftZRhs = struct {
        lapse: []const f64,
        w: []const f64,
        u: []const f64,

        pub fn project(self: ShiftZRhs, engine: Engine) f64 {
            const lapse = engine.value(self.lapse);
            const lgrad = engine.gradient(self.lapse);

            const w = engine.value(self.w);
            const wgrad = engine.gradient(self.w);

            const u = engine.value(self.u);
            const ugrad = engine.gradient(self.u);

            const term1 = 2.0 * (w * lgrad[0] + lapse * wgrad[0]);
            const term2 = u * lgrad[1] + lapse * ugrad[1];

            return term1 + term2;
        }
    };

    pub const ShiftOp = struct {
        pub fn apply(_: ShiftOp, engine: Engine, shift: []const f64) f64 {
            return engine.laplacian(shift);
        }

        pub fn applyDiag(_: ShiftOp, engine: Engine) f64 {
            return engine.laplacianDiag();
        }
    };

    pub const Hamiltonian = struct {
        eta: []const f64,
        seed: []const f64,
        u: []const f64,
        y: []const f64,
        w: []const f64,

        pub fn project(self: Hamiltonian, engine: Engine) f64 {
            const pos: [N]f64 = engine.position();
            const rho = pos[0];

            const ehess: [N][N]f64 = engine.hessian(self.eta);
            const egrad: [N]f64 = engine.gradient(self.eta);
            const eta: f64 = engine.value(self.eta);

            const shess: [N][N]f64 = engine.hessian(self.seed);
            const sgrad: [N]f64 = engine.gradient(self.seed);
            const seed: f64 = engine.value(self.seed);

            const u = engine.value(self.u);
            const y = engine.value(self.y);
            const w = engine.value(self.w);

            const term1 = ehess[0][0] + egrad[0] / rho + ehess[1][1];
            const term2 = eta / 4.0 * (rho * shess[0][0] + 2.0 * sgrad[0] + rho * shess[1][1]);

            const scale = eta * eta * eta * eta * eta * @exp(2.0 * rho * seed) / 4.0;
            const term3 = 1.0 / 3.0 * (rho * rho * y * y + rho * u * y + u * u);
            const term4 = w * w;

            return term1 + term2 + scale * (term3 + term4);
        }
    };

    pub const PsiEvolution = struct {
        lapse: []const f64,
        shiftr: []const f64,
        shiftz: []const f64,
        psi: []const f64,
        u: []const f64,
        y: []const f64,

        pub fn project(self: PsiEvolution, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];

            const lapse = engine.value(self.lapse);
            const shiftr = engine.value(self.shiftr);
            const shiftz = engine.value(self.shiftz);

            const pgrad = engine.gradient(self.psi);
            const u = engine.value(self.u);
            const y = engine.value(self.y);

            const term1 = shiftr * pgrad[0] + shiftz * pgrad[1] + shiftr / rho;
            const term2 = lapse / 3.0 * (2.0 * rho * y + u);

            return term1 + term2;
        }
    };

    pub const SeedEvolution = struct {
        lapse: []const f64,
        shiftr: []const f64,
        shiftz: []const f64,
        seed: []const f64,
        y: []const f64,

        pub fn project(self: SeedEvolution, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];
            const rho2 = rho * rho;

            const seed = engine.value(self.seed);
            const sgrad = engine.gradient(self.seed);

            const lapse = engine.value(self.lapse);
            const shiftr = engine.value(self.shiftr);
            const shiftz = engine.value(self.shiftz);

            const shiftr_grad = engine.gradient(self.shiftr);

            const y = engine.value(self.y);

            const term1 = shiftr * sgrad[0] + shiftz * sgrad[1];
            const term2 = -lapse * y + shiftr * seed / rho;
            const term3 = shiftr_grad[0] / rho - shiftr / rho2;

            return term1 + term2 + term3;
        }
    };

    pub const YEvolution = struct {
        lapse: []const f64,
        shiftr: []const f64,
        shiftz: []const f64,
        psi: []const f64,
        seed: []const f64,
        y: []const f64,
        w: []const f64,

        pub fn project(self: YEvolution, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];
            const rho2 = rho * rho;

            const lapse = engine.value(self.lapse);
            const lgrad = engine.gradient(self.lapse);
            const lhess = engine.hessian(self.lapse);

            const shiftr = engine.value(self.shiftr);
            const shiftz = engine.value(self.shiftz);
            const shiftr_grad = engine.gradient(self.shiftr);
            const shiftz_grad = engine.gradient(self.shiftz);

            const psi = engine.value(self.psi);
            const pgrad = engine.gradient(self.psi);
            const phess = engine.hessian(self.psi);

            const seed = engine.value(self.seed);
            const sgrad = engine.gradient(self.seed);
            const shess = engine.hessian(self.seed);

            const w = engine.value(self.w);

            const y = engine.value(self.y);
            const ygrad = engine.gradient(self.y);

            const term1 = shiftr * ygrad[0] + shiftz * ygrad[1] + shiftr * y / rho;
            const term2 = w / rho * (shiftz_grad[0] - shiftr_grad[1]);

            const scale1 = @exp(-2.0 * rho * seed - 2.0 * psi);
            const term3 = lgrad[0] / rho * (rho * sgrad[0] + seed + 2.0 * pgrad[0]);
            const term4 = lgrad[0] / rho2 - lhess[0][0] / rho - lgrad[1] * sgrad[1];

            const scale2 = lapse * scale1;
            const term5 = pgrad[0] / rho * (rho * sgrad[0] + seed + pgrad[0]) - pgrad[1] * sgrad[1];
            const term6 = -shess[0][0] - sgrad[0] / rho - shess[1][1] + seed / rho2;
            const term7 = pgrad[0] / rho2 - phess[0][0] / rho;

            return term1 + term2 + scale1 * (term3 + term4) + scale2 * (term5 + term6 + term7);
        }
    };

    pub const UEvolution = struct {
        lapse: []const f64,
        shiftr: []const f64,
        shiftz: []const f64,
        psi: []const f64,
        seed: []const f64,
        u: []const f64,
        w: []const f64,

        pub fn project(self: UEvolution, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];

            const lapse = engine.value(self.lapse);
            const lgrad = engine.gradient(self.lapse);
            const lhess = engine.hessian(self.lapse);

            const shiftr = engine.value(self.shiftr);
            const shiftz = engine.value(self.shiftz);
            const shiftr_grad = engine.gradient(self.shiftr);
            const shiftz_grad = engine.gradient(self.shiftz);

            const psi = engine.value(self.psi);
            const pgrad = engine.gradient(self.psi);
            const phess = engine.hessian(self.psi);

            const seed = engine.value(self.seed);
            const sgrad = engine.gradient(self.seed);

            const w = engine.value(self.w);

            const ugrad = engine.gradient(self.u);

            const term1 = shiftr * ugrad[0] + shiftz * ugrad[1];
            const term2 = 2 * w * (shiftr_grad[1] - shiftz_grad[0]);

            const scale1 = @exp(-2 * rho * seed - 2 * psi);
            const term3 = 2 * lgrad[1] * (pgrad[1] + rho * sgrad[1]);
            const term4 = -2 * lgrad[0] * (rho * sgrad[0] + seed + pgrad[0]);
            const term5 = lhess[0][0] - lhess[1][1];

            const scale2 = lapse * scale1;
            const term6 = pgrad[1] * (2 * rho * sgrad[1] + pgrad[1]);
            const term7 = -pgrad[0] * (2 * rho * sgrad[0] + 2 * seed + pgrad[0]);
            const term8 = phess[0][0] - phess[1][1] - 2 * seed / rho - 2 * sgrad[0];

            return term1 + term2 + scale1 * (term3 + term4 + term5) + scale2 * (term6 + term7 + term8);
        }
    };

    pub const WEvolution = struct {
        lapse: []const f64,
        shiftr: []const f64,
        shiftz: []const f64,
        psi: []const f64,
        seed: []const f64,
        u: []const f64,
        w: []const f64,

        pub fn project(self: WEvolution, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];

            const lapse = engine.value(self.lapse);
            const lgrad = engine.gradient(self.lapse);
            const lhess = engine.hessian(self.lapse);

            const shiftr = engine.value(self.shiftr);
            const shiftz = engine.value(self.shiftz);
            const shiftr_grad = engine.gradient(self.shiftr);
            const shiftz_grad = engine.gradient(self.shiftz);

            const psi = engine.value(self.psi);
            const pgrad = engine.gradient(self.psi);
            const phess = engine.hessian(self.psi);

            const seed = engine.value(self.seed);
            const sgrad = engine.gradient(self.seed);

            const u = engine.value(self.u);

            const wgrad = engine.gradient(self.w);

            const term1 = shiftr * wgrad[0] + shiftz * wgrad[1];
            const term2 = 1.0 / 2.0 * u * (shiftz_grad[0] - shiftr_grad[1]);

            const scale1 = @exp(-2.0 * rho * seed - 2.0 * psi);
            const term3 = lgrad[0] * (rho * sgrad[1] + pgrad[1]);
            const term4 = lgrad[1] * (rho * sgrad[0] + seed + pgrad[0]);
            const term5 = -lhess[0][1];

            const scale2 = lapse * scale1;
            const term6 = pgrad[0] * (rho * sgrad[1] + pgrad[1]);
            const term7 = pgrad[1] * (rho * sgrad[0] + seed);
            const term8 = sgrad[1] - phess[0][1];

            return term1 + term2 + scale1 * (term3 + term4 + term5) + scale2 * (term6 + term7 + term8);
        }
    };

    pub const Dynamic = enum {
        psi,
        seed,
        u,
        w,
        y,
    };

    const Output = enum {
        psi,
        seed,
        u,
        y,
        w,
        constraint,
        lapse,
        shiftr,
        shiftz,
    };

    pub const Gauge = struct {
        gpa: Allocator,
        // Worker
        worker: *const NodeWorker,
        // Gauge variables
        lapse: []f64,
        shiftr: []f64,
        shiftz: []f64,
        // Scratch variables
        rhs: []f64,

        pub fn solve(self: Gauge, dynamic: SystemConst(Dynamic)) !void {
            const worker = self.worker.order(O);

            const psi = dynamic.field(.psi);
            const seed = dynamic.field(.seed);
            const u = dynamic.field(.u);
            const y = dynamic.field(.y);
            const w = dynamic.field(.w);

            // ***********************************
            // Solve Elliptic Gauges Conditions

            var solver = try MultigridMethod.init(self.gpa, self.worker.manager.numNodes(), BiCGStabSolver.new(20000, 10e-14), .{
                .max_iters = 100,
                .tolerance = 10e-10,
                .presmooth = 5,
                .postsmooth = 5,
            });
            defer solver.deinit();

            // ***********************************
            // Solve Lapse

            const lapse_rhs: LapseRhs = .{
                .psi = psi,
                .seed = seed,
                .u = u,
                .y = y,
                .w = w,
            };

            const lapse_op: LapseOp = .{
                .psi = psi,
                .seed = seed,
                .u = u,
                .w = w,
                .y = y,
            };

            @memset(self.lapse, 0.0);
            worker.project(lapse_rhs, self.rhs);

            try solver.solve(
                self.worker,
                lapse_op,
                lapse_boundary,
                self.lapse,
                self.rhs,
            );

            worker.fillGhostNodes(lapse_boundary, self.lapse);

            for (0..self.lapse.len) |i| {
                self.lapse[i] += 1.0;
            }

            // ***********************************
            // Solve Shift

            const shift_op: ShiftOp = .{};

            const shiftr_rhs: ShiftRRhs = .{
                .lapse = self.lapse,
                .u = u,
                .w = w,
            };

            const shiftz_rhs: ShiftZRhs = .{
                .lapse = self.lapse,
                .u = u,
                .w = w,
            };

            // ShiftR
            @memset(self.shiftr, 0.0);
            worker.project(shiftr_rhs, self.rhs);

            try solver.solve(
                self.worker,
                shift_op,
                shiftr_boundary,
                self.shiftr,
                self.rhs,
            );

            worker.fillGhostNodes(shiftr_boundary, self.shiftr);

            // ShiftZ
            @memset(self.shiftz, 0.0);
            worker.project(shiftz_rhs, self.rhs);

            try solver.solve(
                self.worker,
                shift_op,
                shiftz_boundary,
                self.shiftz,
                self.rhs,
            );

            worker.fillGhostNodes(shiftz_boundary, self.shiftz);
        }
    };

    pub const Evolution = struct {
        // GPA allocator
        gpa: Allocator,
        // Worker
        worker: *const NodeWorker,
        // Gauge variables
        lapse: []f64,
        shiftr: []f64,
        shiftz: []f64,
        // Scratch variables
        rhs: []f64,

        pub const Tag = Dynamic;
        pub const Error = error{OutOfMemory};

        pub fn preprocess(self: Evolution, dynamic: System(Tag)) Error!void {
            const psi = dynamic.field(.psi);
            const seed = dynamic.field(.seed);
            const u = dynamic.field(.u);
            const y = dynamic.field(.y);
            const w = dynamic.field(.w);

            const worker = self.worker.order(M);
            const worker0 = self.worker.order(0);

            for (0..self.worker.mesh.numLevels()) |rev_level| {
                const level = self.worker.mesh.numLevels() - 1 - rev_level;
                worker0.restrictLevel(level, psi);
                worker0.restrictLevel(level, seed);
                worker0.restrictLevel(level, u);
                worker0.restrictLevel(level, y);
                worker0.restrictLevel(level, w);
            }

            worker.fillGhostNodes(psi_boundary, psi);
            worker.fillGhostNodes(seed_boundary, seed);
            worker.fillGhostNodes(u_boundary, u);
            worker.fillGhostNodes(y_boundary, y);
            worker.fillGhostNodes(w_boundary, w);
        }

        pub fn derivative(self: Evolution, deriv: System(Tag), dynamic: SystemConst(Tag), time: f64) Error!void {
            _ = time;

            const worker = self.worker.order(O);

            const psi = dynamic.field(.psi);
            const seed = dynamic.field(.seed);
            const u = dynamic.field(.u);
            const y = dynamic.field(.y);
            const w = dynamic.field(.w);

            // ************************************
            // Solve Gauge

            const gauge: Gauge = .{
                .gpa = self.gpa,
                .worker = self.worker,
                .lapse = self.lapse,
                .shiftr = self.shiftr,
                .shiftz = self.shiftz,
                .rhs = self.rhs,
            };

            try gauge.solve(dynamic);

            // ************************************
            // Evolve Psi

            const psi_evolve: PsiEvolution = .{
                .lapse = self.lapse,
                .shiftr = self.shiftr,
                .shiftz = self.shiftz,
                .psi = psi,
                .u = u,
                .y = y,
            };

            worker.project(psi_evolve, deriv.field(.psi));

            // ************************************
            // Evolve seed

            const seed_evolve: SeedEvolution = .{
                .lapse = self.lapse,
                .shiftr = self.shiftr,
                .shiftz = self.shiftz,
                .seed = seed,
                .y = y,
            };

            worker.project(seed_evolve, deriv.field(.seed));

            // ************************************
            // Evolve Y

            const y_evolve: YEvolution = .{
                .lapse = self.lapse,
                .shiftr = self.shiftr,
                .shiftz = self.shiftz,
                .psi = psi,
                .seed = seed,
                .y = y,
                .w = w,
            };

            worker.project(y_evolve, deriv.field(.y));

            // ***********************************
            // Evolve U

            const u_evolve: UEvolution = .{
                .lapse = self.lapse,
                .shiftr = self.shiftr,
                .shiftz = self.shiftz,
                .psi = psi,
                .seed = seed,
                .u = u,
                .w = w,
            };

            worker.project(u_evolve, deriv.field(.u));

            // ************************************
            // Evolve W

            const w_evolve: WEvolution = .{
                .lapse = self.lapse,
                .shiftr = self.shiftr,
                .shiftz = self.shiftz,
                .psi = psi,
                .seed = seed,
                .u = u,
                .w = w,
            };

            worker.project(w_evolve, deriv.field(.w));

            const eps = 10.0;

            const workerm = self.worker.order(M);

            workerm.dissipation(eps, deriv.field(.psi), psi);
            workerm.dissipation(eps, deriv.field(.seed), seed);
            workerm.dissipation(eps, deriv.field(.u), u);
            workerm.dissipation(eps, deriv.field(.w), w);
            workerm.dissipation(eps, deriv.field(.y), y);
        }
    };

    // Run
    fn run(allocator: Allocator) !void {
        std.debug.print("Running Brill Evolution\n", .{});

        var mesh = try TreeMesh.init(allocator, .{
            .origin = [2]f64{ 0.0, -10.0 },
            .size = [2]f64{ 20.0, 20.0 },
        });
        defer mesh.deinit();

        // Globally refine two times
        for (0..3) |r| {
            std.debug.print("Running Global Refinement {}\n", .{r});

            try mesh.refineGlobal(allocator);
        }

        for (0..1) |r| {
            std.debug.print("Running Refinement {}\n", .{r});

            const flags = try allocator.alloc(bool, mesh.numCells());
            defer allocator.free(flags);

            @memset(flags, false);

            for (0..mesh.cells.len) |cell_id| {
                const bounds: RealBox = mesh.cells.items(.bounds)[cell_id];
                const center = bounds.center();

                const radius = @sqrt((center[0]) * (center[0]) + (center[1]) * (center[1]));

                if (radius < 2) {
                    flags[cell_id] = true;
                }
            }

            try mesh.refine(allocator, flags);
        }

        var manager = try NodeManager.init(allocator, [1]usize{16} ** N, 8);
        defer manager.deinit();

        try manager.build(allocator, &mesh);

        std.debug.print("Num nodes: {}\n", .{manager.numNodes()});
        std.debug.print("Min spacing {}\n", .{manager.minSpacing()});

        // Create worker
        var worker = try NodeWorker.init(&mesh, &manager);
        defer worker.deinit();

        std.debug.print("Allocating Memory\n", .{});

        const rhs = try allocator.alloc(f64, manager.numNodes());
        defer allocator.free(rhs);

        const lapse = try allocator.alloc(f64, manager.numNodes());
        defer allocator.free(lapse);

        const shiftr = try allocator.alloc(f64, manager.numNodes());
        defer allocator.free(shiftr);

        const shiftz = try allocator.alloc(f64, manager.numNodes());
        defer allocator.free(shiftz);

        const constraint = try allocator.alloc(f64, manager.numNodes());
        defer allocator.free(constraint);

        const eta = try allocator.alloc(f64, manager.numNodes());
        defer allocator.free(eta);

        // Runge Kutta 4 context
        var rk4 = try Rk4Integrator(Dynamic).init(allocator, manager.numNodes());
        defer rk4.deinit();

        @memset(rk4.sys.field(.u), 0.0);
        @memset(rk4.sys.field(.y), 0.0);
        @memset(rk4.sys.field(.w), 0.0);

        // *****************************
        // Initial Data

        std.debug.print("Solving Initial Data\n", .{});

        // Seed
        worker.order(O).project(Seed{ .amplitude = 1.0, .sigma = 1.0 }, rk4.sys.field(.seed));
        worker.order(O).fillGhostNodes(seed_boundary, rk4.sys.field(.seed));

        // Conformal factor
        const initial_rhs: InitialEtaRhs = .{
            .seed = rk4.sys.field(.seed),
        };

        const initial_op: InitialEtaOp = .{
            .seed = rk4.sys.field(.seed),
        };

        worker.order(O).project(initial_rhs, rhs);

        var solver = try MultigridMethod.init(allocator, manager.numNodes(), BiCGStabSolver.new(20000, 10e-14), .{
            .max_iters = 100,
            .tolerance = 10e-10,
            .presmooth = 5,
            .postsmooth = 5,
        });
        defer solver.deinit();

        @memset(eta, 0.0);

        try solver.solve(
            &worker,
            initial_op,
            eta_boundary,
            eta,
            rhs,
        );

        {
            const psi = rk4.sys.field(.psi);

            for (0..eta.len) |i| {
                eta[i] += 1.0;
            }

            for (0..eta.len) |i| {
                psi[i] = 2.0 * @log(eta[i]);
            }
        }

        std.debug.print("Running Evolution\n", .{});

        // ******************************
        // Step 0

        {
            const evolution: Evolution = .{
                .gpa = allocator,
                .worker = &worker,
                .lapse = lapse,
                .shiftr = shiftr,
                .shiftz = shiftz,
                .rhs = rhs,
            };

            try evolution.preprocess(rk4.sys);

            const gauge: Gauge = .{
                .gpa = allocator,
                .worker = &worker,
                .lapse = lapse,
                .shiftr = shiftr,
                .shiftz = shiftz,
                .rhs = rhs,
            };

            try gauge.solve(rk4.sys.toConst());

            const hamiltonain: Hamiltonian = .{
                .eta = eta,
                .seed = rk4.sys.field(.seed),
                .u = rk4.sys.field(.u),
                .y = rk4.sys.field(.y),
                .w = rk4.sys.field(.w),
            };

            worker.order(O).project(hamiltonain, constraint);
        }

        // Output
        {
            const file = try std.fs.cwd().createFile("output/evolution0.vtu", .{});
            defer file.close();

            const output = SystemConst(Output).view(manager.numNodes(), .{
                .psi = rk4.sys.field(.psi),
                .seed = rk4.sys.field(.seed),
                .u = rk4.sys.field(.u),
                .y = rk4.sys.field(.y),
                .w = rk4.sys.field(.w),
                .lapse = lapse,
                .shiftr = shiftr,
                .shiftz = shiftz,
                .constraint = constraint,
            });

            var buffer = std.io.bufferedWriter(file.writer());

            try DataOut.writeVtk(
                Output,
                allocator,
                &worker,
                output,
                buffer.writer(),
            );

            try buffer.flush();
        }

        // ******************************
        // Evolve

        // Build scratch allocator
        var arena: ArenaAllocator = ArenaAllocator.init(allocator);
        defer arena.deinit();

        const scratch: Allocator = arena.allocator();

        const steps: usize = 100;
        const h: f64 = cfl * manager.minSpacing();

        for (0..steps) |step| {
            const con = worker.norm(constraint);
            std.debug.print("Step {}/{}, Time: {}, Constraint: {}\n", .{ step + 1, steps, rk4.time, con });

            // ********************************
            // Step

            const evolution: Evolution = .{
                .gpa = scratch,
                .worker = &worker,
                .lapse = lapse,
                .shiftr = shiftr,
                .shiftz = shiftz,
                .rhs = rhs,
            };

            try rk4.step(evolution, h);
            try evolution.preprocess(rk4.sys);

            _ = arena.reset(.retain_capacity);

            // ******************************
            // Gauge

            const gauge: Gauge = .{
                .gpa = scratch,
                .worker = &worker,
                .lapse = lapse,
                .shiftr = shiftr,
                .shiftz = shiftz,
                .rhs = rhs,
            };

            try gauge.solve(rk4.sys.toConst());

            _ = arena.reset(.retain_capacity);

            // *******************************
            // Constraint

            {
                const psi = rk4.sys.field(.psi);

                worker.order(O).fillGhostNodes(psi_boundary, psi);

                for (0..eta.len) |i| {
                    eta[i] = @exp(psi[i] / 2.0);
                }
            }

            const hamiltonian: Hamiltonian = .{
                .eta = eta,
                .seed = rk4.sys.field(.seed),
                .u = rk4.sys.field(.u),
                .y = rk4.sys.field(.y),
                .w = rk4.sys.field(.w),
            };

            worker.order(O).project(hamiltonian, constraint);

            // *******************************
            // Output

            const file_name = try std.fmt.allocPrint(allocator, "output/evolution{}.vtu", .{step + 1});
            defer allocator.free(file_name);

            const file = try std.fs.cwd().createFile(file_name, .{});
            defer file.close();

            const output = SystemConst(Output).view(manager.numNodes(), .{
                .psi = rk4.sys.field(.psi),
                .seed = rk4.sys.field(.seed),
                .u = rk4.sys.field(.u),
                .y = rk4.sys.field(.y),
                .w = rk4.sys.field(.w),
                .lapse = lapse,
                .shiftr = shiftr,
                .shiftz = shiftz,
                .constraint = constraint,
            });

            var buffer = std.io.bufferedWriter(file.writer());

            try DataOut.writeVtk(
                Output,
                allocator,
                &worker,
                output,
                buffer.writer(),
            );

            try buffer.flush();

            _ = arena.reset(.retain_capacity);
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
    try BrillEvolution.run(gpa.allocator());
}
