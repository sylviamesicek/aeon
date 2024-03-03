//! Uses old inexact equations and boundary conditions.

// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const aeon = @import("aeon");

const common = aeon.common;
const geometry = aeon.geometry;
const lac = aeon.lac;
const mesh_ = aeon.mesh;

pub const N = 2;
pub const M = 3;
pub const O = 2;

const cfl: f64 = 0.1;

pub const BrillEvolution = struct {
    const DataOut = aeon.DataOut(N, M);

    const BoundaryKind = common.BoundaryKind;
    const Engine = common.Engine(N, M, O);
    const Rk4Integrator = common.RungeKutta4Integrator;
    const System = common.System;
    const SystemConst = common.SystemConst;

    const FaceIndex = geometry.FaceIndex(N);
    const IndexSpace = geometry.IndexSpace(N);
    const IndexMixin = geometry.IndexMixin(N);
    const RealBox = geometry.RealBox(N);

    const BiCGStabSolver = lac.BiCGStabSolver;

    const Mesh = mesh_.Mesh(N);
    const NodeManager = mesh_.NodeManager(N, M);

    const MultigridMethod = mesh_.MultigridMethod(N, M, BiCGStabSolver);

    pub const RField = struct {
        pub fn eval(_: @This(), pos: [N]f64) f64 {
            const r2 = pos[0] * pos[0] + pos[1] * pos[1];
            return -@abs(pos[0]) / r2;
        }
    };

    pub const ZField = struct {
        pub fn eval(_: @This(), pos: [N]f64) f64 {
            const r2 = pos[0] * pos[0] + pos[1] * pos[1];
            return -@abs(pos[1]) / r2;
        }
    };

    pub const RFlatness = struct {
        comptime robin_value: RField = .{},
        comptime robin_rhs: common.ZeroField(N) = .{},

        pub const kind: BoundaryKind = .robin;
        pub const priority: usize = 0;
    };

    pub const ZFlatness = struct {
        comptime robin_value: ZField = .{},
        comptime robin_rhs: common.ZeroField(N) = .{},

        pub const kind: BoundaryKind = .robin;
        pub const priority: usize = 0;
    };

    pub const EvenBoundarySet = struct {
        pub const card: usize = 3;

        pub fn boundaryIdFromFace(face: FaceIndex) usize {
            if (face.axis == 0 and face.side == false) {
                return 0;
            } else if (face.axis == 0 and face.side == true) {
                return 1;
            } else {
                return 2;
            }
        }

        pub const BoundaryType0: type = common.EvenBoundary;
        pub const BoundaryType1: type = RFlatness;
        pub const BoundaryType2: type = ZFlatness;

        pub fn boundary0(_: @This()) BoundaryType0 {
            return .{};
        }

        pub fn boundary1(_: @This()) BoundaryType1 {
            return .{};
        }

        pub fn boundary2(_: @This()) BoundaryType2 {
            return .{};
        }
    };

    pub const OddBoundarySet = struct {
        pub const card: usize = 3;

        pub fn boundaryIdFromFace(face: FaceIndex) usize {
            if (face.axis == 0 and face.side == false) {
                return 0;
            } else if (face.axis == 0) {
                return 1;
            } else {
                return 2;
            }
        }

        pub const BoundaryType0: type = common.OddBoundary;
        pub const BoundaryType1: type = RFlatness;
        pub const BoundaryType2: type = ZFlatness;

        pub fn boundary0(_: @This()) BoundaryType0 {
            return .{};
        }

        pub fn boundary1(_: @This()) BoundaryType1 {
            return .{};
        }

        pub fn boundary2(_: @This()) BoundaryType2 {
            return .{};
        }
    };

    const lapse_boundary = EvenBoundarySet{};
    const shiftr_boundary = OddBoundarySet{};
    const shiftz_boundary = EvenBoundarySet{};
    const psi_boundary = EvenBoundarySet{};
    const seed_boundary = OddBoundarySet{};
    const w_boundary = OddBoundarySet{};
    const u_boundary = EvenBoundarySet{};
    const x_boundary = OddBoundarySet{};
    const constraint_boundary = EvenBoundarySet{};

    pub const Seed = struct {
        amplitude: f64,
        sigma: f64,

        pub const order: usize = O;

        pub fn eval(self: Seed, engine: Engine) f64 {
            const pos = engine.position();

            const rho = pos[0];
            const z = pos[1];

            const rho2 = rho * rho;
            const z2 = z * z;
            const sigma2 = self.sigma * self.sigma;

            return self.amplitude * rho * @exp(-(rho2 + z2) / sigma2);
        }
    };

    pub const InitialPsiRhs = struct {
        seed: []const f64,

        pub const order: usize = O;

        pub fn eval(self: InitialPsiRhs, engine: Engine) f64 {
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

    pub const InitialPsiOp = struct {
        seed: []const f64,

        pub const order: usize = O;

        pub fn apply(self: InitialPsiOp, engine: Engine, psi: []const f64) f64 {
            const position: [N]f64 = engine.position();
            const r = position[0];

            const hessian: [N][N]f64 = engine.hessian(psi);
            const grad: [N]f64 = engine.gradient(psi);
            const val: f64 = engine.value(psi);

            const shessian: [N][N]f64 = engine.hessian(self.seed);
            const sgrad: [N]f64 = engine.gradient(self.seed);

            var term1 = hessian[0][0] + hessian[1][1];

            if (r == 0.0) {
                term1 += hessian[0][0];
            } else {
                term1 += grad[0] / r;
            }

            const term2 = val / 4.0 * (r * shessian[0][0] + 2.0 * sgrad[0] + r * shessian[1][1]);

            return term1 + term2;
        }

        pub fn applyDiag(self: InitialPsiOp, engine: Engine) f64 {
            const position: [N]f64 = engine.position();
            const r = position[0];

            const hessian: [N][N]f64 = engine.hessianDiag();
            const grad: [N]f64 = engine.gradientDiag();
            const val: f64 = engine.valueDiag();

            const shessian: [N][N]f64 = engine.hessian(self.seed);
            const sgrad: [N]f64 = engine.gradient(self.seed);

            var term1 = hessian[0][0] + hessian[1][1];

            if (r == 0.0) {
                term1 += hessian[0][0];
            } else {
                term1 += grad[0] / r;
            }

            const term2 = val / 4.0 * (r * shessian[0][0] + 2.0 * sgrad[0] + r * shessian[1][1]);

            return term1 + term2;
        }
    };

    pub const LapseRhs = struct {
        psi: []const f64,
        seed: []const f64,
        u: []const f64,
        w: []const f64,
        x: []const f64,

        pub const order: usize = O;

        pub fn eval(self: LapseRhs, engine: Engine) f64 {
            const pos: [N]f64 = engine.position();
            const rho = pos[0];

            const psi = engine.value(self.psi) + 1.0;
            const seed = engine.value(self.seed);
            const u = engine.value(self.u);
            const w = engine.value(self.w);
            const x = engine.value(self.x);

            const scale = @exp(2.0 * rho * seed) * psi * psi * psi * psi;

            const term1 = 2.0 / 3.0 * (rho * rho * w * w + rho * u * w + u * u);
            const term2 = 2.0 * x * x;

            return scale * (term1 + term2);
        }
    };

    pub const LapseOp = struct {
        psi: []const f64,
        seed: []const f64,
        u: []const f64,
        x: []const f64,
        w: []const f64,

        pub const order: usize = O;

        pub fn apply(self: LapseOp, engine: Engine, lapse: []const f64) f64 {
            const pos = engine.position();
            const rho = pos[0];

            const hessian: [N][N]f64 = engine.hessian(lapse);
            const grad: [N]f64 = engine.gradient(lapse);
            const val: f64 = engine.value(lapse);

            const psi = engine.value(self.psi) + 1.0;
            const pgrad = engine.gradient(self.psi);

            const seed = engine.value(self.seed);
            const u = engine.value(self.u);
            const x = engine.value(self.x);
            const w = engine.value(self.w);

            var term1 = hessian[0][0] + hessian[1][1];

            if (rho == 0.0) {
                term1 += hessian[0][0];
            } else {
                term1 += grad[0] / rho;
            }

            const term2 = 2.0 / psi * (pgrad[0] * grad[0] + pgrad[1] * grad[1]);

            const scale = -val * @exp(2.0 * rho * seed) * psi * psi * psi * psi;

            const term3 = 2.0 / 3.0 * (rho * rho * w * w + rho * u * w + u * u);
            const term4 = 2.0 * x * x;

            return term1 + term2 + scale * (term3 + term4);
        }

        pub fn applyDiag(self: LapseOp, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];

            const hessian: [N][N]f64 = engine.hessianDiag();
            const grad: [N]f64 = engine.gradientDiag();
            const val: f64 = engine.valueDiag();

            const psi = engine.value(self.psi) + 1.0;
            const pgrad = engine.gradient(self.psi);

            const seed = engine.value(self.seed);
            const u = engine.value(self.u);
            const x = engine.value(self.x);
            const w = engine.value(self.w);

            var term1 = hessian[0][0] + hessian[1][1];

            if (rho == 0.0) {
                term1 += hessian[0][0];
            } else {
                term1 += grad[0] / rho;
            }

            const term2 = 2.0 / psi * (pgrad[0] * grad[0] + pgrad[1] * grad[1]);

            const scale = -val * @exp(2.0 * rho * seed) * psi * psi * psi * psi;

            const term3 = 2.0 / 3.0 * (rho * rho * w * w + rho * u * w + u * u);
            const term4 = 2.0 * x * x;

            return term1 + term2 + scale * (term3 + term4);
        }
    };

    pub const ShiftRRhs = struct {
        lapse: []const f64,
        x: []const f64,
        u: []const f64,

        pub const order: usize = O;

        pub fn eval(self: ShiftRRhs, engine: Engine) f64 {
            const lapse = engine.value(self.lapse) + 1.0;
            const lgrad = engine.gradient(self.lapse);

            const x = engine.value(self.x);
            const xgrad = engine.gradient(self.x);

            const u = engine.value(self.u);
            const ugrad = engine.gradient(self.u);

            const term1 = 2.0 * (x * lgrad[1] + lapse * xgrad[1]);
            const term2 = -u * lgrad[0] - lapse * ugrad[0];

            return term1 + term2;
        }
    };

    pub const ShiftZRhs = struct {
        lapse: []const f64,
        x: []const f64,
        u: []const f64,

        pub const order: usize = O;

        pub fn eval(self: ShiftZRhs, engine: Engine) f64 {
            const lapse = engine.value(self.lapse) + 1.0;
            const lgrad = engine.gradient(self.lapse);

            const x = engine.value(self.x);
            const xgrad = engine.gradient(self.x);

            const u = engine.value(self.u);
            const ugrad = engine.gradient(self.u);

            const term1 = 2.0 * (x * lgrad[0] + lapse * xgrad[0]);
            const term2 = u * lgrad[1] + lapse * ugrad[1];

            return term1 + term2;
        }
    };

    pub const ShiftOp = struct {
        pub const order: usize = O;

        pub fn apply(_: ShiftOp, engine: Engine, shift: []const f64) f64 {
            return engine.laplacian(shift);
        }

        pub fn applyDiag(_: ShiftOp, engine: Engine) f64 {
            return engine.laplacianDiag();
        }
    };

    pub const Hamiltonian = struct {
        psi: []const f64,
        seed: []const f64,
        u: []const f64,
        x: []const f64,
        w: []const f64,

        pub const order: usize = O;

        pub fn eval(self: Hamiltonian, engine: Engine) f64 {
            const pos: [N]f64 = engine.position();
            const rho = pos[0];

            const phess: [N][N]f64 = engine.hessian(self.psi);
            const pgrad: [N]f64 = engine.gradient(self.psi);
            const psi: f64 = engine.value(self.psi) + 1.0;

            const shess: [N][N]f64 = engine.hessian(self.seed);
            const sgrad: [N]f64 = engine.gradient(self.seed);
            const seed: f64 = engine.value(self.seed);

            const u = engine.value(self.u);
            const x = engine.value(self.x);
            const w = engine.value(self.w);

            var term1 = phess[0][0] + phess[1][1];

            if (rho == 0.0) {
                term1 += phess[0][0];
            } else {
                term1 += pgrad[0] / rho;
            }

            const term2 = psi / 4.0 * (rho * shess[0][0] + 2.0 * sgrad[0] + rho * shess[1][1]);

            const scale = psi * psi * psi * psi * psi * @exp(2.0 * rho * seed) / 4.0;
            const term3 = 1.0 / 3.0 * (rho * rho * w * w + rho * u * w + u * u);
            const term4 = x * x;

            return term1 + term2 + scale * (term3 + term4);
        }
    };

    pub const PsiEvolution = struct {
        lapse: []const f64,
        shiftr: []const f64,
        shiftz: []const f64,
        psi: []const f64,
        u: []const f64,
        w: []const f64,

        pub const order: usize = O;

        pub fn eval(self: PsiEvolution, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];

            const lapse = engine.value(self.lapse) + 1.0;
            const shiftr = engine.value(self.shiftr);
            const shiftz = engine.value(self.shiftz);
            const shiftr_grad = engine.gradient(self.shiftr);

            const psi = engine.value(self.psi) + 1.0;
            const pgrad = engine.gradient(self.psi);
            const u = engine.value(self.u);
            const w = engine.value(self.w);

            const term1 = shiftr * pgrad[0] + shiftz * pgrad[1];
            const term2 = psi * lapse / 6.0 * (u + 2 * rho * w);

            if (rho == 0.0) {
                return term1 + term2 + psi / 2.0 * shiftr_grad[0];
            } else {
                return term1 + term2 + psi * shiftr / (2.0 * rho);
            }
        }
    };

    pub const SeedEvolution = struct {
        lapse: []const f64,
        shiftr: []const f64,
        shiftz: []const f64,
        seed: []const f64,
        w: []const f64,

        pub const order: usize = O;

        pub fn eval(self: SeedEvolution, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];
            const rho2 = rho * rho;

            if (rho == 0.0) {
                return 0.0;
            }

            const seed = engine.value(self.seed);
            const sgrad = engine.gradient(self.seed);

            const lapse = engine.value(self.lapse) + 1.0;
            const shiftr = engine.value(self.shiftr);
            const shiftz = engine.value(self.shiftz);

            const shiftr_grad = engine.gradient(self.shiftr);

            const w = engine.value(self.w);

            const term1 = shiftr * sgrad[0] + shiftz * sgrad[1];
            const term2 = -lapse * w + shiftr * seed / rho;
            const term3 = shiftr_grad[0] / rho - shiftr / rho2;

            return term1 + term2 + term3;
        }
    };

    pub const WEvolution = struct {
        lapse: []const f64,
        shiftr: []const f64,
        shiftz: []const f64,
        psi: []const f64,
        seed: []const f64,
        x: []const f64,
        w: []const f64,

        pub const order: usize = O;

        pub fn eval(self: WEvolution, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];
            const rho2 = rho * rho;

            if (rho == 0.0) {
                return 0.0;
            }

            const lapse = engine.value(self.lapse) + 1.0;
            const lgrad = engine.gradient(self.lapse);
            const lhess = engine.hessian(self.lapse);

            const shiftr = engine.value(self.shiftr);
            const shiftz = engine.value(self.shiftz);
            const shiftr_grad = engine.gradient(self.shiftr);
            const shiftz_grad = engine.gradient(self.shiftz);

            const psi = engine.value(self.psi) + 1.0;
            const pgrad = engine.gradient(self.psi);
            const phess = engine.hessian(self.psi);

            const seed = engine.value(self.seed);
            const sgrad = engine.gradient(self.seed);
            const shess = engine.hessian(self.seed);

            const x = engine.value(self.x);
            const w = engine.value(self.w);
            const wgrad = engine.gradient(self.w);

            const term1 = shiftr * wgrad[0] + shiftz * wgrad[1] + shiftr * w / rho;
            const term2 = x / rho * (shiftz_grad[0] - shiftr_grad[1]);

            const scale1 = @exp(-2.0 * rho * seed) / (psi * psi * psi * psi);
            const term3 = lgrad[0] / rho * (rho * sgrad[0] + seed + 4.0 * pgrad[0] / psi);
            const term4 = lgrad[0] / rho2 - lhess[0][0] / rho - lgrad[1] * sgrad[1];

            const scale2 = lapse * scale1;
            const term5 = 2.0 * pgrad[0] / (rho * psi) * (rho * sgrad[0] + seed + 3.0 * pgrad[0] / psi) - 2.0 * pgrad[1] / psi * sgrad[1];
            const term6 = -shess[0][0] - sgrad[0] / rho - shess[1][1] + seed / rho2;
            const term7 = 2.0 / psi * (pgrad[0] / rho2 - phess[0][0] / rho);

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
        x: []const f64,

        pub const order: usize = O;

        pub fn eval(self: UEvolution, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];

            const lapse = engine.value(self.lapse) + 1.0;
            const lgrad = engine.gradient(self.lapse);
            const lhess = engine.hessian(self.lapse);

            const shiftr = engine.value(self.shiftr);
            const shiftz = engine.value(self.shiftz);
            const shiftr_grad = engine.gradient(self.shiftr);
            const shiftz_grad = engine.gradient(self.shiftz);

            const psi = engine.value(self.psi) + 1.0;
            const pgrad = engine.gradient(self.psi);
            const phess = engine.hessian(self.psi);

            const seed = engine.value(self.seed);
            const sgrad = engine.gradient(self.seed);

            const x = engine.value(self.x);
            const ugrad = engine.gradient(self.u);

            const term1 = shiftr * ugrad[0] + shiftz * ugrad[1];
            const term2 = 2 * x * (shiftr_grad[1] - shiftz_grad[0]);

            const scale1 = @exp(-2 * rho * seed) / (psi * psi * psi * psi);
            const term3 = 2 * lgrad[1] * (2 * pgrad[1] / psi + rho * sgrad[1]);
            const term4 = -2 * lgrad[0] * (rho * sgrad[0] + seed + 2 * pgrad[0] / psi);
            const term5 = lhess[0][0] - lhess[1][1];

            const scale2 = 2 * lapse * scale1;
            const term6 = pgrad[1] / psi * (2 * rho * sgrad[1] + 3 * pgrad[1] / psi);
            const term7 = -pgrad[0] / psi * (2 * rho * sgrad[0] + 2 * seed + 3 * pgrad[0] / psi);
            var term8 = phess[0][0] / psi - phess[1][1] / psi - sgrad[0];

            if (rho == 0.0) {
                term8 -= sgrad[0];
            } else {
                term8 -= seed / rho;
            }

            return term1 + term2 + scale1 * (term3 + term4 + term5) + scale2 * (term6 + term7 + term8);
        }
    };

    pub const XEvolution = struct {
        lapse: []const f64,
        shiftr: []const f64,
        shiftz: []const f64,
        psi: []const f64,
        seed: []const f64,
        u: []const f64,
        x: []const f64,

        pub const order: usize = O;

        pub fn eval(self: XEvolution, engine: Engine) f64 {
            const pos = engine.position();
            const rho = pos[0];

            if (rho == 0.0) {
                return 0.0;
            }

            const lapse = engine.value(self.lapse) + 1.0;
            const lgrad = engine.gradient(self.lapse);
            const lhess = engine.hessian(self.lapse);

            const shiftr = engine.value(self.shiftr);
            const shiftz = engine.value(self.shiftz);
            const shiftr_grad = engine.gradient(self.shiftr);
            const shiftz_grad = engine.gradient(self.shiftz);

            const psi = engine.value(self.psi) + 1.0;
            const pgrad = engine.gradient(self.psi);
            const phess = engine.hessian(self.psi);

            const seed = engine.value(self.seed);
            const sgrad = engine.gradient(self.seed);

            const u = engine.value(self.u);
            const xgrad = engine.gradient(self.x);

            const term1 = shiftr * xgrad[0] + shiftz * xgrad[1];
            const term2 = 1.0 / 2.0 * u * (shiftz_grad[0] - shiftr_grad[1]);

            const scale1 = @exp(-2.0 * rho * seed) / (psi * psi * psi * psi);
            const term3 = lgrad[0] * (rho * sgrad[1] + 2 * pgrad[1] / psi);
            const term4 = lgrad[1] * (rho * sgrad[0] + seed + 2 * pgrad[0] / psi);
            const term5 = -lhess[0][1];

            const scale2 = lapse * scale1;
            const term6 = pgrad[0] / psi * (2 * rho * sgrad[1] + 6 * pgrad[1] / psi);
            const term7 = pgrad[1] / psi * (2 * rho * sgrad[0] + 2 * seed);
            const term8 = sgrad[1] - 2 * phess[0][1] / psi;

            return term1 + term2 + scale1 * (term3 + term4 + term5) + scale2 * (term6 + term7 + term8);
        }
    };

    pub const Dynamic = enum {
        psi,
        seed,
        u,
        w,
        x,
    };

    const Output = enum {
        psi,
        seed,
        u,
        x,
        w,
        constraint,
        lapse,
        shiftr,
        shiftz,
    };

    pub const Gauge = struct {
        gpa: Allocator,
        // Worker
        manager: *const NodeManager,
        // Gauge variables
        lapse: []f64,
        shiftr: []f64,
        shiftz: []f64,
        // Scratch variables
        rhs: []f64,

        pub fn solve(self: Gauge, dynamic: SystemConst(Dynamic)) !void {
            const psi = dynamic.field(.psi);
            const seed = dynamic.field(.seed);
            const u = dynamic.field(.u);
            const x = dynamic.field(.x);
            const w = dynamic.field(.w);

            // ***********************************
            // Solve Elliptic Gauges Conditions

            var solver = try MultigridMethod.init(self.gpa, self.manager.numNodes(), BiCGStabSolver.new(20000, 10e-14), .{
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
                .x = x,
                .w = w,
            };

            const lapse_op: LapseOp = .{
                .psi = psi,
                .seed = seed,
                .u = u,
                .w = w,
                .x = x,
            };

            @memset(self.lapse, 0.0);
            self.manager.project(lapse_rhs, self.rhs);

            try solver.solve(
                self.manager,
                lapse_op,
                lapse_boundary,
                self.lapse,
                self.rhs,
            );

            self.manager.fillGhostNodes(O, lapse_boundary, self.lapse);

            // ***********************************
            // Solve Shift

            const shift_op: ShiftOp = .{};

            const shiftr_rhs: ShiftRRhs = .{
                .lapse = self.lapse,
                .u = u,
                .x = x,
            };

            const shiftz_rhs: ShiftZRhs = .{
                .lapse = self.lapse,
                .u = u,
                .x = x,
            };

            // ShiftR
            @memset(self.shiftr, 0.0);
            self.manager.project(shiftr_rhs, self.rhs);

            try solver.solve(
                self.manager,
                shift_op,
                shiftr_boundary,
                self.shiftr,
                self.rhs,
            );

            self.manager.fillGhostNodes(O, shiftr_boundary, self.shiftr);

            // ShiftZ
            @memset(self.shiftz, 0.0);
            self.manager.project(shiftz_rhs, self.rhs);

            try solver.solve(
                self.manager,
                shift_op,
                shiftz_boundary,
                self.shiftz,
                self.rhs,
            );

            self.manager.fillGhostNodes(O, shiftz_boundary, self.shiftz);
        }
    };

    pub const Evolution = struct {
        // GPA allocator
        gpa: Allocator,
        // Worker
        manager: *const NodeManager,
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
            const x = dynamic.field(.x);
            const w = dynamic.field(.w);

            for (0..self.manager.numLevels()) |rev_level| {
                const level = self.manager.numLevels() - 1 - rev_level;

                self.manager.restrictLevel(0, level, psi);
                self.manager.restrictLevel(0, level, seed);
                self.manager.restrictLevel(0, level, u);
                self.manager.restrictLevel(0, level, x);
                self.manager.restrictLevel(0, level, w);
            }

            self.manager.fillGhostNodes(O, psi_boundary, psi);
            self.manager.fillGhostNodes(O, seed_boundary, seed);
            self.manager.fillGhostNodes(O, u_boundary, u);
            self.manager.fillGhostNodes(O, x_boundary, x);
            self.manager.fillGhostNodes(O, w_boundary, w);
        }

        pub fn derivative(self: Evolution, deriv: System(Tag), dynamic: SystemConst(Tag), time: f64) Error!void {
            _ = time;

            const psi = dynamic.field(.psi);
            const seed = dynamic.field(.seed);
            const u = dynamic.field(.u);
            const x = dynamic.field(.x);
            const w = dynamic.field(.w);

            // ************************************
            // Solve Gauge

            const gauge: Gauge = .{
                .gpa = self.gpa,
                .manager = self.manager,
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
                .w = w,
            };

            self.manager.project(psi_evolve, deriv.field(.psi));

            // ************************************
            // Evolve seed

            const seed_evolve: SeedEvolution = .{
                .lapse = self.lapse,
                .shiftr = self.shiftr,
                .shiftz = self.shiftz,
                .seed = seed,
                .w = w,
            };

            self.manager.project(seed_evolve, deriv.field(.seed));

            // ************************************
            // Evolve W

            const w_evolve: WEvolution = .{
                .lapse = self.lapse,
                .shiftr = self.shiftr,
                .shiftz = self.shiftz,
                .psi = psi,
                .seed = seed,
                .w = w,
                .x = x,
            };

            self.manager.project(w_evolve, deriv.field(.w));

            // ***********************************
            // Evolve U

            const u_evolve: UEvolution = .{
                .lapse = self.lapse,
                .shiftr = self.shiftr,
                .shiftz = self.shiftz,
                .psi = psi,
                .seed = seed,
                .u = u,
                .x = x,
            };

            self.manager.project(u_evolve, deriv.field(.u));

            // ************************************
            // Evolve X

            const x_evolve: XEvolution = .{
                .lapse = self.lapse,
                .shiftr = self.shiftr,
                .shiftz = self.shiftz,
                .psi = psi,
                .seed = seed,
                .u = u,
                .x = x,
            };

            self.manager.project(x_evolve, deriv.field(.x));

            // const eps = 10.0;

            // self.manager.dissipation(M, eps, deriv.field(.psi), psi);
            // self.manager.dissipation(M, eps, deriv.field(.seed), seed);
            // self.manager.dissipation(M, eps, deriv.field(.u), u);
            // self.manager.dissipation(M, eps, deriv.field(.w), w);
            // self.manager.dissipation(M, eps, deriv.field(.x), x);
        }
    };

    // Run
    fn run(allocator: Allocator) !void {
        std.debug.print("Running Brill Evolution\n", .{});

        var mesh = try Mesh.init(allocator, .{
            .origin = [2]f64{ 0.0, -10.0 },
            .size = [2]f64{ 20.0, 20.0 },
        });
        defer mesh.deinit();

        // Globally refine two times
        for (0..3) |r| {
            std.debug.print("Running Global Refinement {}\n", .{r});

            try mesh.refineGlobal(allocator);
        }

        // for (0..1) |r| {
        //     std.debug.print("Running Refinement {}\n", .{r});

        //     const flags = try allocator.alloc(bool, mesh.numCells());
        //     defer allocator.free(flags);

        //     @memset(flags, false);

        //     for (0..mesh.cells.len) |cell_id| {
        //         const bounds: RealBox = mesh.cells.items(.bounds)[cell_id];
        //         const center = bounds.center();

        //         const radius = @sqrt((center[0]) * (center[0]) + (center[1]) * (center[1]));

        //         if (radius < 2) {
        //             flags[cell_id] = true;
        //         }
        //     }

        //     try mesh.refine(allocator, flags);
        // }

        var manager = try NodeManager.init(allocator, &mesh, [1]usize{16} ** N, 8);
        defer manager.deinit();

        std.debug.print("Num nodes: {}\n", .{manager.numNodes()});
        std.debug.print("Min spacing {}\n", .{manager.minSpacing()});

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

        // Runge Kutta 4 context
        var rk4 = try Rk4Integrator(Dynamic).init(allocator, manager.numNodes());
        defer rk4.deinit();

        @memset(rk4.sys.field(.u), 0.0);
        @memset(rk4.sys.field(.x), 0.0);
        @memset(rk4.sys.field(.w), 0.0);

        // *****************************
        // Initial Data

        std.debug.print("Solving Initial Data\n", .{});

        // Seed
        manager.project(Seed{ .amplitude = 1.0, .sigma = 1.0 }, rk4.sys.field(.seed));
        manager.fillGhostNodes(O, seed_boundary, rk4.sys.field(.seed));

        // Conformal factor
        const initial_rhs: InitialPsiRhs = .{
            .seed = rk4.sys.field(.seed),
        };

        const initial_op: InitialPsiOp = .{
            .seed = rk4.sys.field(.seed),
        };

        manager.project(initial_rhs, rhs);

        var solver = try MultigridMethod.init(allocator, manager.numNodes(), BiCGStabSolver.new(20000, 10e-14), .{
            .max_iters = 100,
            .tolerance = 10e-10,
            .presmooth = 5,
            .postsmooth = 5,
        });
        defer solver.deinit();

        @memset(rk4.sys.field(.psi), 0.0);

        try solver.solve(
            &manager,
            initial_op,
            psi_boundary,
            rk4.sys.field(.psi),
            rhs,
        );

        std.debug.print("Running Evolution\n", .{});

        // ******************************
        // Step 0

        {
            const evolution: Evolution = .{
                .gpa = allocator,
                .manager = &manager,
                .lapse = lapse,
                .shiftr = shiftr,
                .shiftz = shiftz,
                .rhs = rhs,
            };

            try evolution.preprocess(rk4.sys);

            const gauge: Gauge = .{
                .gpa = allocator,
                .manager = &manager,
                .lapse = lapse,
                .shiftr = shiftr,
                .shiftz = shiftz,
                .rhs = rhs,
            };

            try gauge.solve(rk4.sys.toConst());

            const hamiltonain: Hamiltonian = .{
                .psi = rk4.sys.field(.psi),
                .seed = rk4.sys.field(.seed),
                .u = rk4.sys.field(.u),
                .x = rk4.sys.field(.x),
                .w = rk4.sys.field(.w),
            };

            @memset(constraint, 0.0);

            manager.project(hamiltonain, constraint);

            std.debug.print("Initial Constraint Violation {}\n", .{manager.normScaled(constraint)});
        }

        // Output Base Data
        {
            for (0..mesh.numLevels()) |level| {
                const file_name = try std.fmt.allocPrint(allocator, "output/evolution-base{}.vtu", .{level});
                defer allocator.free(file_name);

                const file = try std.fs.cwd().createFile(file_name, .{});
                defer file.close();

                const output = SystemConst(Output).view(manager.numNodes(), .{
                    .psi = rk4.sys.field(.psi),
                    .seed = rk4.sys.field(.seed),
                    .u = rk4.sys.field(.u),
                    .x = rk4.sys.field(.x),
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
                    &manager,
                    output,
                    .{ .ghost = true },
                    buffer.writer(),
                );

                try buffer.flush();
            }
        }

        // Output
        {
            const file = try std.fs.cwd().createFile("output/evolution0.vtu", .{});
            defer file.close();

            const output = SystemConst(Output).view(manager.numNodes(), .{
                .psi = rk4.sys.field(.psi),
                .seed = rk4.sys.field(.seed),
                .u = rk4.sys.field(.u),
                .x = rk4.sys.field(.x),
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
                &manager,
                output,
                .{ .ghost = true },
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
            const con = manager.normScaled(constraint);
            std.debug.print("Step {}/{}, Time: {}, Constraint: {}\n", .{ step + 1, steps, rk4.time, con });

            // ********************************
            // Step

            const evolution: Evolution = .{
                .gpa = scratch,
                .manager = &manager,
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
                .manager = &manager,
                .lapse = lapse,
                .shiftr = shiftr,
                .shiftz = shiftz,
                .rhs = rhs,
            };

            try gauge.solve(rk4.sys.toConst());

            _ = arena.reset(.retain_capacity);

            // *******************************
            // Constraint

            const hamiltonian: Hamiltonian = .{
                .psi = rk4.sys.field(.psi),
                .seed = rk4.sys.field(.seed),
                .u = rk4.sys.field(.u),
                .x = rk4.sys.field(.x),
                .w = rk4.sys.field(.w),
            };

            manager.project(hamiltonian, constraint);

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
                .x = rk4.sys.field(.x),
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
                &manager,
                output,
                .{ .ghost = true },
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
