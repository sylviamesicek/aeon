// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const aeon = @import("aeon");

const bsamr = aeon.bsamr;
const geometry = aeon.geometry;
const nodes = aeon.nodes;

pub fn BrillEvolution(comptime M: usize) type {
    const N = 2;
    return struct {
        const BoundaryKind = nodes.BoundaryKind;
        const Robin = nodes.Robin;

        const DataOut = aeon.DataOut(N, M);
        const MultigridMethod = aeon.methods.MultigridMethod(N, M, BiCGStabSolver);
        const LinearMapMethod = aeon.methods.LinearMapMethod(N, M, BiCGStabSolver);
        const FEIntegrator = aeon.methods.ForwardEulerIntegrator;
        const Rk4Integrator = aeon.methods.RungeKutta4Integrator;
        const System = aeon.methods.System;
        const SystemConst = aeon.methods.SystemConst;

        const Mesh = bsamr.Mesh(N);
        const DofManager = bsamr.DofManager(N, M);
        const RegridManager = bsamr.RegridManager(N);

        const FaceIndex = geometry.FaceIndex(N);
        const IndexSpace = geometry.IndexSpace(N);
        const IndexMixin = geometry.IndexMixin(N);

        const BiCGStabSolver = aeon.lac.BiCGStabSolver;

        const Engine = aeon.mesh.Engine(N, M);

        pub const EvenBoundary = struct {
            pub fn kind(face: FaceIndex) BoundaryKind {
                if (face.side == false and face.axis == 0) {
                    return .even;
                } else {
                    return .robin;
                }
            }

            pub fn robin(_: EvenBoundary, pos: [N]f64, _: FaceIndex) Robin {
                const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);

                return .{
                    .value = 1.0 / r,
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

            pub fn robin(_: OddBoundary, pos: [N]f64, _: FaceIndex) Robin {
                const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);

                return .{
                    .value = 1.0 / r,
                    .flux = 1.0,
                    .rhs = 0.0,
                };
            }
        };

        const lapse_boundary = EvenBoundary{};
        const shiftr_boundary = OddBoundary{};
        const shiftz_boundary = EvenBoundary{};
        const conformal_boundary = EvenBoundary{};
        const seed_boundary = OddBoundary{};
        const y_boundary = OddBoundary{};
        const u_boundary = EvenBoundary{};
        const x_boundary = OddBoundary{};
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

        pub const InitialConformalRhs = struct {
            seed: []const f64,

            pub fn project(self: InitialConformalRhs, engine: Engine) f64 {
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

        pub const InitialConformalOp = struct {
            seed: []const f64,

            pub fn apply(self: InitialConformalOp, engine: Engine, conformal: []const f64) f64 {
                const position: [N]f64 = engine.position();
                const r = position[0];

                const hessian: [N][N]f64 = engine.hessian(conformal);
                const grad: [N]f64 = engine.gradient(conformal);
                const val: f64 = engine.value(conformal);

                const shessian: [N][N]f64 = engine.hessian(self.seed);
                const sgrad: [N]f64 = engine.gradient(self.seed);

                const term1 = hessian[0][0] + grad[0] / r + hessian[1][1];
                const term2 = 1.0 / 4.0 * val * (r * shessian[0][0] + 2.0 * sgrad[0] + r * shessian[1][1]);

                return term1 + term2;
            }

            pub fn applyDiag(self: InitialConformalOp, engine: Engine) f64 {
                const position: [N]f64 = engine.position();
                const r = position[0];

                const hessian: [N][N]f64 = engine.hessianDiag();
                const grad: [N]f64 = engine.gradientDiag();
                const val: f64 = engine.valueDiag();

                const shessian: [N][N]f64 = engine.hessian(self.seed);
                const sgrad: [N]f64 = engine.gradient(self.seed);

                const term1 = hessian[0][0] + grad[0] / r + hessian[1][1];
                const term2 = 1.0 / 4.0 * val * (r * shessian[0][0] + 2.0 * sgrad[0] + r * shessian[1][1]);

                return term1 + term2;
            }
        };

        pub const LapseRhs = struct {
            conformal: []const f64,
            seed: []const f64,
            u: []const f64,
            y: []const f64,
            x: []const f64,

            pub fn project(self: LapseRhs, engine: Engine) f64 {
                const pos: [N]f64 = engine.position();
                const rho = pos[0];

                const conformal = engine.value(self.conformal) + 1.0;
                const seed = engine.value(self.seed);
                const u = engine.value(self.u);
                const y = engine.value(self.y);
                const x = engine.value(self.x);

                const scale = conformal * conformal * conformal * conformal * @exp(2.0 * rho * seed);

                const term1 = 2.0 / 3.0 * (u - 0.5 * rho * y) * (u - 0.5 * rho * y);
                const term2 = 1.0 / 2.0 * rho * rho * y * y;
                const term3 = 2.0 * x * x;

                return scale * (term1 + term2 + term3);
            }
        };

        pub const LapseOp = struct {
            conformal: []const f64,
            seed: []const f64,
            u: []const f64,
            y: []const f64,
            x: []const f64,

            pub fn apply(self: LapseOp, engine: Engine, lapse: []const f64) f64 {
                const pos = engine.position();
                const rho = pos[0];

                const hessian: [N][N]f64 = engine.hessian(lapse);
                const grad: [N]f64 = engine.gradient(lapse);
                const val: f64 = engine.value(lapse);

                const pval = engine.value(self.conformal) + 1.0;
                const pgrad = engine.gradient(self.conformal);

                const seed = engine.value(self.seed);
                const u = engine.value(self.u);
                const y = engine.value(self.y);
                const x = engine.value(self.x);

                const term1 = hessian[0][0] + grad[0] / rho + hessian[1][1];
                const term2 = 2.0 / pval * (pgrad[0] * grad[0] + pgrad[1] * grad[1]);

                const scale = -val * pval * pval * pval * pval * @exp(2.0 * rho * seed);

                const term3 = 2.0 / 3.0 * (u - 0.5 * rho * y) * (u - 0.5 * rho * y);
                const term4 = 1.0 / 2.0 * rho * rho * y * y;
                const term5 = 2.0 * x * x;

                return term1 + term2 + scale * (term3 + term4 + term5);
            }

            pub fn applyDiag(self: LapseOp, engine: Engine) f64 {
                const pos = engine.position();
                const rho = pos[0];

                const hessian: [N][N]f64 = engine.hessianDiag();
                const grad: [N]f64 = engine.gradientDiag();
                const val: f64 = engine.valueDiag();

                const pval = engine.value(self.conformal) + 1.0;
                const pgrad = engine.gradient(self.conformal);

                const seed = engine.value(self.seed);
                const u = engine.value(self.u);
                const y = engine.value(self.y);
                const x = engine.value(self.x);

                const term1 = hessian[0][0] + grad[0] / rho + hessian[1][1];
                const term2 = 2.0 / pval * (pgrad[0] * grad[0] + pgrad[1] * grad[1]);

                const scale = -val * pval * pval * pval * pval * @exp(2.0 * rho * seed);

                const term3 = 2.0 / 3.0 * (u - 0.5 * rho * y) * (u - 0.5 * rho * y);
                const term4 = 1.0 / 2.0 * rho * rho * y * y;
                const term5 = 2.0 * x * x;

                return term1 + term2 + scale * (term3 + term4 + term5);
            }
        };

        pub const ShiftRRhs = struct {
            lapse: []const f64,
            x: []const f64,
            u: []const f64,

            pub fn project(self: ShiftRRhs, engine: Engine) f64 {
                const lapse = engine.value(self.lapse) + 1.0;
                const lgrad = engine.gradient(self.lapse);

                const x = engine.value(self.x);
                const xgrad = engine.gradient(self.x);

                const u = engine.value(self.u);
                const ugrad = engine.gradient(self.u);

                const term1 = x * lgrad[1] + lapse * xgrad[1];
                const term2 = -u * lgrad[0] - lapse * ugrad[0];

                return term1 + term2;
            }
        };

        pub const ShiftZRhs = struct {
            lapse: []const f64,
            x: []const f64,
            u: []const f64,

            pub fn project(self: ShiftZRhs, engine: Engine) f64 {
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
            pub fn apply(_: ShiftOp, engine: Engine, shift: []const f64) f64 {
                return engine.laplacian(shift);
            }

            pub fn applyDiag(_: ShiftOp, engine: Engine) f64 {
                return engine.laplacianDiag();
            }
        };

        pub const Constraint = struct {
            conformal: []const f64,
            seed: []const f64,
            u: []const f64,
            y: []const f64,
            x: []const f64,

            pub fn project(self: Constraint, engine: Engine) f64 {
                const pos: [N]f64 = engine.position();
                const rho = pos[0];

                const chessian: [N][N]f64 = engine.hessian(self.conformal);
                const cgrad: [N]f64 = engine.gradient(self.conformal);
                const conformal: f64 = engine.value(self.conformal) + 1.0;

                const shessian: [N][N]f64 = engine.hessian(self.seed);
                const sgrad: [N]f64 = engine.gradient(self.seed);
                const seed: f64 = engine.value(self.seed);

                const u = engine.value(self.u);
                const y = engine.value(self.y);
                const x = engine.value(self.x);

                const term1 = chessian[0][0] + cgrad[0] / rho + chessian[1][1];
                const term2 = 1.0 / 4.0 * conformal * (rho * shessian[0][0] + 2.0 * sgrad[0] + rho * shessian[1][1]);

                const scale = conformal * conformal * conformal * conformal * conformal * @exp(2.0 * rho * seed);
                const term3 = 1.0 / 3.0 * (u - 0.5 * rho * y) * (u - 0.5 * rho * y);
                const term4 = 1.0 / 4.0 * rho * rho * y * y;
                const term5 = x * x;

                return term1 + term2 + scale * (term3 + term4 + term5);
            }
        };

        pub const ConformalEvolution = struct {
            lapse: []const f64,
            shiftr: []const f64,
            shiftz: []const f64,
            conformal: []const f64,
            u: []const f64,
            y: []const f64,

            pub fn project(self: ConformalEvolution, engine: Engine) f64 {
                const pos = engine.position();
                const rho = pos[0];

                const lapse = engine.value(self.lapse) + 1.0;
                const shiftr = engine.value(self.shiftr);
                const shiftz = engine.value(self.shiftz);

                const conformal = engine.value(self.conformal) + 1.0;
                const cgrad = engine.gradient(self.conformal);
                const u = engine.value(self.u);
                const y = engine.value(self.y);

                const term1 = shiftr * cgrad[0] + shiftz * cgrad[1];
                const term2 = 0.5 * conformal * shiftr / rho;
                const term3 = 1.0 / 6.0 * conformal * lapse * (u - 2.0 * rho * y);

                return term1 + term2 + term3;
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

                const seed = engine.value(self.seed);
                const sgrad = engine.gradient(self.seed);

                const lapse = engine.value(self.lapse) + 1.0;
                const shiftr = engine.value(self.shiftr);
                const shiftz = engine.value(self.shiftz);

                const shiftr_grad = engine.gradient(self.shiftr);

                const y = engine.value(self.y);

                const term1 = shiftr * sgrad[0] + shiftz * sgrad[1];
                const term2 = lapse * y + shiftr * seed / rho;
                const term3 = shiftr_grad[0] / rho - shiftr / (rho * rho);

                return term1 + term2 + term3;
            }
        };

        pub const YEvolution = struct {
            lapse: []const f64,
            shiftr: []const f64,
            shiftz: []const f64,
            conformal: []const f64,
            seed: []const f64,
            y: []const f64,
            x: []const f64,

            pub fn project(self: YEvolution, engine: Engine) f64 {
                const pos = engine.position();
                const rho = pos[0];
                const rho2 = rho * rho;

                const lapse = engine.value(self.lapse) + 1.0;
                const lapse_grad = engine.gradient(self.lapse);
                const lapse_hess = engine.hessian(self.lapse);

                const shiftr = engine.value(self.shiftr);
                const shiftz = engine.value(self.shiftz);
                const shiftr_grad = engine.gradient(self.shiftr);
                const shiftz_grad = engine.gradient(self.shiftz);

                const conformal = engine.value(self.conformal) + 1.0;
                const conformal_grad = engine.gradient(self.conformal);
                const conformal_hess = engine.hessian(self.conformal);

                const seed = engine.value(self.seed);
                const seed_grad = engine.gradient(self.seed);
                const seed_hess = engine.hessian(self.seed);

                const x = engine.value(self.x);

                const y = engine.value(self.y);
                const y_grad = engine.gradient(self.y);

                const term1 = shiftr * y_grad[0] + shiftz * y_grad[1] + shiftr * y / rho;
                const term2 = -x / rho * (shiftz_grad[0] - shiftr_grad[1]);

                const scale1 = @exp(-2.0 * rho * seed) / (conformal * conformal * conformal * conformal);
                const term3 = lapse_hess[0][0] / rho - lapse_grad[0] / rho2;
                const term4 = -lapse_grad[0] / rho * (rho * seed_grad[0] + seed + 4.0 * conformal_grad[0] / conformal);
                const term5 = lapse_grad[1] * seed_grad[1];

                const scale2 = lapse * scale1;
                const term6 = 2.0 / conformal * (conformal_hess[0][0] / rho - conformal_grad[0] / rho2);
                const term7 = seed_hess[0][0] + seed_hess[1][1] + seed_grad[0] / rho - seed / rho2;
                const term8 = -2.0 * conformal_grad[0] / (rho * conformal) * (rho * seed_grad[0] + seed + 3.0 * conformal_grad[0] / conformal);
                const term9 = 2.0 * conformal_grad[1] * seed_grad[1] / conformal;

                return term1 + term2 + scale1 * (term3 + term4 + term5) + scale2 * (term6 + term7 + term8 + term9);
            }
        };

        pub const UEvolution = struct {
            lapse: []const f64,
            shiftr: []const f64,
            shiftz: []const f64,
            conformal: []const f64,
            seed: []const f64,
            u: []const f64,
            x: []const f64,

            pub fn project(self: UEvolution, engine: Engine) f64 {
                const pos = engine.position();
                const rho = pos[0];

                const lapse = engine.value(self.lapse) + 1.0;
                const lapse_grad = engine.gradient(self.lapse);
                const lapse_hess = engine.hessian(self.lapse);

                const shiftr = engine.value(self.shiftr);
                const shiftz = engine.value(self.shiftz);
                const shiftr_grad = engine.gradient(self.shiftr);
                const shiftz_grad = engine.gradient(self.shiftz);

                const conformal = engine.value(self.conformal) + 1.0;
                const conformal_grad = engine.gradient(self.conformal);
                const conformal_hess = engine.hessian(self.conformal);

                const seed = engine.value(self.seed);
                const seed_grad = engine.gradient(self.seed);

                const x = engine.value(self.x);

                const u_grad = engine.gradient(self.u);

                const term1 = shiftr * u_grad[0] + shiftz * u_grad[1];
                const term2 = 2.0 * x * (shiftr_grad[1] - shiftz_grad[0]);

                const scale1 = @exp(-2.0 * rho * seed) / (conformal * conformal * conformal * conformal);
                const term3 = 2.0 * lapse_grad[1] * (2.0 * conformal_grad[1] / conformal + rho * seed_grad[1]);
                const term4 = -2.0 * lapse_grad[0] * (2.0 * conformal_grad[0] / conformal + rho * seed_grad[0] + seed);
                const term5 = lapse_hess[0][0] + lapse_hess[1][1];

                const scale2 = -2.0 * lapse * scale1;
                const term6 = -conformal_hess[0][0] / conformal + conformal_hess[1][1] / conformal + seed_grad[0] + seed / rho;
                const term7 = conformal_grad[0] / conformal * (3.0 * conformal_grad[0] / conformal + 2.0 * rho * seed_grad[0] + 2.0 * seed);
                const term8 = -conformal_grad[1] / conformal * (3.0 * conformal_grad[1] / conformal + 2.0 * rho * seed_grad[1]);

                return term1 + term2 + scale1 * (term3 + term4 + term5) + scale2 * (term6 + term7 + term8);
            }
        };

        pub const XEvolution = struct {
            lapse: []const f64,
            shiftr: []const f64,
            shiftz: []const f64,
            conformal: []const f64,
            seed: []const f64,
            u: []const f64,
            x: []const f64,

            pub fn project(self: XEvolution, engine: Engine) f64 {
                const pos = engine.position();
                const rho = pos[0];

                const lapse = engine.value(self.lapse) + 1.0;
                const lapse_grad = engine.gradient(self.lapse);
                const lapse_hess = engine.hessian(self.lapse);

                const shiftr = engine.value(self.shiftr);
                const shiftz = engine.value(self.shiftz);
                const shiftr_grad = engine.gradient(self.shiftr);
                const shiftz_grad = engine.gradient(self.shiftz);

                const conformal = engine.value(self.conformal) + 1.0;
                const conformal_grad = engine.gradient(self.conformal);
                const conformal_hess = engine.hessian(self.conformal);

                const seed = engine.value(self.seed);
                const seed_grad = engine.gradient(self.seed);

                const u = engine.value(self.u);

                const x_grad = engine.gradient(self.x);

                const term1 = shiftr * x_grad[0] + shiftz * x_grad[1];
                const term2 = 1.0 / 2.0 * u * (shiftz_grad[0] - shiftr_grad[1]);

                const scale1 = @exp(-2.0 * rho * seed) / (conformal * conformal * conformal * conformal);
                const term3 = -lapse_hess[0][1];
                const term4 = lapse_grad[0] * (rho * seed_grad[1] + 2.0 * conformal_grad[1] / conformal);
                const term5 = lapse_grad[1] * (rho * seed_grad[0] + 2.0 * conformal_grad[0] / conformal + seed);

                const scale2 = lapse * scale1;
                const term6 = -2.0 * conformal_hess[0][1] / conformal;
                const term7 = conformal_grad[0] / conformal * (3.0 * conformal_grad[1] / conformal + 2.0 * rho * seed_grad[1]);
                const term8 = 2.0 * conformal_grad[1] / conformal * (rho * seed_grad[0] + seed) + seed_grad[1];

                return term1 + term2 + scale1 * (term3 + term4 + term5) + scale2 * (term6 + term7 + term8);
            }
        };

        pub const Dynamic = enum {
            conformal,
            seed,
            u,
            x,
            y,
        };

        const Output = enum {
            conformal,
            seed,
            u,
            y,
            x,
            constraint,
        };

        pub const Evolution = struct {
            mesh: *const Mesh,
            dofs: *const DofManager,

            gpa: Allocator,

            // Gauge variables (all node vectors)

            lapse: []f64,
            shiftr: []f64,
            shiftz: []f64,

            // Scratch cells variables
            sol: []f64,
            rhs: []f64,

            pub const Tag = Dynamic;

            pub fn derivative(self: Evolution, deriv: System(Tag), dynamic: System(Tag), time: f64) void {
                _ = time;

                const mesh: *const Mesh = self.mesh;
                const dofs: *const DofManager = self.dofs;

                const lapse = self.lapse;
                const shiftr = self.shiftr;
                const shiftz = self.shiftz;

                const sol = self.sol;
                const rhs = self.rhs;

                const conformal = dynamic.field(.conformal);
                const seed = dynamic.field(.seed);
                const u = dynamic.field(.u);
                const y = dynamic.field(.y);
                const x = dynamic.field(.x);

                // ***************************************
                // Fill Boundaries and restrict

                for (0..mesh.levels.len) |rev_level_id| {
                    const level_id = mesh.levels.len - 1 - rev_level_id;

                    dofs.fillLevelBoundary(mesh, level_id, conformal_boundary, conformal);
                    dofs.fillLevelBoundary(mesh, level_id, seed_boundary, seed);
                    dofs.fillLevelBoundary(mesh, level_id, u_boundary, u);
                    dofs.fillLevelBoundary(mesh, level_id, y_boundary, y);
                    dofs.fillLevelBoundary(mesh, level_id, x_boundary, x);

                    dofs.restrictLevel(mesh, level_id, conformal);
                    dofs.restrictLevel(mesh, level_id, seed);
                    dofs.restrictLevel(mesh, level_id, u);
                    dofs.restrictLevel(mesh, level_id, y);
                    dofs.restrictLevel(mesh, level_id, x);
                }

                // ***********************************
                // Solve Elliptic Gauges Conditions

                const solver: MultigridMethod = .{
                    .base_solver = BiCGStabSolver.new(20000, 10e-14),
                    .max_iters = 100,
                    .tolerance = 10e-10,
                    .presmooth = 5,
                    .postsmooth = 5,
                };

                // ***********************************
                // Solve Lapse

                const lapse_rhs: LapseRhs = .{
                    .conformal = conformal,
                    .seed = seed,
                    .u = u,
                    .x = x,
                    .y = y,
                };

                const lapse_op: LapseOp = .{
                    .conformal = conformal,
                    .seed = seed,
                    .u = u,
                    .x = x,
                    .y = y,
                };

                dofs.project(mesh, lapse_rhs, rhs);

                @memset(sol, 0.0);

                solver.solve(
                    self.gpa,
                    mesh,
                    dofs,
                    lapse_op,
                    lapse_boundary,
                    sol,
                    rhs,
                ) catch {
                    unreachable;
                };

                dofs.transfer(mesh, lapse_boundary, lapse, sol);

                // ***********************************
                // Solve Shift

                const shift_op: ShiftOp = .{};

                const shiftr_rhs: ShiftRRhs = .{
                    .lapse = lapse,
                    .u = u,
                    .x = x,
                };

                const shiftz_rhs: ShiftZRhs = .{
                    .lapse = lapse,
                    .u = u,
                    .x = x,
                };

                // ShiftR

                dofs.project(mesh, shiftr_rhs, rhs);

                @memset(sol, 0.0);

                solver.solve(
                    self.gpa,
                    mesh,
                    dofs,
                    shift_op,
                    shiftr_boundary,
                    sol,
                    rhs,
                ) catch {
                    unreachable;
                };

                dofs.transfer(mesh, shiftr_boundary, shiftr, sol);

                // ShiftZ

                dofs.project(mesh, shiftz_rhs, rhs);

                @memset(sol, 0.0);

                solver.solve(
                    self.gpa,
                    mesh,
                    dofs,
                    shift_op,
                    shiftz_boundary,
                    sol,
                    rhs,
                ) catch {
                    unreachable;
                };

                dofs.transfer(mesh, shiftz_boundary, shiftz, sol);

                // ************************************
                // Evolve conformal

                const conformal_evolve: ConformalEvolution = .{
                    .lapse = lapse,
                    .shiftr = shiftr,
                    .shiftz = shiftz,
                    .conformal = conformal,
                    .u = u,
                    .y = y,
                };

                dofs.project(mesh, conformal_evolve, sol);
                dofs.transfer(mesh, conformal_boundary, deriv.field(.conformal), sol);

                // ************************************
                // Evolve seed

                const seed_evolve: SeedEvolution = .{
                    .lapse = lapse,
                    .shiftr = shiftr,
                    .shiftz = shiftz,
                    .seed = seed,
                    .y = y,
                };

                dofs.project(mesh, seed_evolve, sol);
                dofs.transfer(mesh, seed_boundary, deriv.field(.seed), sol);

                // ************************************
                // Evolve Y

                const y_evolve: YEvolution = .{
                    .lapse = lapse,
                    .shiftr = shiftr,
                    .shiftz = shiftz,
                    .conformal = conformal,
                    .seed = seed,
                    .y = y,
                    .x = x,
                };

                dofs.project(mesh, y_evolve, sol);
                dofs.transfer(mesh, y_boundary, deriv.field(.y), sol);

                // ***********************************
                // Evolve U

                const u_evolve: UEvolution = .{
                    .lapse = lapse,
                    .shiftr = shiftr,
                    .shiftz = shiftz,
                    .conformal = conformal,
                    .seed = seed,
                    .u = u,
                    .x = x,
                };

                dofs.project(mesh, u_evolve, sol);
                dofs.transfer(mesh, u_boundary, deriv.field(.u), sol);

                // ************************************
                // Evolve X

                const x_evolve: XEvolution = .{
                    .lapse = lapse,
                    .shiftr = shiftr,
                    .shiftz = shiftz,
                    .conformal = conformal,
                    .seed = seed,
                    .u = u,
                    .x = x,
                };

                dofs.project(mesh, x_evolve, sol);
                dofs.transfer(mesh, x_boundary, deriv.field(.x), sol);
            }
        };

        // Run

        fn run(allocator: Allocator) !void {
            std.debug.print("Running Poisson Elliptic Solver\n", .{});

            var mesh = try Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, -10.0 },
                    .size = [2]f64{ 20.0, 20.0 },
                },
                .tile_width = 8,
                .index_size = [2]usize{ 1, 1 },
            });
            defer mesh.deinit();

            var dofs = DofManager.init(allocator);
            defer dofs.deinit();

            try dofs.build(&mesh);

            // Globally refine

            for (0..4) |_| {
                const amr: RegridManager = .{
                    .max_levels = 16,
                    .patch_efficiency = 0.1,
                    .block_efficiency = 0.7,
                };

                var tags = try allocator.alloc(bool, dofs.numTiles());
                defer allocator.free(tags);

                @memset(tags, true);

                try amr.regrid(allocator, tags, &mesh, dofs.tile_map);
                try dofs.build(&mesh);
            }

            std.debug.print("NDofs: {}\n", .{dofs.numNodes()});
            std.debug.print("Allocating Memory\n", .{});

            // Cell Vectors

            const sol = try allocator.alloc(f64, dofs.numCells());
            defer allocator.free(sol);

            const rhs = try allocator.alloc(f64, dofs.numCells());
            defer allocator.free(rhs);

            // Gauge Vectors

            const lapse = try allocator.alloc(f64, dofs.numNodes());
            defer allocator.free(lapse);

            const shiftr = try allocator.alloc(f64, dofs.numNodes());
            defer allocator.free(shiftr);

            const shiftz = try allocator.alloc(f64, dofs.numNodes());
            defer allocator.free(shiftz);

            const constraint = try allocator.alloc(f64, dofs.numNodes());
            defer allocator.free(constraint);

            @memset(lapse, 0.0);
            @memset(shiftr, 0.0);
            @memset(shiftz, 0.0);

            // Runge Kutta 4 context

            var rk4 = try Rk4Integrator(Dynamic).init(allocator, dofs.numNodes());
            defer rk4.deinit();

            // *****************************
            // Initial Data

            std.debug.print("Solving Initial Data\n", .{});

            @memset(rk4.sys.field(.u), 0.0);
            @memset(rk4.sys.field(.y), 0.0);
            @memset(rk4.sys.field(.x), 0.0);

            // Seed

            dofs.project(&mesh, Seed{
                .amplitude = 1.0,
                .sigma = 1.0,
            }, sol);
            dofs.transfer(&mesh, seed_boundary, rk4.sys.field(.seed), sol);

            // Conformal factor

            const initial_rhs: InitialConformalRhs = .{
                .seed = rk4.sys.field(.seed),
            };

            dofs.project(&mesh, initial_rhs, rhs);

            const initial_op: InitialConformalOp = .{
                .seed = rk4.sys.field(.seed),
            };

            const solver: MultigridMethod = .{
                .base_solver = BiCGStabSolver.new(20000, 10e-14),
                .max_iters = 100,
                .tolerance = 10e-10,
                .presmooth = 5,
                .postsmooth = 5,
            };

            try solver.solve(
                allocator,
                &mesh,
                &dofs,
                initial_op,
                conformal_boundary,
                sol,
                rhs,
            );

            dofs.transfer(&mesh, conformal_boundary, rk4.sys.field(.conformal), sol);

            // ******************************
            // Step 0

            {
                const constraint_proj: Constraint = .{
                    .conformal = rk4.sys.field(.conformal),
                    .seed = rk4.sys.field(.seed),
                    .u = rk4.sys.field(.u),
                    .y = rk4.sys.field(.y),
                    .x = rk4.sys.field(.x),
                };

                dofs.project(&mesh, constraint_proj, sol);
                dofs.transfer(&mesh, constraint_boundary, constraint, sol);
            }

            // Output
            {
                const file = try std.fs.cwd().createFile("output/evolution0.vtu", .{});
                defer file.close();

                const output = SystemConst(Output).view(dofs.numNodes(), .{
                    .conformal = rk4.sys.field(.conformal),
                    .seed = rk4.sys.field(.seed),
                    .u = rk4.sys.field(.u),
                    .y = rk4.sys.field(.y),
                    .x = rk4.sys.field(.x),
                    .constraint = constraint,
                });

                var buffer = std.io.bufferedWriter(file.writer());

                try DataOut.writeVtkNodes(
                    Output,
                    allocator,
                    &mesh,
                    &dofs,
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

            const steps: usize = 10;
            const h: f64 = 0.01;

            for (0..steps) |step| {
                // Reset allocations
                defer _ = arena.reset(.retain_capacity);

                std.debug.print("Step {}/{}\n", .{ step + 1, steps });

                // ********************************
                // Step

                const evolution: Evolution = .{
                    .mesh = &mesh,
                    .dofs = &dofs,
                    .gpa = scratch,
                    .lapse = lapse,
                    .shiftr = shiftr,
                    .shiftz = shiftz,
                    .sol = sol,
                    .rhs = rhs,
                };

                rk4.step(evolution, h);

                // *******************************
                // Constraint

                const constraint_proj: Constraint = .{
                    .conformal = rk4.sys.field(.conformal),
                    .seed = rk4.sys.field(.seed),
                    .u = rk4.sys.field(.u),
                    .y = rk4.sys.field(.y),
                    .x = rk4.sys.field(.x),
                };

                dofs.project(&mesh, constraint_proj, sol);
                dofs.transfer(&mesh, constraint_boundary, constraint, sol);

                // *******************************
                // Output

                const file_name = try std.fmt.allocPrint(allocator, "output/evolution{}.vtu", .{step + 1});
                defer allocator.free(file_name);

                const file = try std.fs.cwd().createFile(file_name, .{});
                defer file.close();

                const output = SystemConst(Output).view(dofs.numNodes(), .{
                    .conformal = rk4.sys.field(.conformal),
                    .seed = rk4.sys.field(.seed),
                    .u = rk4.sys.field(.u),
                    .y = rk4.sys.field(.y),
                    .x = rk4.sys.field(.x),
                    .constraint = constraint,
                });

                var buffer = std.io.bufferedWriter(file.writer());

                try DataOut.writeVtkNodes(
                    Output,
                    allocator,
                    &mesh,
                    &dofs,
                    output,
                    buffer.writer(),
                );

                try buffer.flush();
            }
        }
    };
}

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
    try BrillEvolution(2).run(gpa.allocator());
}
