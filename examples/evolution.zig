// Imports
const std = @import("std");
const Allocator = std.mem.Allocator;

const aeon = @import("aeon");
const dofs = aeon.dofs;
const geometry = aeon.geometry;
const index = aeon.index;
const lac = aeon.lac;
const methods = aeon.methods;

pub fn BrillEvolution(comptime O: usize) type {
    const N = 2;
    return struct {
        const BoundaryCondition = dofs.BoundaryCondition;
        const DataOut = aeon.DataOut(N);
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const DofUtilsTotal = dofs.DofUtilsTotal(N, O);
        const LinearMapMethod = methods.LinearMapMethod(N, O, BiCGStabSolver);
        const Rk4 = methods.RungeKutta4Integrator(Hyperbolic);
        const SystemSlice = aeon.SystemSlice;
        const SystemSliceConst = aeon.SystemSliceConst;
        const SystemValue = aeon.SystemValue;
        const SystemBoundaryCondition = dofs.SystemBoundaryCondition;

        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);

        const BiCGStabSolver = lac.BiCGStabSolver;

        const Mesh = aeon.mesh.Mesh(N);

        fn evenBoundaryCondition(pos: [N]f64, face: Face) BoundaryCondition {
            if (face.side == false and face.axis == 0) {
                return BoundaryCondition.nuemann(0.0);
            } else {
                const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                return BoundaryCondition.robin(1.0 / r, 1.0, 0.0);
            }
        }

        fn oddBoundaryCondition(pos: [N]f64, face: Face) BoundaryCondition {
            if (face.side == false and face.axis == 0) {
                return BoundaryCondition.diritchlet(0.0);
            } else {
                const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                return BoundaryCondition.robin(1.0 / r, 1.0, 0.0);
            }
        }

        pub const Hyperbolic = enum {
            psi,
            s,
            u,
            y,
            x,
        };

        pub const Elliptic = enum {
            lapse,
            shift_r,
            shift_z,
        };

        pub const SProjection = struct {
            amplitude: f64,
            sigma: f64,

            pub const System = enum {
                s,
            };
            pub const Context = aeon.EmptySystem;

            pub fn project(self: SProjection, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const pos = engine.position();

                const rho = pos[0];
                const z = pos[1];

                const rho2 = rho * rho;
                const z2 = z * z;
                const sigma2 = self.sigma * self.sigma;

                return .{
                    .s = self.amplitude * rho * @exp(-(rho2 + z2) / sigma2),
                };
            }

            pub fn boundaryCtx(_: SProjection, _: [N]f64, _: Face) SystemBoundaryCondition(Context) {
                return .{};
            }
        };

        pub const InitialDataRhs = struct {
            pub const System = InitialDataOperator.System;
            pub const Context = InitialDataOperator.Context;

            pub fn project(_: InitialDataRhs, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const s_hessian: [N][N]f64 = engine.hessianCtx(.s);
                const s_lap = s_hessian[0][0] + s_hessian[1][1];

                return .{
                    .psi = s_lap,
                };
            }
        };

        pub const InitialDataOperator = struct {
            pub const System = enum {
                psi,
            };

            pub const Context = SProjection.System;

            pub fn apply(_: InitialDataOperator, comptime Setting: dofs.EngineSetting, engine: dofs.Engine(N, O, Setting, System, Context)) SystemValue(System) {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianSys(.psi);
                const gradient: [N]f64 = engine.gradientSys(.psi);
                const value: f64 = engine.valueSys(.psi);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                const s_hessian: [N][N]f64 = engine.hessianCtx(.s);
                const s_lap = s_hessian[0][0] + s_hessian[1][1];

                return .{
                    .psi = -lap - s_lap * value,
                };
            }

            pub fn boundarySys(_: InitialDataOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                return .{
                    .psi = evenBoundaryCondition(pos, face),
                };
            }

            pub fn boundaryCtx(_: InitialDataOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(Context) {
                return .{
                    .s = oddBoundaryCondition(pos, face),
                };
            }
        };

        pub const LapseRhs = struct {
            pub const System = LapseOperator.System;
            pub const Context = LapseOperator.Context;

            pub fn project(_: LapseRhs, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const position: [N]f64 = engine.position();
                const r = position[0];

                const psi_val = engine.valueCtx(.psi) + 1.0;

                const s = engine.valueCtx(.s);
                const u = engine.valueCtx(.u);
                const y = engine.valueCtx(.y);
                const x = engine.valueCtx(.x);

                const psi = psi_val;

                const scale = psi * psi * psi * psi * @exp(2 * r * s);

                const term3 = 2.0 / 3.0 * (u - 0.5 * r * y) * (u - 0.5 * r * y);
                const term4 = 0.5 * r * r * y * y;
                const term5 = 2.0 * x * x;

                return .{
                    .lapse = scale * (term3 + term4 + term5),
                };
            }

            pub fn boundaryCtx(_: LapseRhs, pos: [N]f64, face: Face) SystemBoundaryCondition(Context) {
                return .{
                    .psi = evenBoundaryCondition(pos, face),
                    .s = oddBoundaryCondition(pos, face),
                    .u = evenBoundaryCondition(pos, face),
                    .x = oddBoundaryCondition(pos, face),
                    .y = oddBoundaryCondition(pos, face),
                };
            }
        };

        pub const LapseOperator = struct {
            pub const Context = enum {
                psi,
                s,
                u,
                y,
                x,
            };

            pub const System = enum {
                lapse,
            };

            pub fn apply(
                _: LapseOperator,
                comptime Setting: dofs.EngineSetting,
                engine: dofs.Engine(N, O, Setting, System, Context),
            ) SystemValue(System) {
                const position: [N]f64 = engine.position();
                const r = position[0];

                const hessian: [N][N]f64 = engine.hessianSys(.lapse);
                const gradient: [N]f64 = engine.gradientSys(.lapse);
                const value: f64 = engine.valueSys(.lapse);

                const lapse = value;
                const lapse_r = gradient[0];
                const lapse_z = gradient[1];
                const lapse_rr = hessian[0][0];
                const lapse_zz = hessian[1][1];

                const psi_val = engine.valueCtx(.psi) + 1.0;
                const psi_grad = engine.gradientCtx(.psi);

                const s = engine.valueCtx(.s);
                const u = engine.valueCtx(.u);
                const y = engine.valueCtx(.y);
                const x = engine.valueCtx(.x);

                const psi = psi_val;
                const psi_r = psi_grad[0];
                const psi_z = psi_grad[1];

                const term1 = lapse_rr + lapse_r / r + lapse_zz;
                const term2 = 2.0 / psi * (psi_r * lapse_r + psi_z * lapse_z);

                const scale = -lapse * psi * psi * psi * psi * @exp(2 * r * s);

                const term3 = 2.0 / 3.0 * (u - 0.5 * r * y) * (u - 0.5 * r * y);
                const term4 = 0.5 * r * r * y * y;
                const term5 = 2.0 * x * x;

                return .{
                    .lapse = term1 + term2 + scale * (term3 + term4 + term5),
                };
            }

            pub fn boundarySys(_: LapseOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                return .{
                    .lapse = evenBoundaryCondition(pos, face),
                };
            }

            pub fn boundaryCtx(_: LapseOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(Context) {
                return .{
                    .psi = evenBoundaryCondition(pos, face),
                    .s = oddBoundaryCondition(pos, face),
                    .u = evenBoundaryCondition(pos, face),
                    .x = oddBoundaryCondition(pos, face),
                    .y = oddBoundaryCondition(pos, face),
                };
            }
        };

        pub const ShiftRRhs = struct {
            pub const System = ShiftROperator.System;
            pub const Context = enum {
                lapse,
                u,
                x,
            };

            pub fn project(_: ShiftRRhs, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const lapse_val = engine.valueCtx(.lapse) + 1.0;
                const lapse_grad = engine.gradientCtx(.lapse);

                const x_val = engine.valueCtx(.x);
                const x_grad = engine.gradientCtx(.x);

                const u_val = engine.valueCtx(.u);
                const u_grad = engine.gradientCtx(.u);

                return .{
                    .shift_r = 2.0 * lapse_val * x_grad[1] + 2.0 * lapse_grad[1] * x_val - lapse_val * u_grad[0] - lapse_grad[0] * u_val,
                };
            }
        };

        pub const ShiftROperator = struct {
            pub const System = enum {
                shift_r,
            };

            pub const Context = aeon.EmptySystem;

            pub fn apply(
                _: ShiftROperator,
                comptime Setting: dofs.EngineSetting,
                engine: dofs.Engine(N, O, Setting, System, Context),
            ) SystemValue(System) {
                const hessian: [N][N]f64 = engine.hessianSys(.shift_r);

                return .{
                    .shift_r = hessian[0][0] + hessian[1][1],
                };
            }

            pub fn boundarySys(_: ShiftROperator, pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                return .{
                    .shift_r = oddBoundaryCondition(pos, face),
                };
            }

            pub fn boundaryCtx(_: ShiftROperator, _: [N]f64, _: Face) SystemBoundaryCondition(Context) {
                return .{};
            }
        };

        pub const ShiftZRhs = struct {
            pub const System = ShiftZOperator.System;
            pub const Context = enum {
                lapse,
                u,
                x,
            };

            pub fn project(_: ShiftZRhs, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const lapse_val = engine.valueCtx(.lapse) + 1.0;
                const lapse_grad = engine.gradientCtx(.lapse);

                const x_val = engine.valueCtx(.x);
                const x_grad = engine.gradientCtx(.x);

                const u_val = engine.valueCtx(.u);
                const u_grad = engine.gradientCtx(.u);

                return .{
                    .shift_z = 2.0 * lapse_val * x_grad[1] + 2.0 * lapse_grad[1] * x_val - lapse_val * u_grad[0] - lapse_grad[0] * u_val,
                };
            }
        };

        pub const ShiftZOperator = struct {
            pub const System = enum {
                shift_z,
            };

            pub const Context = aeon.EmptySystem;

            pub fn apply(
                _: ShiftZOperator,
                comptime Setting: dofs.EngineSetting,
                engine: dofs.Engine(N, O, Setting, System, Context),
            ) SystemValue(System) {
                const hessian: [N][N]f64 = engine.hessianSys(.shift_z);

                return .{
                    .shift_z = hessian[0][0] + hessian[1][1],
                };
            }

            pub fn boundarySys(_: ShiftZOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                return .{
                    .shift_z = evenBoundaryCondition(pos, face),
                };
            }

            pub fn boundaryCtx(_: ShiftZOperator, _: [N]f64, _: Face) SystemBoundaryCondition(Context) {
                return .{};
            }
        };

        pub const EllipticBoundary = struct {
            pub const System = Elliptic;

            pub fn boundary(_: @This(), pos: [N]f64, face: Face) SystemBoundaryCondition(Elliptic) {
                return .{
                    .lapse = evenBoundaryCondition(pos, face),
                    .shift_r = oddBoundaryCondition(pos, face),
                    .shift_z = evenBoundaryCondition(pos, face),
                };
            }
        };

        pub const HyperbolicBoundary = struct {
            pub const System = Hyperbolic;

            pub fn boundary(_: @This(), pos: [N]f64, face: Face) SystemBoundaryCondition(Hyperbolic) {
                return .{
                    .psi = evenBoundaryCondition(pos, face),
                    .s = oddBoundaryCondition(pos, face),
                    .u = evenBoundaryCondition(pos, face),
                    .y = oddBoundaryCondition(pos, face),
                    .x = oddBoundaryCondition(pos, face),
                };
            }
        };

        pub const DerivativeOperator = struct {
            pub const System = Hyperbolic;
            pub const Context = Elliptic;

            pub fn apply(
                _: DerivativeOperator,
                comptime Setting: dofs.EngineSetting,
                engine: dofs.Engine(N, O, Setting, System, Context),
            ) SystemValue(System) {
                const pos = engine.position();
                const r = pos[0];
                const z = pos[1];
                _ = z;

                const r2 = r * r;

                const psi_val = engine.valueSys(.psi);
                const psi_grad = engine.gradientSys(.psi);
                const psi_hessian = engine.hessianSys(.psi);

                const psi = psi_val + 1.0;
                const psi_r = psi_grad[0];
                const psi_z = psi_grad[1];
                const psi_rr = psi_hessian[0][0];
                const psi_zz = psi_hessian[1][1];
                const psi_rz = psi_hessian[0][1];

                const psi2 = psi * psi;
                const psi4 = psi2 * psi2;

                const s_val = engine.valueSys(.s);
                const s_grad = engine.gradientSys(.s);
                const s_hessian = engine.hessianSys(.s);

                const s = s_val;
                const s_r = s_grad[0];
                const s_z = s_grad[1];
                const s_rr = s_hessian[0][0];
                const s_zz = s_hessian[1][1];

                const u_val = engine.valueSys(.u);
                const u_grad = engine.gradientSys(.u);

                const u = u_val;
                const u_r = u_grad[0];
                const u_z = u_grad[1];

                const y_val = engine.valueSys(.y);
                const y_grad = engine.gradientSys(.y);

                const y = y_val;
                const y_r = y_grad[0];
                const y_z = y_grad[1];

                const x_val = engine.valueSys(.x);
                const x_grad = engine.gradientSys(.x);

                const x = x_val;
                const x_r = x_grad[0];
                const x_z = x_grad[1];

                const lapse_val = engine.valueCtx(.lapse);
                const lapse_grad = engine.gradientCtx(.lapse);
                const lapse_hessian = engine.hessianCtx(.lapse);

                const lapse = lapse_val + 1.0;
                const lapse_r = lapse_grad[0];
                const lapse_z = lapse_grad[1];
                const lapse_rr = lapse_hessian[0][0];
                const lapse_zz = lapse_hessian[1][1];
                const lapse_rz = lapse_hessian[0][1];

                const shift_r_val = engine.valueCtx(.shift_r);
                const shift_r_grad = engine.gradientCtx(.shift_r);

                const shift_r = shift_r_val;
                const shift_r_r = shift_r_grad[0];
                const shift_r_z = shift_r_grad[1];

                const shift_z_val = engine.valueCtx(.shift_z);
                const shift_z_grad = engine.gradientCtx(.shift_z);

                const shift_z = shift_z_val;
                const shift_z_r = shift_z_grad[0];
                const shift_z_z = shift_z_grad[1];
                _ = shift_z_z;

                const conformal = @exp(-2.0 * r * s) / psi4;

                const psi_deriv = blk: {
                    // TODO Check this.
                    const term1 = shift_r * psi_r + shift_z * psi_z;
                    const term2 = psi * shift_r / (2.0 * r);
                    const term3 = lapse * (u - 2 * r * y) / 6.0;

                    break :blk term1 + term2 + term3;
                };

                const s_deriv = blk: {
                    const term1 = shift_r * s_r + shift_z * s_z;
                    const term2 = lapse * y + shift_r_r / r - shift_r / r2 + s * shift_r / r;

                    break :blk term1 + term2;
                };

                const y_deriv = blk: {
                    const term1 = shift_r * y_r + shift_z * y_z + y * shift_r / r;
                    const term2 = -x * (shift_z_r - shift_r_z) / r;
                    const term3 = conformal * (lapse_rr / r - lapse_r / r2 - lapse_r / r * (r * s_r + s + 4.0 * psi_r / psi) + lapse_z * s_z);
                    const term4 = lapse * conformal * (2.0 / psi * (psi_rr / r - psi_r / r2) + s_rr + s_zz + s_r / r - s / r2 - 2.0 * psi_r * (r * s_r + s + 3.0 * psi_r / psi) / (r * psi) + 2 * psi_z * s_z / psi);
                    break :blk term1 + term2 + term3 + term4;
                };

                const u_deriv = blk: {
                    const term1 = shift_r * u_r + shift_z * u_z + 2.0 * x * (shift_r_z - shift_z_r);
                    const term2 = conformal * (2.0 * lapse_z * (2.0 * psi_z / psi + r * s_z) - 2.0 * lapse_r * (2 * psi_r / psi + r * s_r + s) + lapse_rr + lapse_zz);
                    const term3 = -2.0 * lapse * conformal * (-psi_rr / psi + psi_zz / psi + s_r + s / r + psi_r / psi * (3.0 * psi_r / psi + 2.0 * r * s_r + 2.0 * s) - psi_z / psi * (3.0 * psi_z / psi + 2 * r * s_z));
                    break :blk term1 + term2 + term3;
                };

                const x_deriv = blk: {
                    const term1 = shift_r * x_r + shift_z * x_z + 0.5 * u * (shift_z_r - shift_r_z);
                    const term2 = conformal * (-lapse_rz + lapse_r * (r * s_z + 2.0 * psi_z / psi) + lapse_z * (r * s_r + s + 2.0 * psi_r / psi));
                    const term3 = lapse * conformal * (-2.0 * psi_rz / psi + psi_r / psi * (3.0 * psi_z / psi + 2 * r * s_z) + 2.0 * psi_z / psi * (r * s_r + s) + s_z);
                    break :blk term1 + term2 + term3;
                };

                return .{
                    .psi = psi_deriv,
                    .s = s_deriv,
                    .y = y_deriv,
                    .u = u_deriv,
                    .x = x_deriv,
                };
            }

            pub fn boundarySys(_: DerivativeOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                return (HyperbolicBoundary{}).boundary(pos, face);
            }

            pub fn boundaryCtx(_: DerivativeOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(Context) {
                return (EllipticBoundary{}).boundary(pos, face);
            }
        };

        pub const Derivative = struct {
            mesh: *const Mesh,
            dof_map: DofMap,
            allocator: Allocator,
            solver: LinearMapMethod,

            dynamic: SystemSlice(Hyperbolic),
            gauge: SystemSlice(Elliptic),

            gauge_cells: SystemSlice(Elliptic),
            gause_rhs: SystemSlice(Elliptic),

            pub const System = Hyperbolic;

            pub fn derivative(self: @This(), deriv: SystemSlice(Hyperbolic), src: SystemSliceConst(Hyperbolic), time: f64) void {
                _ = time;

                // Copy to hyperbolic dof vectors

                for (0..self.mesh.blocks.len) |block_id| {
                    DofUtils.copyDofsFromCells(
                        Hyperbolic,
                        self.mesh,
                        self.dof_map,
                        block_id,
                        self.dynamic,
                        src,
                    );
                }

                for (0..self.mesh.blocks.len) |block_id| {
                    DofUtils.fillBoundary(
                        self.mesh,
                        self.dof_map,
                        block_id,
                        HyperbolicBoundary{},
                        self.dynamic,
                    );
                }

                // Set initial guess for elliptic cells

                @memset(self.gauge_cells.field(.lapse), 1.0);
                @memset(self.gauge_cells.field(.shift_r), 0.0);
                @memset(self.gauge_cells.field(.shift_z), 0.0);

                // Solve Lapse RHS

                const lapse_rhs = SystemSlice(LapseRhs.System).view(self.mesh.cell_total, .{ .lapse = self.gause_rhs.field(.lapse) });
                const lapse_ctx = SystemSliceConst(LapseRhs.Context).view(self.dof_map.ndofs(), .{
                    .psi = self.dynamic.field(.psi),
                    .s = self.dynamic.field(.s),
                    .u = self.dynamic.field(.u),
                    .x = self.dynamic.field(.x),
                    .y = self.dynamic.field(.y),
                });

                DofUtilsTotal.project(
                    self.mesh,
                    self.dof_map,
                    LapseRhs{},
                    lapse_rhs,
                    lapse_ctx,
                );

                // Solve Lapse

                const lapse_view = SystemSlice(LapseRhs.System).view(self.mesh.cell_total, .{ .lapse = self.gauge_cells.field(.lapse) });

                const lapse_solve_ctx = SystemSliceConst(LapseOperator.Context).view(self.mesh.cell_total, .{
                    .psi = src.field(.psi),
                    .s = src.field(.s),
                    .u = src.field(.u),
                    .x = src.field(.x),
                    .y = src.field(.y),
                });

                self.solver.solve(
                    self.allocator,
                    self.mesh,
                    self.dof_map,
                    LapseOperator{},
                    lapse_view,
                    lapse_solve_ctx,
                    lapse_rhs.toConst(),
                ) catch {
                    unreachable;
                };

                // Fill `gauge.field(.lapse)`

                const gauge_lapse = SystemSlice(LapseOperator.System).view(self.dof_map.ndofs(), .{ .lapse = self.gauge.field(.lapse) });

                for (0..self.mesh.blocks.len) |block_id| {
                    DofUtils.copyDofsFromCells(
                        LapseOperator.System,
                        self.mesh,
                        self.dof_map,
                        block_id,
                        gauge_lapse,
                        lapse_view.toConst(),
                    );
                }

                for (0..self.mesh.blocks.len) |block_id| {
                    DofUtils.fillBoundary(
                        self.mesh,
                        self.dof_map,
                        block_id,
                        DofUtils.operSystemBoundary(LapseOperator{}),
                        gauge_lapse,
                    );
                }

                // Solve shift r

                const shift_r_view = SystemSlice(ShiftRRhs.System).view(self.mesh.cell_total, .{ .shift_r = self.gauge_cells.field(.shift_r) });
                const shift_r_rhs = SystemSlice(ShiftRRhs.System).view(self.mesh.cell_total, .{ .shift_r = self.gause_rhs.field(.shift_r) });

                const shift_r_ctx = SystemSliceConst(ShiftRRhs.Context).view(self.dof_map.ndofs(), .{
                    .lapse = self.gauge.field(.lapse),
                    .u = self.dynamic.field(.u),
                    .x = self.dynamic.field(.x),
                });

                DofUtilsTotal.project(
                    self.mesh,
                    self.dof_map,
                    ShiftRRhs{},
                    shift_r_rhs,
                    shift_r_ctx,
                );

                self.solver.solve(
                    self.allocator,
                    self.mesh,
                    self.dof_map,
                    ShiftROperator{},
                    shift_r_view,
                    aeon.EmptySystem.sliceConst(),
                    shift_r_rhs.toConst(),
                ) catch {
                    unreachable;
                };

                // Solve shift z

                const shift_z_view = SystemSlice(ShiftZRhs.System).view(self.mesh.cell_total, .{ .shift_z = self.gauge_cells.field(.shift_z) });
                const shift_z_rhs = SystemSlice(ShiftZRhs.System).view(self.mesh.cell_total, .{ .shift_z = self.gause_rhs.field(.shift_z) });

                const shift_z_ctx = SystemSliceConst(ShiftZRhs.Context).view(self.dof_map.ndofs(), .{
                    .lapse = self.gauge.field(.lapse),
                    .u = self.dynamic.field(.u),
                    .x = self.dynamic.field(.x),
                });

                DofUtilsTotal.project(
                    self.mesh,
                    self.dof_map,
                    ShiftZRhs{},
                    shift_z_rhs,
                    shift_z_ctx,
                );

                self.solver.solve(
                    self.allocator,
                    self.mesh,
                    self.dof_map,
                    ShiftZOperator{},
                    shift_z_view,
                    aeon.EmptySystem.sliceConst(),
                    shift_z_rhs.toConst(),
                ) catch {
                    unreachable;
                };

                // Fill with boundary conditions

                for (0..self.mesh.blocks.len) |block_id| {
                    DofUtils.copyDofsFromCells(
                        Elliptic,
                        self.mesh,
                        self.dof_map,
                        block_id,
                        self.gauge,
                        self.gauge_cells.toConst(),
                    );
                }

                for (0..self.mesh.blocks.len) |block_id| {
                    DofUtils.fillBoundary(
                        self.mesh,
                        self.dof_map,
                        block_id,
                        EllipticBoundary{},
                        self.gauge,
                    );
                }

                for (0..self.mesh.blocks.len) |block_id| {
                    DofUtils.apply(
                        self.mesh,
                        self.dof_map,
                        block_id,
                        DerivativeOperator{},
                        deriv,
                        self.dynamic.toConst(),
                        self.gauge.toConst(),
                    );
                }
            }
        };

        // Run

        fn run(allocator: Allocator) !void {
            std.debug.print("Running Brill Initial Data Solver\n", .{});

            // Build mesh

            var mesh = try Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, -10.0 },
                    .size = [2]f64{ 20.0, 20.0 },
                },
                .tile_width = 64,
                .index_size = [2]usize{ 1, 1 },
            });
            defer mesh.deinit();

            const dof_map: DofMap = try DofMap.init(allocator, &mesh);
            defer dof_map.deinit(allocator);

            std.debug.print("NDofs: {}\n", .{mesh.cell_total});

            // System to evolve

            var rk4 = try Rk4.init(allocator, mesh.cell_total);
            defer rk4.deinit();

            var dynamic = try SystemSlice(Hyperbolic).init(allocator, dof_map.ndofs());
            defer dynamic.deinit(allocator);

            var gauge = try SystemSlice(Elliptic).init(allocator, dof_map.ndofs());
            defer gauge.deinit(allocator);

            var gauge_cells = try SystemSlice(Elliptic).init(allocator, mesh.cell_total);
            defer gauge_cells.deinit(allocator);

            var gauge_rhs = try SystemSlice(Elliptic).init(allocator, mesh.cell_total);
            defer gauge_rhs.deinit(allocator);

            // ***************************
            // Solver ********************
            // ***************************

            const solver = LinearMapMethod.new(BiCGStabSolver.new(1000000, 10e-12));

            // ***************************
            // Initial Data **************
            // ***************************

            @memset(rk4.sys.field(.u), 0.0);
            @memset(rk4.sys.field(.y), 0.0);
            @memset(rk4.sys.field(.x), 0.0);

            std.debug.print("Solving Initial Data\n", .{});

            // Project initial s value

            const s = SystemSlice(InitialDataOperator.Context).view(mesh.cell_total, .{
                .s = rk4.sys.field(.s),
            });

            DofUtilsTotal.project(&mesh, dof_map, SProjection{
                .amplitude = 1.0,
                .sigma = 1.0,
            }, s, aeon.EmptySystem.sliceConst());

            // Project rhs value

            const rhs = try SystemSlice(InitialDataOperator.System).init(allocator, mesh.cell_total);
            defer rhs.deinit(allocator);

            {
                const s_dofs = try SystemSlice(InitialDataOperator.Context).init(allocator, dof_map.ndofs());
                defer s_dofs.deinit(allocator);

                for (0..mesh.blocks.len) |block_id| {
                    DofUtils.copyDofsFromCells(
                        InitialDataOperator.Context,
                        &mesh,
                        dof_map,
                        block_id,
                        s_dofs,
                        s.toConst(),
                    );
                }

                for (0..mesh.blocks.len) |block_id| {
                    DofUtils.fillBoundary(
                        &mesh,
                        dof_map,
                        block_id,
                        DofUtils.operContextBoundary(InitialDataOperator{}),
                        s_dofs,
                    );
                }

                DofUtilsTotal.project(&mesh, dof_map, InitialDataRhs{}, rhs, s_dofs.toConst());
            }

            const metric = SystemSlice(InitialDataOperator.System).view(mesh.cell_total, .{
                .psi = rk4.sys.field(.psi),
            });

            @memset(rk4.sys.field(.psi), 0.0);

            try solver.solve(
                allocator,
                &mesh,
                dof_map,
                InitialDataOperator{},
                metric,
                s.toConst(),
                rhs.toConst(),
            );

            std.debug.print("Initial Data Solved\n", .{});

            // *****************************
            // Evolve **********************
            // *****************************

            const steps: usize = 100;
            const h: f64 = 0.01;

            for (0..steps) |step| {
                std.debug.print("Running Step {}\n", .{step});
                // **************************
                // Output *******************
                // **************************

                const file_name = try std.fmt.allocPrint(allocator, "output/evolution_{}.vtu", .{step});
                defer allocator.free(file_name);

                const file = try std.fs.cwd().createFile(file_name, .{});
                defer file.close();

                try DataOut.writeVtk(Hyperbolic, allocator, &mesh, rk4.sys.toConst(), file.writer());

                // Step

                rk4.step(Derivative{
                    .mesh = &mesh,
                    .dof_map = dof_map,
                    .solver = solver,
                    .allocator = allocator,
                    .dynamic = dynamic,
                    .gauge = gauge,
                    .gauge_cells = gauge_cells,
                    .gause_rhs = gauge_rhs,
                }, h);
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
