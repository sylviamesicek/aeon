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
        const DofMap = dofs.DofMap(N, O);
        const DofUtils = dofs.DofUtils(N, O);
        const LinearMapMethod = methods.LinearMapMethod(N, O, BiCGStabSolver);
        const Rk4 = methods.RungeKutta4Integrator(TemporalDerivative.Hyperbolic);
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
            if (face.side == false) {
                return BoundaryCondition.nuemann(0.0);
            } else {
                const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                return BoundaryCondition.robin(1.0 / r, 1.0, 0.0);
            }
        }

        fn oddBoundaryCondition(pos: [N]f64, face: Face) BoundaryCondition {
            if (face.side == false) {
                return BoundaryCondition.diritchlet(0.0);
            } else {
                const r: f64 = @sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                return BoundaryCondition.robin(1.0 / r, 1.0, 0.0);
            }
        }

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

        pub const MetricInitialData = struct {
            pub const System = enum {
                psi,
            };

            pub const Context = SProjection.System;

            pub fn apply(_: MetricInitialData, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianSys(.psi);
                const gradient: [N]f64 = engine.gradientSys(.psi);
                const value: f64 = engine.valueSys(.psi);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                const s_hessian: [N][N]f64 = engine.hessianSys(.s);
                const s_lap = s_hessian[0][0] + s_hessian[1][1];

                return .{
                    .psi = -lap - s_lap * value,
                };
            }

            pub fn applyDiagonal(_: MetricInitialData, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianDiagonal();
                const gradient: [N]f64 = engine.gradientDiagonal();
                const value: f64 = engine.valueDiagonal();

                const seed: f64 = engine.valueCtx(.seed);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                return .{
                    .psi = -lap - seed * value,
                };
            }

            pub fn boundarySys(_: MetricInitialData, pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                return .{
                    .psi = evenBoundaryCondition(pos, face),
                };
            }

            pub fn boundaryCtx(_: MetricInitialData, pos: [N]f64, face: Face) SystemBoundaryCondition(Context) {
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

                const seed = engine.valueCtx(.s);
                const u = engine.valueCtx(.u);
                const y = engine.valueCtx(.y);
                const x = engine.valueCtx(.x);

                const psi = psi_val;

                const scale = psi * psi * psi * psi * @exp(2 * r * seed);

                const term3 = (u - 0.5 * r * y) * (u - 0.5 * r * y);
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

            pub fn apply(_: LapseOperator, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
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

                const seed = engine.valueCtx(.s);
                const u = engine.valueCtx(.u);
                const y = engine.valueCtx(.y);
                const x = engine.valueCtx(.x);

                const psi = psi_val;
                const psi_r = psi_grad[0];
                const psi_z = psi_grad[1];

                const term1 = lapse_rr + lapse_r / r + lapse_zz;
                const term2 = 2.0 / psi * (psi_r * lapse_r + psi_z * lapse_z);

                const scale = lapse * psi * psi * psi * psi * @exp(2 * r * seed);

                const term3 = (u - 0.5 * r * y) * (u - 0.5 * r * y);
                const term4 = 0.5 * r * r * y * y;
                const term5 = 2.0 * x * x;

                return .{
                    .lapse = term1 + term2 - scale * (term3 + term4 + term5),
                };
            }

            pub fn applyDiagonal(_: LapseOperator, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
                const position: [N]f64 = engine.position();
                const r = position[0];

                const hessian: [N][N]f64 = engine.hessianDiagonal();
                const gradient: [N]f64 = engine.gradientDiagonal();
                const value: f64 = engine.valueDiagonal();

                const lapse = value;
                const lapse_r = gradient[0];
                const lapse_z = gradient[1];
                const lapse_rr = hessian[0][0];
                const lapse_zz = hessian[1][1];

                const psi_val = engine.valueCtx(.psi) + 1.0;
                const psi_grad = engine.gradientCtx(.psi);

                const seed = engine.valueCtx(.s);
                const u = engine.valueCtx(.u);
                const y = engine.valueCtx(.y);
                const x = engine.valueCtx(.x);

                const psi = psi_val;
                const psi_r = psi_grad[0];
                const psi_z = psi_grad[1];

                const term1 = lapse_rr + lapse_r / r + lapse_zz;
                const term2 = 2.0 / psi * (psi_r * lapse_r + psi_z * lapse_z);

                const scale = lapse * psi * psi * psi * psi * @exp(2 * r * seed);

                const term3 = (u - 0.5 * r * y) * (u - 0.5 * r * y);
                const term4 = 0.5 * r * r * y * y;
                const term5 = 2.0 * x * x;

                return .{
                    .lapse = term1 + term2 - scale * (term3 + term4 + term5),
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

            pub fn project(_: LapseRhs, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
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

            pub fn boundaryCtx(_: LapseRhs, pos: [N]f64, face: Face) SystemBoundaryCondition(Context) {
                return .{
                    .lapse = evenBoundaryCondition(pos, face),
                    .u = evenBoundaryCondition(pos, face),
                    .x = oddBoundaryCondition(pos, face),
                };
            }
        };

        pub const ShiftROperator = struct {
            pub const System = enum {
                shfit_r,
            };

            pub const Context = aeon.EmptySystem;

            pub fn apply(_: ShiftROperator, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
                const hessian: [N][N]f64 = engine.hessianSys(.shift_r);

                return .{
                    .shift_r = hessian[0][0] + hessian[1][1],
                };
            }

            pub fn applyDiagonal(_: ShiftROperator, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
                const hessian: [N][N]f64 = engine.hessianDiagonal();

                return .{
                    .shift_r = hessian[0][0] + hessian[1][1],
                };
            }

            pub fn boundarySys(_: ShiftROperator, pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                return .{
                    .shfit_r = evenBoundaryCondition(pos, face),
                };
            }

            pub fn boundaryCtx(_: ShiftROperator, _: [N]f64, _: Face) SystemBoundaryCondition(Context) {
                return .{};
            }
        };

        pub const ShiftZRhs = struct {
            pub const System = ShiftROperator.System;
            pub const Context = enum {
                lapse,
                u,
                x,
            };

            pub fn project(_: LapseRhs, engine: dofs.ProjectionEngine(N, O, Context)) SystemValue(System) {
                const lapse_val = engine.valueCtx(.lapse) + 1.0;
                const lapse_grad = engine.gradientCtx(.lapse);

                const x_val = engine.valueCtx(.x);
                const x_grad = engine.gradientCtx(.x);

                const u_val = engine.valueCtx(.u);
                const u_grad = engine.gradientCtx(.u);

                return .{
                    .shift_r = lapse_val * x_grad[1] + lapse_grad[1] * x_val - lapse_val * u_grad[0] - lapse_grad[0] * u_val,
                };
            }

            pub fn boundaryCtx(_: LapseRhs, pos: [N]f64, face: Face) SystemBoundaryCondition(Context) {
                return .{
                    .lapse = evenBoundaryCondition(pos, face),
                    .u = evenBoundaryCondition(pos, face),
                    .x = oddBoundaryCondition(pos, face),
                };
            }
        };

        pub const ShiftZOperator = struct {
            pub const System = enum {
                shfit_z,
            };

            pub const Context = aeon.EmptySystem;

            pub fn apply(_: ShiftZOperator, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
                const hessian: [N][N]f64 = engine.hessianSys(.shift_z);

                return .{
                    .shift_z = hessian[0][0] + hessian[1][1],
                };
            }

            pub fn applyDiagonal(_: ShiftZOperator, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
                const hessian: [N][N]f64 = engine.hessianDiagonal();

                return .{
                    .shift_z = hessian[0][0] + hessian[1][1],
                };
            }

            pub fn boundarySys(_: ShiftZOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(System) {
                return .{
                    .shift_z = oddBoundaryCondition(pos, face),
                };
            }

            pub fn boundaryCtx(_: ShiftZOperator, _: [N]f64, _: Face) SystemBoundaryCondition(Context) {
                return .{};
            }
        };

        pub const TemporalDerivative = struct {
            mesh: Mesh,
            dof_map: DofMap,
            allocator: Allocator,
            solver: LinearMapMethod,
            hyperbolic_dofs: SystemSlice(Hyperbolic),
            elliptic_dofs: SystemSlice(Elliptic),
            elliptic_cells: SystemSlice(Elliptic),
            elliptic_rhs: SystemSlice(Elliptic),

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

            pub const EllipticBoundary = struct {
                pub const System = Elliptic;

                pub fn boundary(_: @This(), pos: [N]f64, face: Face(N)) SystemBoundaryCondition(Elliptic) {
                    return .{
                        .lapse = evenBoundaryCondition(pos, face),
                        .shift_r = oddBoundaryCondition(pos, face),
                        .shift_z = evenBoundaryCondition(pos, face),
                    };
                }
            };

            pub const HyperbolicBoundary = struct {
                pub const System = Hyperbolic;

                pub fn boundary(_: @This(), pos: [N]f64, face: Face(N)) SystemBoundaryCondition(Hyperbolic) {
                    return .{
                        .psi = evenBoundaryCondition(pos, face),
                        .s = oddBoundaryCondition(pos, face),
                        .u = evenBoundaryCondition(pos, face),
                        .y = oddBoundaryCondition(pos, face),
                        .x = oddBoundaryCondition(pos, face),
                    };
                }
            };

            pub const System = Hyperbolic;

            pub fn derivative(self: @This(), deriv: SystemSlice(Hyperbolic), src: SystemSliceConst(Hyperbolic)) void {
                _ = deriv;

                // Copy to hyperbolic dof vectors

                for (0..self.mesh.blocks.len) |block_id| {
                    DofUtils.copyCellsFromDofs(Hyperbolic, self.mesh, self.dof_map, block_id, self.hyperbolic_dofs, src);
                }

                for (0..self.mesh.blocks.len) |block_id| {
                    DofUtils.fillBoundary(self.mesh, self.dof_map, block_id, HyperbolicBoundary{}, self.hyperbolic_dofs);
                }

                // Set initial guess for elliptic cells

                @memset(self.elliptic_cells.field(.lapse), 1.0);
                @memset(self.elliptic_cells.field(.shift_r), 0.0);
                @memset(self.elliptic_cells.field(.shift_z), 0.0);

                // Solve Lapse

                const lapse_view = SystemSlice(LapseRhs.System).view(self.mesh.cell_total, .{ .lapse = self.elliptic_cells.field(.lapse) });
                const lapse_rhs = SystemSlice(LapseRhs.System).view(self.mesh.cell_total, .{ .lapse = self.elliptic_rhs.field(.lapse) });
                const lapse_ctx = SystemSliceConst(LapseRhs.Context).view(self.mesh.cell_total, .{
                    .psi = self.hyperbolic_dofs.field(.psi),
                    .seed = self.hyperbolic_dofs.field(.s),
                    .u = self.hyperbolic_dofs.field(.u),
                    .x = self.hyperbolic_dofs.field(.x),
                    .y = self.hyperbolic_dofs.field(.y),
                });

                DofUtils.projectCells(
                    self.mesh,
                    self.dof_map,
                    LapseRhs{},
                    lapse_rhs,
                    lapse_ctx,
                );

                self.solver.solve(
                    self.allocator,
                    &self.mesh,
                    self.dof_map,
                    LapseOperator{},
                    lapse_view,
                    lapse_rhs.toConst(),
                    lapse_ctx,
                ) catch {
                    unreachable;
                };

                // Solve shift

                const shift_r_ctx = SystemSliceConst(ShiftRRhs.Context).view(self.mesh.cell_total, .{
                    .psi = self.hyperbolic_dofs.field(.psi),
                    .s = self.hyperbolic_dofs.field(.s),
                    .u = self.hyperbolic_dofs.field(.u),
                    .x = self.hyperbolic_dofs.field(.x),
                    .y = self.hyperbolic_dofs.field(.y),
                });
                _ = shift_r_ctx;
            }
        };

        // Run

        fn run(allocator: Allocator) !void {
            std.debug.print("Running Brill Initial Data Solver\n", .{});

            // Build mesh

            var mesh = try Mesh.init(allocator, .{
                .physical_bounds = .{
                    .origin = [2]f64{ 0.0, 0.0 },
                    .size = [2]f64{ 10.0, 10.0 },
                },
                .tile_width = 128,
                .index_size = [2]usize{ 1, 1 },
            });
            defer mesh.deinit();

            const dof_map: DofMap = try DofMap.init(allocator, &mesh);
            defer dof_map.deinit(allocator);

            std.debug.print("NDofs: {}\n", .{mesh.cell_total});

            // System to evolve

            const rk4 = Rk4.init(allocator, mesh.cell_total);
            defer rk4.deinit();

            var hyperbolic_dofs = try SystemSlice(TemporalDerivative.Hyperbolic).init(allocator, dof_map.ndofs());
            defer hyperbolic_dofs.deinit(allocator);

            var elliptic_dofs = try SystemSlice(TemporalDerivative.Elliptic).init(allocator, dof_map.ndofs());
            defer elliptic_dofs.deinit(allocator);

            var elliptic_cells = try SystemSlice(TemporalDerivative.Elliptic).init(allocator, dof_map.ndofs());
            defer elliptic_cells.deinit(allocator);

            var elliptic_rhs = try SystemSlice(TemporalDerivative.Elliptic).init(allocator, mesh.cell_total);
            defer elliptic_rhs.deinit(allocator);

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

            const seed = SystemSlice(MetricInitialData.Context).view(mesh.cell_total, .{
                .s = rk4.sys.field(.s),
            });

            DofUtils.projectCells(mesh, dof_map, SProjection{
                .amplitude = 1.0,
                .sigma = 1.0,
            }, seed, aeon.EmptySystem.sliceConst());

            const rhs = SystemSlice(MetricInitialData.System).view(mesh.cell_total, .{
                .psi = rk4.sys.field(.s),
            });

            const metric = try SystemSlice(MetricInitialData.System).view(mesh.cell_total, .{
                .psi = rk4.sys.field(.psi),
            });

            @memset(rk4.sys.field(.psi), 0.0);

            try solver.solve(
                allocator,
                &mesh,
                dof_map,
                MetricInitialData{},
                metric,
                rhs.toConst(),
                seed.toConst(),
            );

            // *****************************
            // Evolve **********************
            // *****************************

            const steps: usize = 100;
            const h: f64 = 0.01;

            for (0..steps) |step| {
                // **************************
                // Output *******************
                // **************************

                const file_name = try std.fmt.allocPrint(allocator, "output/evolution_{}.vtu", .{step});
                defer allocator.free(file_name);

                const file = try std.fs.cwd().createFile(file_name, .{});
                defer file.close();

                try DofUtils.writeCellsToVtk(TemporalDerivative.Hyperbolic, allocator, &mesh, rk4.sys, file.writer());

                // Step

                rk4.step(TemporalDerivative{
                    .mesh = mesh,
                    .dof_map = dof_map,
                    .solver = solver,
                    .allocator = allocator,
                    .hyperbolic_dofs = hyperbolic_dofs,
                    .elliptic_dofs = elliptic_dofs,
                    .elliptic_cells = elliptic_cells,
                    .elliptic_rhs = elliptic_rhs,
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
