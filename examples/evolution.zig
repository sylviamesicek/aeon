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
        const Rk4 = methods.RungeKutta4Integrator(Evolution);
        const SystemSlice = aeon.SystemSlice;
        const SystemSliceConst = aeon.SystemSliceConst;
        const SystemValue = aeon.SystemValue;
        const SystemBoundaryCondition = dofs.SystemBoundaryCondition;

        const Face = geometry.Face(N);
        const IndexSpace = geometry.IndexSpace(N);
        const Index = index.Index(N);

        const BiCGStabSolver = lac.BiCGStabSolver;

        const Mesh = aeon.mesh.Mesh(N);

        pub const Evolution = enum {
            lapse,
            shift_r,
            shift_z,
            psi,
            seed,
            u,
            y,
            x,
        };

        pub const Seed = enum {
            seed,
        };

        pub const Metric = enum {
            psi,
        };

        pub const SeedProjection = struct {
            amplitude: f64,
            sigma: f64,

            pub const System = Seed;

            pub fn project(self: SeedProjection, pos: [N]f64) SystemValue(System) {
                const rho = pos[0];
                const z = pos[1];

                const rho2 = rho * rho;
                const z2 = z * z;
                const sigma2 = self.sigma * self.sigma;

                const term1: f64 = 0.5 * self.amplitude;
                const term2 = 2 * rho2 * rho2 - 6 * rho2 * sigma2 + sigma2 * sigma2 + 2 * rho2 * z2;
                const term3 = @exp(-(rho2 + z2) / sigma2) / (sigma2 * sigma2 * sigma2);

                return .{
                    .seed = term1 * term2 * term3,
                };
            }
        };

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

        pub const MetricInitialData = struct {
            pub const Context = Seed;
            pub const System = Metric;

            pub fn apply(_: MetricInitialData, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
                const position: [N]f64 = engine.position();

                const hessian: [N][N]f64 = engine.hessianSys(.psi);
                const gradient: [N]f64 = engine.gradientSys(.psi);
                const value: f64 = engine.valueSys(.psi);

                const seed: f64 = engine.valueCtx(.seed);

                const lap = hessian[0][0] + hessian[1][1] + gradient[0] / position[0];

                return .{
                    .psi = -lap - seed * value,
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
                    .seed = oddBoundaryCondition(pos, face),
                };
            }
        };

        pub const LapseOperator = struct {
            pub const Context = enum {
                psi,
                seed,
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

                const seed = engine.valueCtx(.seed);
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

                const seed = engine.valueCtx(.seed);
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
                    .seed = oddBoundaryCondition(pos, face),
                    .u = evenBoundaryCondition(pos, face),
                    .x = oddBoundaryCondition(pos, face),
                    .y = oddBoundaryCondition(pos, face),
                };
            }
        };

        pub const ShiftROperator = struct {
            pub const Context = enum {
                lapse,
                u,
                x,
            };

            pub const System = enum {
                shfit_r,
            };

            pub fn apply(_: LapseOperator, engine: dofs.Engine(N, O, System, Context)) SystemValue(System) {
                const position: [N]f64 = engine.position();
                const r = position[0];
                _ = r;

                const hessian: [N][N]f64 = engine.hessianSys(.shift_r);
                const value: f64 = engine.valueSys(.shift_r);
                _ = value;

                const beta = hessian[0][0] + hessian[1][1];
                _ = beta;

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

                const seed = engine.valueCtx(.seed);
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
                    .shfit_r = evenBoundaryCondition(pos, face),
                };
            }

            pub fn boundaryCtx(_: LapseOperator, pos: [N]f64, face: Face) SystemBoundaryCondition(Context) {
                return .{
                    .lapse = evenBoundaryCondition(pos, face),
                    .u = evenBoundaryCondition(pos, face),
                    .x = oddBoundaryCondition(pos, face),
                };
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

            const block_map: []usize = try allocator.alloc(usize, mesh.tile_total);
            defer allocator.free(block_map);

            mesh.buildBlockMap(block_map);

            const dof_map: DofMap = try DofMap.init(allocator, &mesh);
            defer dof_map.deinit(allocator);

            std.debug.print("NDofs: {}\n", .{mesh.cell_total});

            // System to evolve

            const rk4 = Rk4.init(allocator, mesh.cell_total);
            defer rk4.deinit();

            // ***************************
            // Solver ********************
            // ***************************

            const solver = LinearMapMethod.new(BiCGStabSolver.new(1000000, 10e-12));

            // ***************************
            // Initial Data **************
            // ***************************

            @memset(rk4.sys.field(.lapse), 1.0);
            @memset(rk4.sys.field(.shift_r), 0.0);
            @memset(rk4.sys.field(.shift_z), 0.0);
            @memset(rk4.sys.field(.u), 0.0);
            @memset(rk4.sys.field(.y), 0.0);
            @memset(rk4.sys.field(.x), 0.0);

            std.debug.print("Solving Initial Data\n", .{});

            const seed = SystemSlice(Seed).view(mesh.cell_total, .{
                .seed = rk4.sys.field(.seed),
            });

            const rhs = SystemSlice(Metric).view(mesh.cell_total, .{
                .metric = rk4.sys.field(.seed),
            });

            DofUtils.projectCells(&mesh, SeedProjection{
                .amplitude = 1.0,
                .sigma = 1.0,
            }, seed);

            const metric = try SystemSlice(Metric).view(mesh.cell_total, .{
                .psi = rk4.sys.field(.psi),
            });

            @memset(rk4.sys.field(.psi), 0.0);

            try solver.solve(
                allocator,
                &mesh,
                block_map,
                dof_map,
                MetricInitialData{},
                metric,
                rhs.toConst(),
                seed.toConst(),
            );

            // *****************************
            // Evolve **********************
            // *****************************

            const steps = 100;

            for (0..steps) |step| {
                // **************************
                // Output *******************
                // **************************

                const file_name = try std.fmt.allocPrint(allocator, "output/evolution_{}.vtu", .{step});
                defer allocator.free(file_name);

                const file = try std.fs.cwd().createFile(file_name, .{});
                defer file.close();

                try DofUtils.writeCellsToVtk(Evolution, allocator, &mesh, rk4.sys, file.writer());
            }

            std.debug.print("Writing Solution To File\n", .{});

            const file = try std.fs.cwd().createFile("output/seed.vtu", .{});
            defer file.close();

            try DofUtils.writeCellsToVtk(Evolution, allocator, &mesh, rk4.sys, file.writer());
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
