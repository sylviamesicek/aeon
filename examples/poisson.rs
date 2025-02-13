use aeon::{mesh::Gaussian, prelude::*, solver::HyperRelaxSolver};

const ORDER: Order<4> = Order::<4>;

const LOWER: f64 = 1e-8;
const UPPER: f64 = 1e-6;

#[derive(Clone)]
struct Conditions;

impl SystemConditions<2> for Conditions {
    type System = Scalar;

    fn radiative(&self, _field: (), _position: [f64; 2]) -> RadiativeParams {
        RadiativeParams::lightlike(0.0)
    }
}

#[derive(Clone)]
struct PoissonEquation<'a> {
    source: &'a [f64],
}

impl<'a> Function<2> for PoissonEquation<'a> {
    type Input = Scalar;
    type Output = Scalar;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
        let input = input.field(());
        let output = output.field_mut(());

        let source = &self.source[engine.node_range()];

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);

            let ddr = engine.second_derivative(input, 0, vertex);
            let ddz = engine.second_derivative(input, 1, vertex);

            output[index] = ddr + ddz - source[index];
        }
    }
}

pub fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    std::fs::create_dir_all("output/poisson")?;

    log::info!("Intializing Mesh.");

    // Generate initial mesh
    let mut mesh = Mesh::new(Rectangle::from_aabb([-20., -20.], [20., 20.]), 4, 2);
    mesh.set_boundary(BoundaryKind::Radiative);
    // Allocate space for system
    let mut source = Vec::new();
    let mut solution = Vec::new();

    log::info!("Performing Adaptive Mesh Refinement.");

    // Perform initial adaptive refinement.
    for i in 0..15 {
        log::info!("Poisson Iteration: {i}");

        source.resize(mesh.num_nodes(), 0.0);
        solution.resize(mesh.num_nodes(), 0.0);

        mesh.project(
            ORDER,
            Gaussian {
                amplitude: 1.0,
                sigma: [1.0, 1.0],
                center: [0., 0.],
            },
            &mut source,
        );
        mesh.fill_boundary(ORDER, Conditions, (&mut source).into());

        // Set initial guess
        solution.fill(0.0);

        let mut solver = HyperRelaxSolver::new();
        solver.dampening = 0.4;
        solver.max_steps = 100_000;
        solver.tolerance = 1e-6;
        solver.cfl = 0.5;
        solver.adaptive = true;

        solver.solve(
            &mut mesh,
            ORDER,
            Conditions,
            PoissonEquation { source: &source },
            (&mut solution).into(),
        )?;

        mesh.flag_wavelets(4, LOWER, UPPER, (&solution).into());
        mesh.set_regrid_level_limit(10);

        mesh.balance_flags();

        // Save data to file.

        let mut flags = vec![0; mesh.num_nodes()];
        mesh.flags_debug(&mut flags);

        let mut checkpoint = SystemCheckpoint::default();
        checkpoint.save_field("Source", &source);
        checkpoint.save_field("Solution", &solution);
        checkpoint.save_int_field("Flags", &flags);

        mesh.export_vtu(
            format!("output/poisson/poisson{i}.vtu"),
            &checkpoint,
            ExportVtuConfig {
                title: "Poisson Equation".to_string(),
                ghost: false,
                stride: 1,
            },
        )?;

        if i == 14 {
            log::info!("Failed to regrid completely within 15 iterations.");
            break;
        }

        if mesh.requires_regridding() {
            let refine = mesh.num_refine_cells();
            let coarsen = mesh.num_coarsen_cells();

            log::info!("Refining {refine} cells, coarsening {coarsen} cells.");
            mesh.regrid();
            continue;
        } else {
            log::info!("Regridded within range in {} iterations.", i + 1);
            break;
        }
    }

    Ok(())
}
