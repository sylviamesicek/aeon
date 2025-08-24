use std::convert::Infallible;
use std::path::PathBuf;

use aeon::solver::SolverCallback;
use aeon::{mesh::Gaussian, prelude::*, solver::HyperRelaxSolver};

const ORDER: usize = 4;

const LOWER: f64 = 1e-8;
const UPPER: f64 = 1e-6;

#[derive(Clone)]
struct Conditions;

impl SystemBoundaryConds<2> for Conditions {
    fn kind(&self, _: usize, _face: Face<2>) -> BoundaryKind {
        BoundaryKind::StrongDirichlet
    }

    fn dirichlet(&self, _channel: usize, _position: [f64; 2]) -> DirichletParams {
        DirichletParams {
            target: 0.0,
            strength: 1.0,
        }
    }

    fn radiative(&self, _channel: usize, _position: [f64; 2]) -> RadiativeParams {
        RadiativeParams::lightlike(0.0)
    }
}

#[derive(Clone)]
struct PoissonEquation<'a> {
    source: &'a [f64],
}

impl<'a> Function<2> for PoissonEquation<'a> {
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: ImageRef,
        mut output: ImageMut,
    ) -> Result<(), Infallible> {
        let input = input.channel(0);
        let output = output.channel_mut(0);

        let source = &self.source[engine.node_range()];

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);

            let ddr = engine.second_derivative(input, 0, vertex);
            let ddz = engine.second_derivative(input, 1, vertex);

            output[index] = ddr + ddz - source[index];
        }

        Ok(())
    }
}

struct Callback;

// Implement visualization for hamiltonian.
impl SolverCallback<2> for Callback {
    type Error = std::io::Error;

    fn callback(
        &mut self,
        mesh: &Mesh<2>,
        input: ImageRef,
        output: ImageRef,
        iteration: usize,
    ) -> Result<(), Self::Error> {
        if iteration % 50 != 0 {
            return Ok(());
        }

        let i = iteration / 50;

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_field("Solution", input.channel(0));
        checkpoint.save_field("Derivative", output.channel(0));
        checkpoint.export_vtu(
            PathBuf::from("output/poisson").join(format!(
                "{}_level_{}_iter_{}.vtu",
                "poisson",
                mesh.num_levels(),
                i
            )),
            ExportVtuConfig {
                title: "poisson".to_string(),
                ghost: false,
                stride: ExportStride::PerVertex,
            },
        )
    }
}

pub fn main() -> eyre::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    std::fs::create_dir_all("output/poisson")?;

    log::info!("Intializing Mesh.");

    // Generate initial mesh
    let mut mesh = Mesh::new(
        HyperBox::from_aabb([-20., -20.], [20., 20.]),
        4,
        2,
        FaceArray::splat(BoundaryClass::OneSided),
    );
    mesh.refine_global();
    mesh.refine_global();
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
            4,
            Gaussian {
                amplitude: 1.0,
                sigma: [1.0, 1.0],
                center: [0., 0.],
            },
            &mut source,
        );
        mesh.fill_boundary(ORDER, Conditions, source.as_mut_slice().into());

        // Set initial guess
        solution.fill(0.0);

        let mut solver = HyperRelaxSolver::new();
        solver.dampening = 0.4;
        solver.max_steps = 100_000;
        solver.tolerance = 1e-6;
        solver.cfl = 0.5;
        solver.adaptive = true;

        solver.solve_with_callback(
            &mut mesh,
            ORDER,
            Conditions,
            Callback,
            PoissonEquation { source: &source },
            solution.as_mut_slice().into(),
        )?;

        mesh.flag_wavelets(4, LOWER, UPPER, solution.as_slice().into());

        mesh.limit_level_range_flags(1, 10);
        mesh.balance_flags();

        // Save data to file.

        let mut flags = vec![0; mesh.num_nodes()];
        mesh.flags_debug(&mut flags);

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_field("Source", &source);
        checkpoint.save_field("Solution", &solution);
        checkpoint.save_int_field("Flags", &flags);
        checkpoint.export_vtu(
            format!("output/poisson/poisson{i}.vtu"),
            ExportVtuConfig {
                title: "Poisson Equation".to_string(),
                ghost: false,
                stride: ExportStride::PerVertex,
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
