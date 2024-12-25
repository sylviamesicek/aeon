use std::array;

use aeon::{fd::Gaussian, prelude::*, system::System};
use aeon_basis::RadiativeParams;
use reborrow::{Reborrow, ReborrowMut};

const MAX_TIME: f64 = 1.0;
const MAX_STEPS: usize = 1000;

const CFL: f64 = 0.1;
const ORDER: Order<4> = Order::<4>;
const DISS_ORDER: Order<6> = Order::<6>;

const SAVE_CHECKPOINT: f64 = 0.01;
const FORCE_SAVE: bool = false;
const REGRID_SKIP: usize = 10;

const LOWER: f64 = 1e-8;
const UPPER: f64 = 1e-6;
const SPEED: [f64; 2] = [1.0, 0.0];

/// The quadrant domain the function is being projected on.
#[derive(Clone)]
struct Quadrant;

impl Boundary<2> for Quadrant {
    fn kind(&self, _face: Face<2>) -> BoundaryKind {
        BoundaryKind::Radiative
    }
}

#[derive(Clone)]
struct WaveConditions;

impl Conditions<2> for WaveConditions {
    type System = Scalar;

    fn radiative(&self, _field: (), _position: [f64; 2], _spacing: f64) -> RadiativeParams {
        RadiativeParams::lightlike(0.0)
    }
}

#[derive(Clone)]
struct WaveEquation {
    speed: [f64; 2],
}

impl Function<2> for WaveEquation {
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

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);

            let dr = engine.derivative(input, 0, vertex);
            let dz = engine.derivative(input, 1, vertex);

            output[index] = -dr * self.speed[0] - dz * self.speed[1];
        }
    }
}

#[derive(Clone)]
struct AnalyticSolution {
    speed: [f64; 2],
    time: f64,
}

impl Projection<2> for AnalyticSolution {
    fn project(&self, position: [f64; 2]) -> f64 {
        let origin: [_; 2] = array::from_fn(|axis| self.speed[axis] * self.time);
        let offset: [_; 2] = array::from_fn(|axis| position[axis] - origin[axis]);
        let r2: f64 = offset.map(|v| v * v).iter().sum();
        (-r2).exp()
    }
}

pub struct HyperbolicOde<'a> {
    mesh: &'a mut Mesh<2>,
}

impl<'a> Ode for HyperbolicOde<'a> {
    fn dim(&self) -> usize {
        Scalar.count() * self.mesh.num_nodes()
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        self.mesh.fill_boundary(
            ORDER,
            Quadrant,
            WaveConditions,
            SystemSliceMut::from_scalar(system),
        );
    }

    fn derivative(&mut self, system: &[f64], result: &mut [f64]) {
        let src = SystemSlice::from_scalar(system);
        let mut dest = SystemSliceMut::from_scalar(result);

        self.mesh.evaluate(
            ORDER,
            Quadrant,
            WaveEquation { speed: SPEED },
            src.rb(),
            dest.rb_mut(),
        );

        self.mesh
            .weak_boundary(ORDER, Quadrant, WaveConditions, src.rb(), dest.rb_mut());
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    std::fs::create_dir_all("output/waves")?;

    log::info!("Intializing Mesh.");

    // Generate initial mesh
    let mut mesh = Mesh::new(Rectangle::from_aabb([-10., -10.], [10., 10.]), 6, 3);
    // Allocate space for system
    let mut system = SystemVec::<Scalar>::default();

    log::info!("Performing Initial Adaptive Mesh Refinement.");

    // Perform initial adaptive refinement.
    for i in 0..15 {
        log::info!("Iteration: {i}");

        let profile = Gaussian {
            amplitude: 1.0,
            sigma: [1.0, 1.0],
            center: [0., 0.],
        };

        system.resize(mesh.num_nodes());

        mesh.project_scalar(ORDER, Quadrant, profile, system.field_mut(()));
        mesh.fill_boundary(ORDER, Quadrant, WaveConditions, system.as_mut_slice());

        mesh.flag_wavelets(4, LOWER, UPPER, Quadrant, system.as_slice());
        mesh.set_regrid_level_limit(10);

        mesh.balance_flags();

        // Save data to file.

        let mut flags = vec![0; mesh.num_nodes()];
        mesh.flags_debug(&mut flags);

        let mut checkpoint = SystemCheckpoint::default();
        checkpoint.save_field("Wave", system.contigious());
        checkpoint.save_int_field("Flags", &flags);

        mesh.export_vtu(
            format!("output/waves/initial{i}.vtu"),
            &checkpoint,
            ExportVtuConfig {
                title: "Initial Wave Mesh".to_string(),
                ghost: false,
                stride: 6,
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

    log::info!("Evolving wave forwards");

    // Allocate vectors
    let mut tmp = SystemVec::default();
    let mut update = SystemVec::<Scalar>::default();
    let mut dissipation = SystemVec::default();
    let mut exact = SystemVec::default();
    let mut error = SystemVec::<Scalar>::default();

    // Integrate
    let mut integrator = Rk4::new();
    let mut time = 0.0;
    let mut step = 0;

    let mut time_since_save = 0.0;
    let mut save_step = 0;

    let mut steps_since_regrid = 0;

    while step < MAX_STEPS && time < MAX_TIME {
        assert!(system.len() == mesh.num_nodes());

        // Get step size
        let h = mesh.min_spacing() * CFL;

        // Resize vectors
        tmp.resize(mesh.num_nodes());
        update.resize(mesh.num_nodes());
        dissipation.resize(mesh.num_nodes());
        exact.resize(mesh.num_nodes());
        error.resize(mesh.num_nodes());

        mesh.project_scalar(
            ORDER,
            Quadrant,
            AnalyticSolution { speed: SPEED, time },
            exact.field_mut(()),
        );

        // Fill boundaries
        mesh.fill_boundary(ORDER, Quadrant, WaveConditions, system.as_mut_slice());
        mesh.fill_boundary(ORDER, Quadrant, WaveConditions, exact.as_mut_slice());

        for i in 0..mesh.num_nodes() {
            error.contigious_mut()[i] = exact.contigious()[i] - system.contigious()[i];
        }

        if steps_since_regrid > REGRID_SKIP {
            steps_since_regrid = 0;

            mesh.fill_boundary(ORDER, Quadrant, WaveConditions, system.as_mut_slice());
            mesh.flag_wavelets(4, LOWER, UPPER, Quadrant, system.as_slice());
            mesh.set_regrid_level_limit(10);

            mesh.balance_flags();

            let num_refine = mesh.num_refine_cells();
            let num_coarsen = mesh.num_coarsen_cells();

            mesh.regrid();

            log::info!(
                "Regrided Mesh at time: {time:.5}, Max Level {}, {} R, {} C",
                mesh.max_level(),
                num_refine,
                num_coarsen,
            );

            // Copy system into tmp.
            tmp.contigious_mut().copy_from_slice(system.contigious());

            system.resize(mesh.num_nodes());
            mesh.transfer_system(ORDER, Quadrant, tmp.as_slice(), system.as_mut_slice());

            continue;
        }

        mesh.fill_boundary(ORDER, Quadrant, WaveConditions, system.as_mut_slice());

        if time_since_save >= SAVE_CHECKPOINT || FORCE_SAVE {
            time_since_save = 0.0;

            log::info!(
                "Saving Checkpoint {save_step}
    Time: {time:.5}, Step: {h:.8}
    Nodes: {}",
                mesh.num_nodes()
            );
            // Output current system to disk
            let mut checkpoint = SystemCheckpoint::default();
            checkpoint.save_field("Wave", system.contigious());
            checkpoint.save_field("Analytic", exact.contigious());
            checkpoint.save_field("Error", error.contigious());

            let mut blocks = vec![0; mesh.num_nodes()];
            mesh.block_debug(&mut blocks);
            checkpoint.save_int_field("Blocks", &mut blocks);

            let mut interfaces = vec![0; mesh.num_nodes()];
            mesh.interface_index_debug(3, &mut interfaces);
            checkpoint.save_int_field("Interfaces", &mut interfaces);

            mesh.export_vtu(
                format!("output/waves/evolution{save_step}.vtu"),
                &checkpoint,
                ExportVtuConfig {
                    title: "evbrill".to_string(),
                    ghost: false,
                    stride: 6,
                },
            )
            .unwrap();

            save_step += 1;
        }

        // Compute step
        integrator.step(
            h,
            &mut HyperbolicOde { mesh: &mut mesh },
            system.contigious(),
            update.contigious_mut(),
        );

        // Compute dissipation
        mesh.dissipation(
            DISS_ORDER,
            Quadrant,
            system.as_slice(),
            dissipation.as_mut_slice(),
        );

        // Add everything together
        for i in 0..system.contigious().len() {
            system.contigious_mut()[i] +=
                update.contigious()[i] + 0.5 * dissipation.contigious()[i];
        }

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;
    }

    Ok(())
}
