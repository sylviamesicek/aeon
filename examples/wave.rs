use std::array;

use aeon::{
    fd::{DissipationFunction, Gaussian},
    prelude::*,
    system::field_count,
};
use reborrow::{Reborrow, ReborrowMut};

const MAX_TIME: f64 = 1.0;
const MAX_STEPS: usize = 1000;

const CFL: f64 = 0.1;
const ORDER: Order<4> = Order::<4>;
const DISS_ORDER: Order<6> = Order::<6>;

const SAVE_CHECKPOINT: f64 = 0.01;
const FORCE_SAVE: bool = true;
const REGRID_SKIP: usize = 10;

const LOWER: f64 = 1e-10;
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

    fn radiative(&self, _field: Self::System, _position: [f64; 2]) -> f64 {
        0.0
    }
}

#[derive(Clone)]
struct WaveEquation {
    speed: [f64; 2],
}

impl Function<2> for WaveEquation {
    type Conditions = WaveConditions;

    type Input = Scalar;
    type Output = Scalar;

    fn conditions(&self) -> Self::Conditions {
        WaveConditions
    }

    fn evaluate(&self, engine: &impl Engine<2, Self::Input>) -> SystemValue<Self::Output> {
        let dr = engine.derivative(Scalar, 0);
        let dz = engine.derivative(Scalar, 0);
        SystemValue::new([-dr * self.speed[0] - dz * self.speed[1]])
    }
}

#[derive(Clone)]
struct AnalyticSolution {
    speed: [f64; 2],
    time: f64,
}

impl Projection<2> for AnalyticSolution {
    type Output = Scalar;

    fn project(&self, position: [f64; 2]) -> SystemValue<Self::Output> {
        let origin: [_; 2] = array::from_fn(|axis| self.speed[axis] * self.time);
        let offset: [_; 2] = array::from_fn(|axis| position[axis] - origin[axis]);
        let r2: f64 = offset.map(|v| v * v).iter().sum();
        SystemValue::new([(-r2).exp()])
    }
}

pub struct HyperbolicOde<'a> {
    mesh: &'a mut Mesh<2>,
}

impl<'a> Ode for HyperbolicOde<'a> {
    fn dim(&self) -> usize {
        field_count::<Scalar>() * self.mesh.num_nodes()
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        self.mesh.fill_boundary(
            ORDER,
            Quadrant,
            WaveConditions,
            SystemSliceMut::from_contiguous(system),
        );
    }

    fn derivative(&mut self, system: &[f64], result: &mut [f64]) {
        let src = SystemSlice::from_contiguous(system);
        let mut dest = SystemSliceMut::from_contiguous(result);

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
    let mut mesh = Mesh::new(Rectangle::from_aabb([-10., -10.], [10., 10.]), 4, 3);
    // Allocate space for system
    let mut system = SystemVec::new();

    log::info!("Performing Initial Adaptive Mesh Refinement.");

    // Perform initial adaptive refinement.
    for i in 0..15 {
        log::info!("Iteration: {i}");

        let profile = Gaussian {
            amplitude: 1.0,
            sigma: 1.0,
            center: [0., 0.],
        };

        system.resize(mesh.num_nodes());

        mesh.project(ORDER, Quadrant, profile, system.as_mut_slice());
        mesh.fill_boundary(ORDER, Quadrant, WaveConditions, system.as_mut_slice());

        mesh.flag_wavelets(LOWER, UPPER, Quadrant, system.as_slice());
        mesh.set_refine_level_limit(10);

        mesh.balance_flags();

        // Save data to file.

        let mut flags = vec![0; mesh.num_nodes()];
        mesh.flags_debug(&mut flags);

        let mut systems = SystemCheckpoint::default();
        systems.save_field("Wave", system.contigious());
        systems.save_int_field("Flags", &flags);

        let path = format!("output/waves/initial{i}.vtu");
        mesh.export_vtk(
            path.as_str(),
            ExportVtkConfig {
                title: "Initial Wave Mesh".to_string(),
                ghost: false,
                systems,
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
    let mut tmp = SystemVec::new();
    let mut update = SystemVec::<Scalar>::new();
    let mut dissipation = SystemVec::new();
    let mut exact = SystemVec::new();
    let mut error = SystemVec::<Scalar>::new();

    // Integrate
    let mut integrator = Rk4::new();
    let mut time = 0.0;
    let mut step = 0;

    let mut time_since_save = 0.0;
    let mut save_step = 0;

    let mut steps_since_regrid = 0;
    let mut regrid_save_step = 0;

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

        mesh.project(
            ORDER,
            Quadrant,
            AnalyticSolution { speed: SPEED, time },
            exact.as_mut_slice(),
        );

        // Fill boundaries
        mesh.fill_boundary(ORDER, Quadrant, WaveConditions, system.as_mut_slice());
        mesh.fill_boundary(ORDER, Quadrant, WaveConditions, exact.as_mut_slice());

        for i in 0..mesh.num_nodes() {
            error.contigious_mut()[i] = exact.contigious()[i] - system.contigious()[i];
        }

        if steps_since_regrid > REGRID_SKIP {
            steps_since_regrid = 0;

            log::info!("Regridding Mesh at time: {time}");
            mesh.fill_boundary(ORDER, Quadrant, WaveConditions, system.as_mut_slice());
            mesh.flag_wavelets(LOWER, UPPER, Quadrant, system.as_slice());
            mesh.set_refine_level_limit(10);

            // Output current system to disk
            let mut systems = SystemCheckpoint::default();
            systems.save_field("Wave", system.contigious());
            systems.save_field("Analytic", exact.contigious());
            systems.save_field("Error", error.contigious());

            let mut flags = vec![0; mesh.num_nodes()];
            mesh.flags_debug(&mut flags);

            mesh.balance_flags();

            let mut bflags = vec![0; mesh.num_nodes()];
            mesh.flags_debug(&mut bflags);

            systems.save_int_field("Flags", &mut flags);
            systems.save_int_field("Balanced Flags", &mut bflags);

            let mut blocks = vec![0; mesh.num_nodes()];
            mesh.block_debug(&mut blocks);
            systems.save_int_field("Blocks", &mut blocks);

            mesh.export_vtk(
                format!("output/waves/regrid{regrid_save_step}.vtu"),
                ExportVtkConfig {
                    title: "Rergrid Wave".to_string(),
                    ghost: false,
                    systems,
                },
            )
            .unwrap();

            regrid_save_step += 1;

            mesh.regrid();

            // Copy system into tmp.
            tmp.contigious_mut().copy_from_slice(system.contigious());

            system.resize(mesh.num_nodes());
            mesh.transfer_system(ORDER, Quadrant, tmp.as_slice(), system.as_mut_slice());

            continue;
        }

        mesh.fill_boundary(ORDER, Quadrant, WaveConditions, system.as_mut_slice());

        if time_since_save >= SAVE_CHECKPOINT || FORCE_SAVE {
            time_since_save = 0.0;

            log::info!("Saving Checkpoint at time: {time}");
            // Output current system to disk
            let mut systems = SystemCheckpoint::default();
            systems.save_field("Wave", system.contigious());
            systems.save_field("Analytic", exact.contigious());
            systems.save_field("Error", error.contigious());

            let mut blocks = vec![0; mesh.num_nodes()];
            mesh.block_debug(&mut blocks);
            systems.save_int_field("Blocks", &mut blocks);

            let mut interfaces = vec![0; mesh.num_nodes()];
            mesh.interface_index_debug(3, &mut interfaces);
            systems.save_int_field("Interfaces", &mut interfaces);

            mesh.export_vtk(
                format!("output/waves/evolution{save_step}.vtu"),
                ExportVtkConfig {
                    title: "evbrill".to_string(),
                    ghost: false,
                    systems,
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
        mesh.evaluate(
            DISS_ORDER,
            Quadrant,
            DissipationFunction(WaveConditions),
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
