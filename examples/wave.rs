use aeon::{
    fd::{DissipationFunction, Gaussian},
    prelude::*,
    system::field_count,
};
use reborrow::{Reborrow, ReborrowMut};

const STEPS: usize = 500;
const CFL: f64 = 0.1;
const ORDER: Order<4> = Order::<4>;
const DISS_ORDER: Order<6> = Order::<6>;

const OUTPUT_SKIP: usize = 10;
// const REGRID_SKIP: usize = 10;

const LOWER: f64 = 1e-10;
const UPPER: f64 = 1e-6;
const SPEED: [f64; 2] = [1.0, 0.0];

/// The quadrant domain the function is being projected on.
#[derive(Clone)]
pub struct Quadrant;

impl Boundary<2> for Quadrant {
    fn kind(&self, _face: Face<2>) -> BoundaryKind {
        BoundaryKind::Radiative
    }
}

#[derive(Clone)]
pub struct WaveConditions;

impl Conditions<2> for WaveConditions {
    type System = Scalar;

    fn radiative(&self, _field: Self::System, _position: [f64; 2]) -> f64 {
        0.0
    }
}

#[derive(Clone)]
pub struct WaveEquation {
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

    // Allocate vectors
    let mut update = SystemVec::<Scalar>::with_length(mesh.num_nodes());
    let mut dissipation = SystemVec::with_length(mesh.num_nodes());

    // Integrate
    let mut integrator = Rk4::new();

    for i in 0..STEPS {
        // Fill ghost nodes of system
        mesh.fill_boundary(ORDER, Quadrant, WaveConditions, system.as_mut_slice());

        if i % OUTPUT_SKIP == 0 {
            log::info!("Output step: {i}");
            // Output current system to disk
            let mut systems = SystemCheckpoint::default();
            systems.save_field("Wave", system.contigious());

            mesh.export_vtk(
                format!("output/waves/evolution{}.vtu", i / OUTPUT_SKIP),
                ExportVtkConfig {
                    title: "evbrill".to_string(),
                    ghost: false,
                    systems,
                },
            )
            .unwrap();
        }

        let h = CFL * mesh.min_spacing();

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
    }

    Ok(())
}
