use std::{array, convert::Infallible};

use aeon::{
    mesh::Gaussian,
    prelude::*,
    solver::{Integrator, Method},
};

const MAX_TIME: f64 = 10.0;
const MAX_STEPS: usize = 10000;

const CFL: f64 = 0.1;
const ORDER: usize = 4;

const SAVE_CHECKPOINT: f64 = 0.01;
const FORCE_SAVE: bool = false;
const REGRID_SKIP: usize = 10;

const LOWER: f64 = 1e-8;
const UPPER: f64 = 1e-6;
const SPEED: [f64; 2] = [1.0, 0.0];

#[derive(Clone)]
struct WaveConditions;

impl SystemBoundaryConds<2> for WaveConditions {
    fn kind(&self, _channel: usize, _face: Face<2>) -> BoundaryKind {
        BoundaryKind::Radiative
    }

    fn radiative(&self, _channel: usize, _position: [f64; 2]) -> RadiativeParams {
        RadiativeParams::lightlike(0.0)
    }
}

#[derive(Clone)]
struct WaveEquation {
    speed: [f64; 2],
}

impl Function<2> for WaveEquation {
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: ImageRef,
        mut output: ImageMut,
    ) -> Result<(), Infallible> {
        let input = input.channel(0);
        let output = output.channel_mut(0);

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);

            let dr = engine.derivative(input, 0, vertex);
            let dz = engine.derivative(input, 1, vertex);

            output[index] = -dr * self.speed[0] - dz * self.speed[1];
        }

        Ok(())
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

pub fn main() -> eyre::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    std::fs::create_dir_all("output/waves")?;

    log::info!("Intializing Mesh.");

    // Generate initial mesh
    let mut mesh = Mesh::new(
        HyperBox::from_aabb([-10., -10.], [10., 10.]),
        6,
        3,
        FaceArray::splat(BoundaryClass::OneSided),
    );
    mesh.refine_global();
    // Allocate space for system
    let mut system = Image::new(1, 0);

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

        log::trace!("Projecting System");

        mesh.project(4, profile, system.channel_mut(0));
        mesh.fill_boundary(ORDER, WaveConditions, system.as_mut());

        log::trace!("Flagging Wavelets");

        mesh.flag_wavelets(4, LOWER, UPPER, system.as_ref());
        mesh.limit_level_range_flags(1, 10);

        log::trace!("Balancing Flags");

        mesh.balance_flags();

        // Save data to file.
        let mut flags = vec![0; mesh.num_nodes()];
        mesh.flags_debug(&mut flags);

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_field("Wave", system.storage());
        checkpoint.save_int_field("Flags", &flags);
        checkpoint.export_vtu(
            format!("output/waves/initial{i}.vtu"),
            ExportVtuConfig {
                title: "Initial Wave Mesh".to_string(),
                ghost: false,
                stride: ExportStride::PerCell,
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

            // let mut string = String::new();
            // mesh.write_debug(&mut string);
            // std::fs::write(PathBuf::from(format!("output/waves/debug{i}.txt")), string)?;

            // let mut checkpoint = Checkpoint::default();
            // checkpoint.attach_mesh(&mesh);
            // checkpoint.save_int_field("Flags", &vec![0; mesh.num_nodes()]);
            // checkpoint.export_vtu(
            //     format!("output/waves/debug{i}.vtu"),
            //     ExportVtuConfig {
            //         title: "Initial Wave Mesh".to_string(),
            //         ghost: false,
            //         stride: ExportStride::PerCell,
            //     },
            // )?;

            continue;
        } else {
            log::info!("Regridded within range in {} iterations.", i + 1);
            break;
        }
    }

    log::info!("Evolving wave forwards");

    // Allocate vectors
    let mut tmp = Image::new(1, 0);
    let mut exact = Image::new(1, 0);
    let mut error = Image::new(1, 0);

    // Integrate
    let mut integrator = Integrator::new(Method::RK4KO6(0.5));
    let mut time = 0.0;
    let mut step = 0;

    let mut time_since_save = 0.0;
    let mut save_step = 0;

    let mut steps_since_regrid = 0;

    while step < MAX_STEPS && time < MAX_TIME {
        assert!(system.num_nodes() == mesh.num_nodes());

        // Get step size
        let h = mesh.min_spacing() * CFL;

        // Resize vectors
        tmp.resize(mesh.num_nodes());
        exact.resize(mesh.num_nodes());
        error.resize(mesh.num_nodes());

        mesh.project(
            4,
            AnalyticSolution { speed: SPEED, time },
            exact.channel_mut(0),
        );

        // Fill boundaries
        mesh.fill_boundary(ORDER, WaveConditions, system.as_mut());
        mesh.fill_boundary(ORDER, WaveConditions, exact.as_mut());

        for i in 0..mesh.num_nodes() {
            error.storage_mut()[i] = exact.storage()[i] - system.storage()[i];
        }

        if steps_since_regrid > REGRID_SKIP {
            steps_since_regrid = 0;

            mesh.fill_boundary(ORDER, WaveConditions, system.as_mut());
            mesh.flag_wavelets(4, LOWER, UPPER, system.as_ref());
            mesh.limit_level_range_flags(1, 10);

            mesh.balance_flags();

            let num_refine = mesh.num_refine_cells();
            let num_coarsen = mesh.num_coarsen_cells();

            mesh.regrid();

            log::info!(
                "Regrided Mesh at time: {time:.5}, Max Level {}, {} R, {} C",
                mesh.num_levels(),
                num_refine,
                num_coarsen,
            );

            // Copy system into tmp.
            tmp.storage_mut().copy_from_slice(system.storage());

            system.resize(mesh.num_nodes());
            mesh.transfer_system(ORDER, tmp.as_ref(), system.as_mut());

            continue;
        }

        mesh.fill_boundary(ORDER, WaveConditions, system.as_mut());

        if time_since_save >= SAVE_CHECKPOINT || FORCE_SAVE {
            time_since_save = 0.0;

            log::info!(
                "Saving Checkpoint {save_step}
    Time: {time:.5}, Step: {h:.8}
    Nodes: {}",
                mesh.num_nodes()
            );
            // Output current system to disk
            let mut checkpoint = Checkpoint::default();
            checkpoint.attach_mesh(&mesh);
            checkpoint.save_field("Wave", system.storage());
            checkpoint.save_field("Analytic", exact.storage());
            checkpoint.save_field("Error", error.storage());
            checkpoint.export_vtu(
                format!("output/waves/evolution{save_step}.vtu"),
                ExportVtuConfig {
                    title: "evbrill".to_string(),
                    ghost: false,
                    stride: ExportStride::PerCell,
                },
            )?;

            save_step += 1;
        }

        // Compute step with dissipation
        integrator
            .step(
                &mut mesh,
                ORDER,
                WaveConditions,
                WaveEquation { speed: SPEED },
                h,
                system.as_mut(),
            )
            .unwrap();

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;
    }

    Ok(())
}
