//! An executable for creating general initial data for numerical relativity simulations in 2D.

use core::f64;
use std::{path::PathBuf, process::ExitCode};

use aeon::{
    kernel::{node_from_vertex, Interpolation},
    mesh::Gaussian,
    prelude::*,
    solver::{Integrator, Method},
    system::SystemConditions,
};
use anyhow::{anyhow, Context, Result};
use clap::{Arg, Command};
use serde::{Deserialize, Serialize};

/// System for storing all fields necessary for axisymmetric evolution.
#[derive(Clone, Serialize, Deserialize)]
pub struct Fields;

impl System for Fields {
    const NAME: &'static str = "Fields";

    type Label = Field;

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        [Field::Phi, Field::Pi, Field::Conformal, Field::Lapse].into_iter()
    }

    fn count(&self) -> usize {
        4
    }

    fn label_from_index(&self, index: usize) -> Self::Label {
        [Field::Phi, Field::Pi, Field::Conformal, Field::Lapse][index]
    }

    fn label_index(&self, label: Self::Label) -> usize {
        match label {
            Field::Phi => 0,
            Field::Pi => 1,
            Field::Conformal => 2,
            Field::Lapse => 3,
        }
    }

    fn label_name(&self, label: Self::Label) -> String {
        match label {
            Field::Phi => "Phi",
            Field::Pi => "Pi",
            Field::Conformal => "Conformal",
            Field::Lapse => "Lapse",
        }
        .to_string()
    }
}

/// Label for indexing fields in `Fields`.
#[derive(Clone, Copy)]
pub enum Field {
    Phi,
    Pi,
    Conformal,
    Lapse,
}

#[derive(Clone)]
struct FieldConditions;

impl SystemConditions<1> for FieldConditions {
    type System = Fields;

    fn parity(&self, label: <Self::System as System>::Label, _face: Face<1>) -> bool {
        match label {
            Field::Phi => false,
            Field::Pi | Field::Conformal | Field::Lapse => true,
        }
    }

    fn radiative(
        &self,
        _label: <Self::System as System>::Label,
        _position: [f64; 1],
    ) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
}

#[derive(Clone)]
struct SymCondition;

impl Condition<1> for SymCondition {
    fn parity(&self, _face: Face<1>) -> bool {
        true
    }

    fn radiative(&self, _position: [f64; 1]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
}

#[derive(Clone)]
struct AntiSymCondition;

impl Condition<1> for AntiSymCondition {
    fn parity(&self, _face: Face<1>) -> bool {
        false
    }

    fn radiative(&self, _position: [f64; 1]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
}

const MAX_LEVELS: usize = 25;
const MAX_TIME_STEPS: usize = 300_000;
const MAX_PROPER_TIME: f64 = 10.0;
const CELL_WIDTH: usize = 6;
const GHOST: usize = 3;
const SIGMA: f64 = 5.35;
const RADIUS: f64 = 40.0;
const REFINE_GLOBAL: usize = 2;
const MAX_NODES: usize = 10_000_000;
const MAX_ERROR_TOLERANCE: f64 = 1e-7;
const MIN_ERROR_TOLERANCE: f64 = 1e-10;
const DISSIPATION: f64 = 0.5;
const SAVE_INTERVAL: f64 = 0.1;
const REGRID_FLAG_INTERVAL: usize = 20;
const CFL: f64 = 0.1;

fn solve_constraints(mesh: &mut Mesh<1>, system: SystemSliceMut<Fields>) {
    let shared = system.into_shared();
    // Unpack individual fields
    let phi = unsafe { shared.field_mut(Field::Phi) };
    let pi = unsafe { shared.field_mut(Field::Pi) };

    mesh.fill_boundary(Order::<4>, ScalarConditions(AntiSymCondition), phi.into());
    mesh.fill_boundary(Order::<4>, ScalarConditions(SymCondition), pi.into());

    let conformal = unsafe { shared.field_mut(Field::Conformal) };
    let lapse = unsafe { shared.field_mut(Field::Lapse) };

    // Perform radial quadrature for conformal factor
    let mut conformal_prev = 1.0;

    for block in 0..mesh.num_blocks() {
        let space = mesh.block_space(block);
        let nodes = mesh.block_nodes(block);
        let bounds = mesh.block_bounds(block);
        let spacing = mesh.block_spacing(block);
        let cell_size = space.cell_size()[0];

        let phi = &phi[nodes.clone()];
        let pi = &pi[nodes.clone()];
        let conformal = &mut conformal[nodes.clone()];

        debug_assert!(phi.len() == space.num_nodes());

        let derivative = |r: f64, a: f64, phi: f64, pi: f64| {
            if r < 10e-15 || r.is_nan() || r.is_infinite() {
                return 0.0;
            }

            2.0 * f64::consts::PI * r * a * (phi * phi + pi * pi) - a * (a * a - 1.0) / (2.0 * r)
        };

        conformal[space.index_from_vertex([0])] = conformal_prev;

        for vertex in 0..cell_size {
            let index = space.index_from_vertex([vertex]);
            let [r] = space.position([vertex as isize], bounds);
            // Intermediate step interpolation
            let r_half = r + spacing / 2.0;
            let phi_half = space.prolong(Interpolation::<4>, [(2 * vertex + 1) as isize], phi);
            let pi_half = space.prolong(Interpolation::<4>, [(2 * vertex + 1) as isize], pi);
            let r_next = r + spacing;
            let phi_next = phi[index + 1];
            let pi_next = pi[index + 1];
            let phi = phi[index];
            let pi = pi[index];
            let a = conformal[index];

            let k1 = derivative(r, a, phi, pi);
            let k2 = derivative(r_half, a + k1 * spacing / 2.0, phi_half, pi_half);
            let k3 = derivative(r_half, a + k2 * spacing / 2.0, phi_half, pi_half);
            let k4 = derivative(r_next, a + k3 * spacing, phi_next, pi_next);

            conformal[index + 1] =
                conformal[index] + spacing / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }

        conformal_prev = conformal[space.index_from_vertex([cell_size])];
    }
    // Fill ghost nodes.
    mesh.fill_boundary(Order::<4>, ScalarConditions(SymCondition), conformal.into());

    // Perform radial quadrature for lapse
    let mut lapse_prev = 1.0 / conformal_prev;

    for block in (0..mesh.num_blocks()).rev() {
        let space = mesh.block_space(block);
        let nodes = mesh.block_nodes(block);
        let bounds = mesh.block_bounds(block);
        let spacing = mesh.block_spacing(block);
        let cell_size = space.cell_size()[0];

        let phi = &phi[nodes.clone()];
        let pi = &pi[nodes.clone()];
        let conformal = &conformal[nodes.clone()];
        let lapse = &mut lapse[nodes.clone()];

        let derivative = |r: f64, alpha: f64, a: f64, phi: f64, pi: f64| {
            if r < 1e-15 || r.is_nan() || r.is_infinite() {
                return 0.0;
            }

            2.0 * f64::consts::PI * r * alpha * (phi * phi + pi * pi)
                + alpha * (a * a - 1.0) / (2.0 * r)
        };

        lapse[space.index_from_vertex([cell_size])] = lapse_prev;

        for vertex in (0..cell_size).rev().map(|i| i + 1) {
            let index = space.index_from_vertex([vertex]);
            let [r] = space.position([vertex as isize], bounds);
            // Intermediate step interpolation
            let r_half = r - spacing / 2.0;
            let phi_half = space.prolong(Interpolation::<4>, [(2 * vertex - 1) as isize], phi);
            let pi_half = space.prolong(Interpolation::<4>, [(2 * vertex - 1) as isize], pi);
            let a_half = space.prolong(Interpolation::<4>, [(2 * vertex - 1) as isize], conformal);
            let r_next = r - spacing;
            let phi_next = phi[index - 1];
            let pi_next = pi[index - 1];
            let a_next = conformal[index - 1];
            let phi = phi[index];
            let pi = pi[index];
            let a = conformal[index];
            let alpha = lapse[index];

            let k1 = derivative(r, alpha, a, phi, pi);
            let k2 = derivative(
                r_half,
                alpha - k1 * spacing / 2.0,
                a_half,
                phi_half,
                pi_half,
            );
            let k3 = derivative(
                r_half,
                alpha - k2 * spacing / 2.0,
                a_half,
                phi_half,
                pi_half,
            );
            let k4 = derivative(r_next, alpha - k3 * spacing, a_next, phi_next, pi_next);

            lapse[index - 1] = lapse[index] - spacing / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }

        lapse_prev = lapse[space.index_from_vertex([0])];
    }

    // Fill lapse ghost nodes
    mesh.fill_boundary(Order::<4>, ScalarConditions(SymCondition), lapse.into());
}

fn generate_initial_scalar_field(mesh: &mut Mesh<1>, amplitude: f64) -> Vec<f64> {
    let mut scalar_field = vec![0.0; mesh.num_nodes()];

    mesh.project(
        Order::<4>,
        Gaussian {
            amplitude,
            sigma: [SIGMA],
            center: [0.0],
        },
        &mut scalar_field,
    );
    mesh.fill_boundary(
        Order::<4>,
        ScalarConditions(SymCondition),
        (&mut scalar_field).into(),
    );

    scalar_field
}

struct InitialData;

impl Function<1> for InitialData {
    type Input = Scalar;
    type Output = Fields;

    fn evaluate(
        &self,
        engine: impl Engine<1>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
        let scalar_field = input.into_scalar();

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);

            output.field_mut(Field::Conformal)[index] = 1.0;
            output.field_mut(Field::Lapse)[index] = 1.0;
            output.field_mut(Field::Phi)[index] = engine.derivative(scalar_field, 0, vertex);
            output.field_mut(Field::Pi)[index] = 0.0;
        }
    }
}

#[derive(Clone)]
struct Derivs;

impl Function<1> for Derivs {
    fn preprocess(&self, mesh: &mut Mesh<1>, input: SystemSliceMut<Self::Input>) {
        solve_constraints(mesh, input);
    }

    fn evaluate(
        &self,
        engine: impl Engine<1>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
        let a = input.field(Field::Conformal);
        let alpha = input.field(Field::Lapse);
        let phi = input.field(Field::Phi);
        let pi = input.field(Field::Pi);

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            let [r] = engine.position(vertex);
            let a_r = engine.derivative(a, 0, vertex);
            let alpha_r = engine.derivative(alpha, 0, vertex);
            let phi_r = engine.derivative(phi, 0, vertex);
            let pi_r = engine.derivative(pi, 0, vertex);
            let alpha = alpha[index];
            let a = a[index];
            let phi = phi[index];
            let pi = pi[index];

            let phi_t = pi_r * alpha / a + alpha_r / a * pi - alpha / (a * a) * pi * a_r;
            let mut pi_t = phi_r * alpha / a + alpha_r / a * phi - alpha / (a * a) * phi * a_r;
            if r < 1e-15 {
                pi_t += alpha / a * phi_r * 2.0;
            } else {
                pi_t += alpha / a * phi * 2.0 / r;
            }

            output.field_mut(Field::Phi)[index] = phi_t;
            output.field_mut(Field::Pi)[index] = pi_t;

            output.field_mut(Field::Conformal)[index] = 0.0;
            output.field_mut(Field::Lapse)[index] = 0.0;
        }
    }

    type Input = Fields;
    type Output = Fields;
}

fn evolve() -> Result<()> {
    // Load configuration
    let matches = Command::new("evsphere")
        .about("A program for generating initial data for numerical relativity using hyperbolic relaxation.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .arg(
            Arg::new("amplitude").short('a').long("amplitude").default_value("1.0").value_name("NUMBER")
        ).arg(Arg::new("output").required(true).help("Output directory").value_name("FILE"))
        .version("0.1.0").get_matches();

    let amplitude = matches
        .get_one::<String>("amplitude")
        .ok_or(anyhow!("Could not find amplitude argument"))?
        .parse::<f64>()
        .map_err(|_| anyhow!("Failed parse amplitude argument"))?
        .clone();

    let output = PathBuf::from(
        matches
            .get_one::<String>("output")
            .ok_or(anyhow!("Failed parse path argument"))?
            .clone(),
    );

    // Build enviornment logger.
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();
    // Find currect working directory
    let dir = std::env::current_dir().context("Failed to find current working directory")?;
    let absolute = if output.is_absolute() {
        output
    } else {
        dir.join(output)
    };

    // Log Header data.
    // log::info!("Simulation name: {}", &config.name);
    // log::info!("Logging Level: {} ", level);
    log::info!(
        "Output Directory: {}",
        absolute
            .to_str()
            .ok_or(anyhow!("Failed to find absolute output directory"))?
    );

    // Create output dir.
    std::fs::create_dir_all(&absolute)?;

    log::info!("A: {:.5e}, sigma: {:.5e}", amplitude, SIGMA);

    // Run brill simulation
    log::trace!(
        "Building Mesh with Radius {}, Cell Width {}, Ghost Nodes {}",
        RADIUS,
        CELL_WIDTH,
        GHOST
    );

    let mut mesh = Mesh::new(
        Rectangle {
            size: [RADIUS],
            origin: [0.0],
        },
        CELL_WIDTH,
        GHOST,
    );
    mesh.set_face_boundary(Face::negative(0), BoundaryKind::Parity);
    mesh.set_face_boundary(Face::positive(0), BoundaryKind::Radiative);

    log::trace!("Refining mesh globally {} times", REFINE_GLOBAL);

    for _ in 0..REFINE_GLOBAL {
        mesh.refine_global();
    }

    let mut system = SystemVec::new(Fields);

    loop {
        system.resize(mesh.num_nodes());

        // Set initial data for scalar field.
        let scalar_field = generate_initial_scalar_field(&mut mesh, amplitude);

        // Fill system using scalar field.
        mesh.evaluate(
            Order::<4>,
            InitialData,
            (&scalar_field).into(),
            system.as_mut_slice(),
        );

        // Solve for conformal and lapse
        solve_constraints(&mut mesh, system.as_mut_slice());

        mesh.fill_boundary(Order::<4>, FieldConditions, system.as_mut_slice());

        let l2_norm = mesh.l2_norm(system.as_slice());

        log::info!("Scalar Field Norm {}", l2_norm);

        let mut checkpoint = SystemCheckpoint::default();
        checkpoint.save_system_ser(system.as_slice());

        mesh.export_vtu(
            absolute.join(format!("initial{}.vtu", mesh.max_level())),
            &checkpoint,
            ExportVtuConfig {
                title: "Massless Scalar Field Initial Data".to_string(),
                ghost: false,
                stride: 1,
            },
        )?;

        if mesh.max_level() >= MAX_LEVELS || mesh.num_nodes() >= MAX_NODES {
            log::error!(
                "Failed to solve initial data, level: {}, nodes: {}",
                mesh.max_level(),
                mesh.num_nodes()
            );
            break;
            // return Err(anyhow!("failed to refine within perscribed limits"));
        }

        mesh.flag_wavelets(4, 0.0, MAX_ERROR_TOLERANCE, system.as_slice());
        mesh.balance_flags();

        if !mesh.requires_regridding() {
            log::trace!(
                "Sucessfully refined mesh to give accuracy: {:.5e}",
                MAX_ERROR_TOLERANCE
            );
            break;
        } else {
            log::trace!(
                "Regridding mesh from level {} to {}",
                mesh.max_level(),
                mesh.max_level() + 1
            );

            mesh.regrid();
        }
    }

    let mut checkpoint = SystemCheckpoint::default();
    checkpoint.save_system_ser(system.as_slice());

    mesh.export_vtu(
        absolute.join("initial.vtu"),
        &checkpoint,
        ExportVtuConfig {
            title: "Massless Scalar Field Initial".to_string(),
            ghost: false,
            stride: 1,
        },
    )?;

    // mesh.export_dat(absolute.join(format!("{}.dat", config.name)), &checkpoint)?;

    // ****************************************
    // Run evolution

    let mut integrator = Integrator::new(Method::RK4KO6(DISSIPATION));
    let mut time = 0.0;
    let mut step = 0;

    let mut proper_time = 0.0;

    let mut save_step = 0;
    let mut steps_since_regrid = 0;
    let mut time_since_save = 0.0;

    while proper_time < MAX_PROPER_TIME {
        assert!(system.len() == mesh.num_nodes());
        mesh.fill_boundary(Order::<4>, FieldConditions, system.as_mut_slice());

        // Check Norm
        let norm = mesh.l2_norm(system.as_slice());

        if norm.is_nan() || norm >= 1e60 {
            log::trace!("Evolution collapses, norm: {}", norm);
            return Err(anyhow!(
                "exceded max allotted steps for evolution: {}",
                step
            ));
        }

        if step >= MAX_TIME_STEPS {
            log::error!("Evolution exceded maximum allocated steps: {}", step);
            return Err(anyhow!(
                "exceded max allotted steps for evolution: {}",
                step
            ));
        }

        if mesh.num_nodes() >= MAX_NODES {
            log::error!(
                "Evolution exceded maximum allocated nodes: {}",
                mesh.num_nodes()
            );
            return Err(anyhow!(
                "exceded max allotted nodes for evolution: {}",
                mesh.num_nodes()
            ));
        }

        if mesh.max_level() >= MAX_LEVELS {
            log::trace!(
                "Evolution collapses, Reached maximum allowed level of refinement: {}",
                mesh.max_level()
            );
            return Err(anyhow!(
                "reached maximum allowed level of refinement: {}",
                mesh.max_level()
            ));
        }

        let h = mesh.min_spacing() * CFL;

        if steps_since_regrid > REGRID_FLAG_INTERVAL {
            steps_since_regrid = 0;

            mesh.flag_wavelets(
                4,
                MIN_ERROR_TOLERANCE,
                MAX_ERROR_TOLERANCE,
                system.as_slice(),
            );
            mesh.balance_flags();

            // let num_refine = mesh.num_refine_cells();
            // let num_coarsen = mesh.num_coarsen_cells();
            mesh.regrid();

            // log::trace!(
            //     "Regrided Mesh at time: {time:.5}, Max Level {}, {} R, {} C",
            //     mesh.max_level(),
            //     num_refine,
            //     num_coarsen,
            // );

            log::trace!(
                "Regrided Mesh at time: {time:.5}, Max Level {}, Num Nodes {}",
                mesh.max_level(),
                mesh.num_nodes(),
            );

            // Copy system into tmp scratch space (provieded by dissipation).
            let scratch = integrator.scratch(system.contigious().len());
            scratch.copy_from_slice(system.contigious());
            system.resize(mesh.num_nodes());
            mesh.transfer_system(
                Order::<4>,
                SystemSlice::from_contiguous(&scratch, &Fields),
                system.as_mut_slice(),
            );

            continue;
        }

        if time_since_save >= SAVE_INTERVAL {
            time_since_save -= SAVE_INTERVAL;

            log::trace!(
                "Saving Checkpoint {save_step}, Time: {time:.5}, Dilated Time: {proper_time:.5}, Step: {step}, Norm: {norm:.5e}, Nodes: {}",
                mesh.num_nodes()
            );

            // Output current system to disk
            let mut systems = SystemCheckpoint::default();
            systems.save_system_ser(system.as_slice());

            mesh.export_vtu(
                absolute.join(format!("evolve_{save_step}.vtu")),
                &systems,
                ExportVtuConfig {
                    title: "Masslesss Scalar Field Evolution".to_string(),
                    ghost: false,
                    stride: 1,
                },
            )?;

            save_step += 1;
        }

        // Compute step
        integrator.step(
            &mut mesh,
            Order::<4>,
            FieldConditions,
            Derivs,
            h,
            system.as_mut_slice(),
        );

        let lapse = system.field(Field::Lapse)[0];

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;

        proper_time += h * lapse;
    }

    Ok(())
}

fn main() -> ExitCode {
    match evolve() {
        Ok(_) => ExitCode::SUCCESS,
        Err(err) => {
            if log::log_enabled!(log::Level::Error) {
                log::error!("{:?}", err);
            } else {
                eprintln!("{:?}", err);
            }
            ExitCode::FAILURE
        }
    }
}
