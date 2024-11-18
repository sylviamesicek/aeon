#![allow(mixed_script_confusables)]

use aeon::basis::RadiativeParams;
use aeon::fd::{ExportVtuConfig, Mesh, SystemCheckpoint, SystemCondition};
use aeon::prelude::*;
use aeon::system::field_count;
use reborrow::{Reborrow, ReborrowMut};

pub mod shared;
pub mod tensor;

use shared::HyperbolicSystem;

pub enum Equations {
    /// Equations using `aeon_tensor` library for tensor manipulation.
    Tensor,
}

const EQUATIONS: Equations = Equations::Tensor;

const MAX_TIME: f64 = 12.0;
const MAX_STEPS: usize = 50000;
const MAX_LEVEL: usize = 14;

const CFL: f64 = 0.1;
const ORDER: Order<4> = Order::<4>;
const DISS_ORDER: Order<6> = Order::<6>;

const SAVE_CHECKPOINT: f64 = 0.04;
const FORCE_SAVE: bool = false;
const REGRID_SKIP: usize = 10;

const LOWER: f64 = 1e-9;
const UPPER: f64 = 1e-6;

/// Initial data in Rinne's hyperbolic variables.
#[derive(Clone, SystemLabel)]
pub enum Rinne {
    Conformal,
    Seed,
    Phi,
}

#[derive(Clone, SystemLabel)]
pub enum Dynamic {
    Grr,
    Grz,
    Gzz,
    S,
    Krr,
    Krz,
    Kzz,
    Y,
    Theta,
    Zr,
    Zz,
    Lapse,
    Shiftr,
    Shiftz,

    Phi,
    Pi,
}

#[derive(Clone)]
pub struct Quadrant;

impl Boundary<2> for Quadrant {
    fn kind(&self, face: Face<2>) -> BoundaryKind {
        match face.side {
            true => BoundaryKind::Radiative,
            false => BoundaryKind::Parity,
        }
    }
}

#[derive(Clone)]
pub struct DynamicConditions;

impl Conditions<2> for DynamicConditions {
    type System = Dynamic;

    fn parity(&self, field: Self::System, face: Face<2>) -> bool {
        let axes = match field {
            Dynamic::Grr | Dynamic::Krr => [true, true],
            Dynamic::Grz | Dynamic::Krz => [false, false],
            Dynamic::Gzz | Dynamic::Kzz => [true, true],
            Dynamic::S | Dynamic::Y => [false, true],

            Dynamic::Theta | Dynamic::Lapse => [true, true],
            Dynamic::Zr | Dynamic::Shiftr => [false, true],
            Dynamic::Zz | Dynamic::Shiftz => [true, false],

            Dynamic::Phi | Dynamic::Pi => [true, true],
        };
        axes[face.axis]
    }

    fn radiative(
        &self,
        field: Self::System,
        _position: [f64; 2],
        _spacing: f64,
    ) -> RadiativeParams {
        match field {
            Dynamic::Grr | Dynamic::Gzz | Dynamic::Lapse => RadiativeParams::lightlike(1.0),
            _ => RadiativeParams::lightlike(0.0),
        }
    }
}

pub fn condition(field: Dynamic) -> SystemCondition<Dynamic, DynamicConditions> {
    SystemCondition::new(field, DynamicConditions)
}

#[derive(Clone)]
pub struct DynamicDerivs {
    pub mass: f64,
}

impl Function<2> for DynamicDerivs {
    type Input = Dynamic;
    type Output = Dynamic;

    fn evaluate(&self, engine: &impl Engine<2, Self::Input>) -> SystemValue<Self::Output> {
        let [rho, z] = engine.position();

        macro_rules! derivatives {
            ($field:ident, $value:ident, $dr:ident, $dz:ident) => {
                let $value = engine.value(Dynamic::$field);
                let $dr = engine.derivative(Dynamic::$field, 0);
                let $dz = engine.derivative(Dynamic::$field, 1);
            };
        }

        macro_rules! second_derivatives {
            ($field:ident, $value:ident, $dr:ident, $dz:ident, $drr:ident, $drz:ident, $dzz:ident) => {
                let $value = engine.value(Dynamic::$field);
                let $dr = engine.derivative(Dynamic::$field, 0);
                let $dz = engine.derivative(Dynamic::$field, 1);

                let $drr = engine.second_derivative(Dynamic::$field, 0, 0);
                let $drz = engine.second_derivative(Dynamic::$field, 0, 1);
                let $dzz = engine.second_derivative(Dynamic::$field, 1, 1);
            };
        }

        // Metric
        second_derivatives!(Grr, grr, grr_r, grr_z, grr_rr, grr_rz, grr_zz);
        second_derivatives!(Gzz, gzz, gzz_r, gzz_z, gzz_rr, gzz_rz, gzz_zz);
        second_derivatives!(Grz, grz, grz_r, grz_z, grz_rr, grz_rz, grz_zz);

        // S
        second_derivatives!(S, s, s_r, s_z, s_rr, s_rz, s_zz);

        // K
        derivatives!(Krr, krr, krr_r, krr_z);
        derivatives!(Kzz, kzz, kzz_r, kzz_z);
        derivatives!(Krz, krz, krz_r, krz_z);

        // Y
        derivatives!(Y, y, y_r, y_z);

        // Gauge
        second_derivatives!(Lapse, lapse, lapse_r, lapse_z, lapse_rr, lapse_rz, lapse_zz);
        derivatives!(Shiftr, shiftr, shiftr_r, shiftr_z);
        derivatives!(Shiftz, shiftz, shiftz_r, shiftz_z);

        // Constraints
        derivatives!(Theta, theta, theta_r, theta_z);
        derivatives!(Zr, zr, zr_r, zr_z);
        derivatives!(Zz, zz, zz_r, zz_z);

        second_derivatives!(Phi, phi, phi_r, phi_z, phi_rr, phi_rz, phi_zz);
        derivatives!(Pi, pi, pi_r, pi_z);

        let system = HyperbolicSystem {
            grr,
            grr_r,
            grr_z,
            grr_rr,
            grr_rz,
            grr_zz,
            grz,
            grz_r,
            grz_z,
            grz_rr,
            grz_rz,
            grz_zz,
            gzz,
            gzz_r,
            gzz_z,
            gzz_rr,
            gzz_rz,
            gzz_zz,
            s,
            s_r,
            s_z,
            s_rr,
            s_rz,
            s_zz,

            krr,
            krr_r,
            krr_z,
            krz,
            krz_r,
            krz_z,
            kzz,
            kzz_r,
            kzz_z,
            y,
            y_r,
            y_z,

            theta,
            theta_r,
            theta_z,
            zr,
            zr_r,
            zr_z,
            zz,
            zz_r,
            zz_z,

            lapse,
            lapse_r,
            lapse_z,
            lapse_rr,
            lapse_rz,
            lapse_zz,
            shiftr,
            shiftr_r,
            shiftr_z,
            shiftz,
            shiftz_r,
            shiftz_z,

            phi,
            phi_r,
            phi_z,
            phi_rr,
            phi_rz,
            phi_zz,

            pi,
            pi_r,
            pi_z,

            mass: self.mass,
        };

        let derivs = match EQUATIONS {
            Equations::Tensor => tensor::hyperbolic(system, [rho, z]),
        };

        let mut result = SystemValue::default();

        result.set_field(Dynamic::Grr, derivs.grr_t);
        result.set_field(Dynamic::Grz, derivs.grz_t);
        result.set_field(Dynamic::Gzz, derivs.gzz_t);
        result.set_field(Dynamic::S, derivs.s_t);

        result.set_field(Dynamic::Krr, derivs.krr_t);
        result.set_field(Dynamic::Krz, derivs.krz_t);
        result.set_field(Dynamic::Kzz, derivs.kzz_t);
        result.set_field(Dynamic::Y, derivs.y_t);

        result.set_field(Dynamic::Theta, derivs.theta_t);
        result.set_field(Dynamic::Zr, derivs.zr_t);
        result.set_field(Dynamic::Zz, derivs.zz_t);

        result.set_field(Dynamic::Lapse, derivs.lapse_t);
        result.set_field(Dynamic::Shiftr, derivs.shiftr_t);
        result.set_field(Dynamic::Shiftz, derivs.shiftz_t);

        result.set_field(Dynamic::Phi, derivs.phi_t);
        result.set_field(Dynamic::Pi, derivs.pi_t);

        result
    }
}

pub struct DynamicOde<'a> {
    mesh: &'a mut Mesh<2>,
    mass: f64,
}

impl<'a> Ode for DynamicOde<'a> {
    fn dim(&self) -> usize {
        field_count::<Dynamic>() * self.mesh.num_nodes()
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        self.mesh.fill_boundary(
            ORDER,
            Quadrant,
            DynamicConditions,
            SystemSliceMut::from_contiguous(system),
        );
    }

    fn derivative(&mut self, system: &[f64], result: &mut [f64]) {
        let src = SystemSlice::from_contiguous(system);
        let mut dest = SystemSliceMut::from_contiguous(result);

        self.mesh.evaluate(
            ORDER,
            Quadrant,
            DynamicDerivs { mass: self.mass },
            src.rb(),
            dest.rb_mut(),
        );

        self.mesh
            .weak_boundary(ORDER, Quadrant, DynamicConditions, src.rb(), dest.rb_mut());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    // Create output directory.
    std::fs::create_dir_all("output/evscalar").expect("Unable to create evscalar directory.");

    // Build discretization
    let mut mesh = Mesh::default();
    let mut checkpoint = SystemCheckpoint::default();

    log::info!("Importing IdScalar data");

    mesh.import_dat("output/ellipticmasslesssec.dat", &mut checkpoint)
        .expect("Unable to load initial data");

    // Read initial data
    let mut initial = SystemVec::<Rinne>::new();
    checkpoint.load_system(&mut initial);

    let mass: f64 = checkpoint.parse_meta("MASS")?;

    // Setup dynamic variables
    let mut dynamic = SystemVec::with_length(mesh.num_nodes());

    // Metric
    dynamic
        .field_mut(Dynamic::Grr)
        .copy_from_slice(initial.field(Rinne::Conformal));
    dynamic
        .field_mut(Dynamic::Gzz)
        .copy_from_slice(initial.field(Rinne::Conformal));
    dynamic.field_mut(Dynamic::Grz).fill(0.0);
    // S
    dynamic
        .field_mut(Dynamic::S)
        .copy_from_slice(initial.field(Rinne::Seed));
    // Extrinsic Curvature
    dynamic.field_mut(Dynamic::Krr).fill(0.0);
    dynamic.field_mut(Dynamic::Kzz).fill(0.0);
    dynamic.field_mut(Dynamic::Krz).fill(0.0);
    // Y
    dynamic.field_mut(Dynamic::Y).fill(0.0);
    // Constraint
    dynamic.field_mut(Dynamic::Theta).fill(0.0);
    dynamic.field_mut(Dynamic::Zr).fill(0.0);
    dynamic.field_mut(Dynamic::Zz).fill(0.0);
    // Gauge
    dynamic.field_mut(Dynamic::Lapse).fill(1.0);
    dynamic.field_mut(Dynamic::Shiftr).fill(0.0);
    dynamic.field_mut(Dynamic::Shiftz).fill(0.0);

    dynamic
        .field_mut(Dynamic::Phi)
        .copy_from_slice(initial.field(Rinne::Phi));
    dynamic.field_mut(Dynamic::Pi).fill(0.0);

    // Fill ghost nodes
    mesh.fill_boundary(ORDER, Quadrant, DynamicConditions, dynamic.as_mut_slice());

    // Allocate vectors
    // let mut derivs = SystemVec::<Dynamic>::new();
    let mut update = SystemVec::<Dynamic>::new();
    let mut dissipation = SystemVec::new();

    // Integrate
    let mut integrator = Rk4::new();
    let mut time = 0.0;
    let mut step = 0;

    let mut time_since_save = 0.0;
    let mut save_step = 0;

    let mut steps_since_regrid = 0;

    // let mut errors = Vec::new();

    while step < MAX_STEPS && time < MAX_TIME {
        assert!(dynamic.len() == mesh.num_nodes());
        // Fill boundaries
        mesh.fill_boundary(ORDER, Quadrant, DynamicConditions, dynamic.as_mut_slice());

        // Check Norm
        let norm = mesh.l2_norm(dynamic.as_slice());
        if norm.is_nan() {
            log::warn!("Norm is NaN");
            break;
        }

        // Get step size
        let h = mesh.min_spacing() * CFL;

        // Resize vectors
        update.resize(mesh.num_nodes());
        dissipation.resize(mesh.num_nodes());

        if steps_since_regrid > REGRID_SKIP {
            steps_since_regrid = 0;

            log::info!("Regridding Mesh at time: {time:.5}");
            mesh.flag_wavelets(4, LOWER, UPPER, Quadrant, dynamic.as_slice());
            mesh.set_regrid_level_limit(MAX_LEVEL);
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

            // let l2_norm = mesh.l2_norm(dynamic.field(Dynamic::Theta).into());
            // let max_norm = mesh.max_norm(dynamic.field(Dynamic::Theta).into());

            // let space = mesh.block_space(0);
            // let index = space.index_from_vertex([0; 2]);

            // let origin = dynamic.field(Dynamic::Theta)[index];

            // errors.push((time, l2_norm, max_norm, origin));

            // let mut error_csv = String::new();

            // for (time, l2_norm, max, origin) in errors.iter() {
            //     error_csv.write_fmt(format_args!("{time}, {l2_norm}, {max}, {origin},\n"))?;
            // }

            // let mut file = File::create("output/massless_errors.txt")?;
            // file.write_all(error_csv.as_bytes())?;

            // Copy system into tmp scratch space (provieded by dissipation).
            dissipation
                .contigious_mut()
                .copy_from_slice(dynamic.contigious());
            dynamic.resize(mesh.num_nodes());
            mesh.transfer_system(
                ORDER,
                Quadrant,
                dissipation.as_slice(),
                dynamic.as_mut_slice(),
            );

            continue;
        }

        if time_since_save >= SAVE_CHECKPOINT || FORCE_SAVE {
            time_since_save -= SAVE_CHECKPOINT;

            log::info!(
                "Saving Checkpoint {save_step}
    Time: {time:.5}, Step: {h:.8}
    Norm: {norm:.5e}
    Nodes: {}",
                mesh.num_nodes()
            );
            // Output current system to disk
            let mut systems = SystemCheckpoint::default();
            systems.save_system(dynamic.as_slice());

            mesh.export_vtu(
                format!("output/evscalar/ellipticmasslesssec{save_step}.vtu"),
                &systems,
                ExportVtuConfig {
                    title: "evscalar".to_string(),
                    ghost: false,
                },
            )
            .unwrap();

            save_step += 1;
        }

        // Compute step
        integrator.step(
            h,
            &mut DynamicOde {
                mesh: &mut mesh,
                mass,
            },
            dynamic.contigious(),
            update.contigious_mut(),
        );

        // Compute dissipation
        mesh.dissipation(
            DISS_ORDER,
            Quadrant,
            dynamic.as_slice(),
            dissipation.as_mut_slice(),
        );

        // Add everything together
        for i in 0..dynamic.contigious_mut().len() {
            dynamic.contigious_mut()[i] +=
                update.contigious()[i] + 0.5 * dissipation.contigious()[i];
        }

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;
    }

    Ok(())
}
