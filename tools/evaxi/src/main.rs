use std::{fmt::Write as _, io::Write as _};

use aeon::{prelude::*, system::System};
use anyhow::{anyhow, Result};
use clap::{Arg, Command};
use reborrow::{Reborrow as _, ReborrowMut as _};
use sharedaxi::{Constraint, Field, FieldConditions, Fields, Gauge, Metric, Quadrant};

mod tensor;

use tensor::hyperbolic;
pub use tensor::HyperbolicSystem;

const MAX_TIME: f64 = 15.0;
const MAX_STEPS: usize = 100000;
const MAX_LEVEL: usize = 128;

const CFL: f64 = 0.1;
const ORDER: Order<4> = Order::<4>;
const DISS_ORDER: Order<6> = Order::<6>;

const SAVE_CHECKPOINT: f64 = 0.02;
const FORCE_SAVE: bool = false;
const REGRID_SKIP: usize = 20;

const LOWER: f64 = 1e-8;
const UPPER: f64 = 1e-6;

#[derive(Clone)]
pub struct FieldDerivs;

impl Function<2> for FieldDerivs {
    type Input = Fields;
    type Output = Fields;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
        let grr_f = input.field(Field::Metric(Metric::Grr));
        let grz_f = input.field(Field::Metric(Metric::Grz));
        let gzz_f = input.field(Field::Metric(Metric::Gzz));
        let s_f = input.field(Field::Metric(Metric::S));

        let krr_f = input.field(Field::Metric(Metric::Krr));
        let krz_f = input.field(Field::Metric(Metric::Krz));
        let kzz_f = input.field(Field::Metric(Metric::Kzz));
        let y_f = input.field(Field::Metric(Metric::Y));

        let lapse_f = input.field(Field::Gauge(Gauge::Lapse));
        let shiftr_f = input.field(Field::Gauge(Gauge::Shiftr));
        let shiftz_f = input.field(Field::Gauge(Gauge::Shiftz));

        let theta_f = input.field(Field::Constraint(Constraint::Theta));
        let zr_f = input.field(Field::Constraint(Constraint::Zr));
        let zz_f = input.field(Field::Constraint(Constraint::Zz));

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let pos = engine.position(vertex);
            let index = engine.index_from_vertex(vertex);

            macro_rules! derivatives {
                ($field:ident, $value:ident, $dr:ident, $dz:ident) => {
                    let $value = $field[index];
                    let $dr = engine.derivative($field, 0, vertex);
                    let $dz = engine.derivative($field, 1, vertex);
                };
            }

            macro_rules! second_derivatives {
                ($field:ident, $value:ident, $dr:ident, $dz:ident, $drr:ident, $drz:ident, $dzz:ident) => {
                    let $value = $field[index];
                    let $dr = engine.derivative($field, 0, vertex);
                    let $dz = engine.derivative($field, 1, vertex);

                    let $drr = engine.second_derivative($field, 0, vertex);
                    let $drz = engine.mixed_derivative($field, 0, 1, vertex);
                    let $dzz = engine.second_derivative($field, 1, vertex);
                };
            }

            // Metric
            second_derivatives!(grr_f, grr, grr_r, grr_z, grr_rr, grr_rz, grr_zz);
            second_derivatives!(gzz_f, gzz, gzz_r, gzz_z, gzz_rr, gzz_rz, gzz_zz);
            second_derivatives!(grz_f, grz, grz_r, grz_z, grz_rr, grz_rz, grz_zz);

            // S
            second_derivatives!(s_f, s, s_r, s_z, s_rr, s_rz, s_zz);

            // K
            derivatives!(krr_f, krr, krr_r, krr_z);
            derivatives!(kzz_f, kzz, kzz_r, kzz_z);
            derivatives!(krz_f, krz, krz_r, krz_z);

            // Y
            derivatives!(y_f, y, y_r, y_z);

            // Gauge
            second_derivatives!(lapse_f, lapse, lapse_r, lapse_z, lapse_rr, lapse_rz, lapse_zz);
            derivatives!(shiftr_f, shiftr, shiftr_r, shiftr_z);
            derivatives!(shiftz_f, shiftz, shiftz_r, shiftz_z);

            // Constraints
            derivatives!(theta_f, theta, theta_r, theta_z);
            derivatives!(zr_f, zr, zr_r, zr_z);
            derivatives!(zz_f, zz, zz_r, zz_z);

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
            };

            let derivs = hyperbolic(system, pos);

            output.field_mut(Field::Metric(Metric::Grr))[index] = derivs.grr_t;
            output.field_mut(Field::Metric(Metric::Grz))[index] = derivs.grz_t;
            output.field_mut(Field::Metric(Metric::Gzz))[index] = derivs.gzz_t;
            output.field_mut(Field::Metric(Metric::S))[index] = derivs.s_t;

            output.field_mut(Field::Metric(Metric::Krr))[index] = derivs.krr_t;
            output.field_mut(Field::Metric(Metric::Krz))[index] = derivs.krz_t;
            output.field_mut(Field::Metric(Metric::Kzz))[index] = derivs.kzz_t;
            output.field_mut(Field::Metric(Metric::Y))[index] = derivs.y_t;

            output.field_mut(Field::Gauge(Gauge::Lapse))[index] = derivs.lapse_t;
            output.field_mut(Field::Gauge(Gauge::Shiftr))[index] = derivs.shiftr_t;
            output.field_mut(Field::Gauge(Gauge::Shiftz))[index] = derivs.shiftz_t;

            output.field_mut(Field::Constraint(Constraint::Theta))[index] = derivs.theta_t;
            output.field_mut(Field::Constraint(Constraint::Zr))[index] = derivs.zr_t;
            output.field_mut(Field::Constraint(Constraint::Zz))[index] = derivs.zz_t;
        }
    }
}

pub struct FieldEvolution<'a> {
    mesh: &'a mut Mesh<2>,
    system: Fields,
}

impl<'a> Ode for FieldEvolution<'a> {
    fn dim(&self) -> usize {
        self.mesh.num_nodes() * self.system.count()
    }

    fn preprocess(&mut self, data: &mut [f64]) {
        self.mesh.fill_boundary(
            ORDER,
            Quadrant,
            FieldConditions,
            SystemSliceMut::from_contiguous(data, &self.system),
        );
    }

    fn derivative(&mut self, f: &[f64], df: &mut [f64]) {
        let src = SystemSlice::from_contiguous(f, &self.system);
        let mut dest = SystemSliceMut::from_contiguous(df, &self.system);

        self.mesh
            .evaluate(ORDER, Quadrant, FieldDerivs, src.rb(), dest.rb_mut());

        self.mesh
            .weak_boundary(ORDER, Quadrant, FieldConditions, src.rb(), dest.rb_mut());
    }
}

pub fn main() -> Result<()> {
    // Load configuration
    let matches = Command::new("evaxi")
        .about("A program for evolving data generated by idaxi")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("v0.0.1")
        .arg(Arg::new("order").long("order").short('o').value_name("NUM"))
        .arg(
            Arg::new("path")
                .help("Path of data file storing mesh and initial data")
                .value_name("FILE")
                .required(true),
        )
        .get_matches();

    // Load data file path.
    let path = matches
        .get_one::<String>("path")
        .ok_or(anyhow!("Failed to specify path argument"))?;

    // Import data from file.
    let mut mesh = Mesh::<2>::default();
    let mut checkpoint = SystemCheckpoint::default();
    mesh.import_dat(path, &mut checkpoint)?;

    let mut fields = checkpoint.read_system::<Fields>();

    let mut update = SystemVec::<Fields>::default();
    let mut dissipation = SystemVec::<Fields>::default();

    // Integrate
    let mut integrator = Rk4::new();
    let mut time = 0.0;
    let mut step = 0;

    let mut time_since_save = 0.0;
    let mut save_step = 0;

    let mut steps_since_regrid = 0;

    let mut errors = Vec::new();

    while step < MAX_STEPS && time < MAX_TIME {
        assert!(fields.len() == mesh.num_nodes());
        // Fill boundaries
        mesh.fill_boundary(ORDER, Quadrant, FieldConditions, fields.as_mut_slice());

        // Check Norm
        let norm = mesh.l2_norm(fields.as_slice());
        if norm.is_nan() {
            log::warn!("Norm is NaN");
            break;
        }

        if mesh.num_nodes() >= 16_000_000 {
            log::warn!("To many degrees of freedom used");
            break;
        }

        // Get step size
        let h = mesh.min_spacing() * CFL;

        // Resize vectors
        update.resize(mesh.num_nodes());
        dissipation.resize(mesh.num_nodes());

        if steps_since_regrid > REGRID_SKIP {
            steps_since_regrid = 0;

            mesh.flag_wavelets(4, LOWER, UPPER, Quadrant, fields.as_slice());
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

            // Copy system into tmp scratch space (provieded by dissipation).
            dissipation
                .contigious_mut()
                .copy_from_slice(fields.contigious());
            fields.resize(mesh.num_nodes());
            mesh.transfer_system(
                ORDER,
                Quadrant,
                dissipation.as_slice(),
                fields.as_mut_slice(),
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
            systems.save_system(fields.as_slice());

            mesh.export_vtu(
                format!("output/evbrill/negcritical{save_step}.vtu"),
                &systems,
                ExportVtuConfig {
                    title: "evbrill".to_string(),
                    ghost: false,
                },
            )
            .unwrap();

            let l2_norm = mesh.l2_norm(fields.field(Field::Constraint(Constraint::Theta)).into());
            let max_norm = mesh.max_norm(fields.field(Field::Constraint(Constraint::Theta)).into());

            let space = mesh.block_space(0);
            let index = space.index_from_vertex([0; 2]);

            let origin = fields.field(Field::Constraint(Constraint::Theta))[index];

            errors.push((time, l2_norm, max_norm, origin));

            let mut error_csv = String::new();

            for (time, l2_norm, max, origin) in errors.iter() {
                error_csv.write_fmt(format_args!("{time}, {l2_norm}, {max}, {origin},\n"))?;
            }

            let mut file = std::fs::File::create("output/negcritical_errors.txt")?;
            file.write_all(error_csv.as_bytes())?;

            save_step += 1;
        }

        // Compute step
        integrator.step(
            h,
            &mut FieldEvolution {
                mesh: &mut mesh,
                system: Fields,
            },
            fields.contigious(),
            update.contigious_mut(),
        );

        // Compute dissipation
        mesh.dissipation(
            DISS_ORDER,
            Quadrant,
            fields.as_slice(),
            dissipation.as_mut_slice(),
        );

        // Add everything together
        mesh.add_assign_fma(
            update.as_slice(),
            0.5,
            dissipation.as_slice(),
            fields.as_mut_slice(),
        );

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;
    }

    log::info!("Writing Error CSV");

    Ok(())
}
