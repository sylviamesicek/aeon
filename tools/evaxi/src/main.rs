use std::{path::PathBuf, process::ExitCode};

use aeon::{prelude::*, system::System};
use aeon_tensor::axisymmetry as axi;
use anyhow::{anyhow, Context as _, Result};
use clap::{Arg, Command};
use reborrow::{Reborrow as _, ReborrowMut as _};
use sharedaxi::{
    import_from_toml, Constraint, EVConfig, Field, FieldConditions, Fields, Gauge, Metric,
    Quadrant, ScalarField, Visualize,
};

mod tensor;

pub use tensor::HyperbolicSystem;

const ORDER: Order<4> = Order::<4>;
const DISS_ORDER: Order<6> = Order::<6>;

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
        let num_scalar_fields = output.system().num_scalar_fields();

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

        let sf_sources =
            engine.alloc::<axi::ScalarFieldSystem>(output.system().num_scalar_fields());

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

            for (i, mass) in output.system().scalar_fields().enumerate() {
                let phi = input.field(Field::ScalarField(ScalarField::Phi, i));
                let pi = input.field(Field::ScalarField(ScalarField::Pi, i));

                let phi_r = engine.derivative(phi, 0, vertex);
                let phi_z = engine.derivative(phi, 1, vertex);

                let phi_rr = engine.second_derivative(phi, 0, vertex);
                let phi_rz = engine.mixed_derivative(phi, 0, 1, vertex);
                let phi_zz = engine.second_derivative(phi, 1, vertex);

                let pi_r = engine.derivative(pi, 0, vertex);
                let pi_z = engine.derivative(pi, 1, vertex);

                use aeon_tensor::*;

                sf_sources[i] = axi::ScalarFieldSystem {
                    phi: ScalarFieldC2 {
                        value: phi[index],
                        derivs: Tensor::from([phi_r, phi_z]),
                        second_derivs: Tensor::from([[phi_rr, phi_rz], [phi_rz, phi_zz]]),
                    },
                    pi: ScalarFieldC1 {
                        value: pi[index],
                        derivs: Tensor::from([pi_r, pi_z]),
                    },
                    mass,
                };
            }

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

            let decomp = {
                use aeon_tensor::*;

                let metric = Metric::new(MatrixFieldC2 {
                    value: Tensor::from(system.metric()),
                    derivs: Tensor::from(system.metric_derivs()),
                    second_derivs: Tensor::from(system.metric_second_derivs()),
                });

                let seed = ScalarFieldC2 {
                    value: system.seed(),
                    derivs: Tensor::from(system.seed_derivs()),
                    second_derivs: Tensor::from(system.seed_second_derivs()),
                };

                let mut source = axi::StressEnergy::vacuum();

                for i in 0..num_scalar_fields {
                    let sf = axi::StressEnergy::scalar(&metric, sf_sources[i].clone());

                    source.energy += sf.energy;
                    source.momentum[0] += sf.momentum[0];
                    source.momentum[1] += sf.momentum[1];

                    source.stress[[0, 0]] += sf.stress[[0, 0]];
                    source.stress[[0, 1]] += sf.stress[[0, 1]];
                    source.stress[[1, 1]] += sf.stress[[1, 1]];
                    source.stress[[1, 0]] += sf.stress[[1, 0]];

                    source.angular_momentum += sf.angular_momentum;
                    source.angular_shear[[0]] += sf.angular_shear[[0]];
                    source.angular_shear[[1]] += sf.angular_shear[[1]];
                    source.angular_stress += sf.angular_stress;
                }

                axi::Decomposition::new(
                    pos,
                    axi::System {
                        metric,
                        seed,
                        k: MatrixFieldC1 {
                            value: Tensor::from(system.extrinsic()),
                            derivs: Tensor::from(system.extrinsic_derivs()),
                        },
                        y: ScalarFieldC1 {
                            value: system.y,
                            derivs: Tensor::from([system.y_r, system.y_z]),
                        },
                        theta: ScalarFieldC1 {
                            value: system.theta,
                            derivs: Tensor::from([system.theta_r, system.theta_z]),
                        },
                        z: VectorFieldC1 {
                            value: Tensor::from([system.zr, system.zz]),
                            derivs: Tensor::from([
                                [system.zr_r, system.zr_z],
                                [system.zz_r, system.zz_z],
                            ]),
                        },
                        lapse: ScalarFieldC2 {
                            value: system.lapse,
                            derivs: Tensor::from([system.lapse_r, system.lapse_z]),
                            second_derivs: Tensor::from([
                                [system.lapse_rr, system.lapse_rz],
                                [system.lapse_rz, system.lapse_zz],
                            ]),
                        },
                        shift: VectorFieldC1 {
                            value: Tensor::from([system.shiftr, system.shiftz]),
                            derivs: Tensor::from([
                                [system.shiftr_r, system.shiftr_z],
                                [system.shiftz_r, system.shiftz_z],
                            ]),
                        },
                        source,
                    },
                )
            };

            let evolve = axi::Decomposition::evolution(&decomp);
            let gauge = axi::Decomposition::gauge(&decomp);

            output.field_mut(Field::Metric(Metric::Grr))[index] = evolve.g[[0, 0]];
            output.field_mut(Field::Metric(Metric::Grz))[index] = evolve.g[[0, 1]];
            output.field_mut(Field::Metric(Metric::Gzz))[index] = evolve.g[[1, 1]];
            output.field_mut(Field::Metric(Metric::S))[index] = evolve.seed;

            output.field_mut(Field::Metric(Metric::Krr))[index] = evolve.k[[0, 0]];
            output.field_mut(Field::Metric(Metric::Krz))[index] = evolve.k[[0, 1]];
            output.field_mut(Field::Metric(Metric::Kzz))[index] = evolve.k[[1, 1]];
            output.field_mut(Field::Metric(Metric::Y))[index] = evolve.y;

            output.field_mut(Field::Constraint(Constraint::Theta))[index] = evolve.theta;
            output.field_mut(Field::Constraint(Constraint::Zr))[index] = evolve.z[[0]];
            output.field_mut(Field::Constraint(Constraint::Zz))[index] = evolve.z[[1]];

            output.field_mut(Field::Gauge(Gauge::Lapse))[index] = gauge.lapse;
            output.field_mut(Field::Gauge(Gauge::Shiftr))[index] = gauge.shift[[0]];
            output.field_mut(Field::Gauge(Gauge::Shiftz))[index] = gauge.shift[[1]];

            for i in 0..num_scalar_fields {
                let scalar = axi::Decomposition::scalar(&decomp, sf_sources[i].clone());

                output.field_mut(Field::ScalarField(ScalarField::Phi, i))[index] = scalar.phi;
                output.field_mut(Field::ScalarField(ScalarField::Pi, i))[index] = scalar.pi;
            }
        }
    }
}

pub struct FieldEvolution<'a> {
    mesh: &'a mut Mesh<2>,
    system: &'a Fields,
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
        let src = SystemSlice::from_contiguous(f, self.system);
        let mut dest = SystemSliceMut::from_contiguous(df, self.system);

        self.mesh
            .evaluate(ORDER, Quadrant, FieldDerivs, src.rb(), dest.rb_mut());

        self.mesh
            .weak_boundary(ORDER, Quadrant, FieldConditions, src.rb(), dest.rb_mut());
    }
}

pub fn evolution() -> Result<bool> {
    // Load configuration
    let matches = Command::new("evaxi")
        .about("A program for evolving data generated by idaxi")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("v0.1.0")
        .arg(
            Arg::new("config")
                .long("config")
                .short('c')
                .help("Path of configuration file for various parameters of evolution")
                .value_name("FILE")
                .required(true),
        )
        .arg(
            Arg::new("path")
                .help("Path of data file storing mesh and initial data")
                .value_name("FILE")
                .required(true),
        )
        .get_matches();

    // Load configuration data
    let config = import_from_toml::<EVConfig>(
        matches
            .get_one::<String>("config")
            .ok_or(anyhow!("Failed to specify config argument"))?,
    )?;

    // Load header data
    let level = config.logging.filter();
    let output = PathBuf::from(
        config
            .output_dir
            .clone()
            .unwrap_or_else(|| format!("{}_output", &config.name)),
    );

    // Initialize logging
    env_logger::builder().filter_level(level).init();

    // Compute absolute directory
    let dir = std::env::current_dir().context("Failed to find current working directory")?;
    let absolute = if output.is_absolute() {
        output
    } else {
        dir.join(output)
    };

    std::fs::create_dir_all(&absolute)?;

    // Parse data file path.
    let path = matches
        .get_one::<String>("path")
        .ok_or(anyhow!("Failed to specify path argument"))?;

    // Import data from file.
    let mut mesh = Mesh::<2>::default();
    let mut checkpoint = SystemCheckpoint::default();
    mesh.import_dat(path, &mut checkpoint)?;

    // Load fields
    let mut fields = checkpoint.read_system_ser::<Fields>();
    // Temporary data
    let mut update = SystemVec::new(fields.system().clone());
    let mut dissipation = SystemVec::new(fields.system().clone());

    // Integrate
    let mut integrator = Rk4::new();
    let mut time = 0.0;
    let mut step = 0;

    let mut save_step = 0;
    let mut steps_since_regrid = 0;

    // let mut errors = Vec::new();

    let max_steps = config.max_steps;
    let max_time = config.max_time;
    let cfl = config.cfl;
    let regrid_steps = config.regrid.flag_interval;
    let lower = config.regrid.coarsen_tolerance;
    let upper = config.regrid.refine_tolerance;
    let max_level = config.regrid.max_levels;

    let mut time_since_save = 0.0;
    let mut save_interval = f64::MAX;
    let mut visualize_stride = 1;

    if let Some(vis @ Visualize { .. }) = config.visualize {
        save_interval = vis.save_interval;
        time_since_save = save_interval;
        visualize_stride = vis.stride;
    }

    let mut does_disperse = true;

    while time < max_time {
        assert!(fields.len() == mesh.num_nodes());
        // Fill boundaries
        mesh.fill_boundary(ORDER, Quadrant, FieldConditions, fields.as_mut_slice());

        // Check Norm
        let norm = mesh.l2_norm(fields.as_slice());

        if norm.is_nan() || norm >= 1e60 {
            log::trace!("Evolution collapses, norm: {}", norm);
            does_disperse = false;
            break;
        }

        if step >= max_steps {
            log::error!("Evolution exceded maximum allocated steps: {}", step);
            return Err(anyhow!(
                "exceded max allotted steps for evolution: {}",
                step
            ));
        }

        if mesh.num_nodes() >= config.max_nodes {
            log::error!(
                "Evolution exceded maximum allocated nodes: {}",
                mesh.num_nodes()
            );
            return Err(anyhow!(
                "exceded max allotted nodes for evolution: {}",
                mesh.num_nodes()
            ));
        }

        if mesh.max_level() >= config.regrid.max_levels {
            log::trace!(
                "Evolution collapses, Reached maximum allowed level of refinement: {}",
                mesh.max_level()
            );
            does_disperse = false;
            break;
        }

        // Get step size
        let h = mesh.min_spacing() * cfl;

        // Resize vectors
        update.resize(mesh.num_nodes());
        dissipation.resize(mesh.num_nodes());

        if steps_since_regrid > regrid_steps {
            steps_since_regrid = 0;

            mesh.flag_wavelets(4, lower, upper, Quadrant, fields.as_slice());
            mesh.set_regrid_level_limit(max_level);
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

        if time_since_save >= save_interval {
            time_since_save -= save_interval;

            //         log::trace!(
            //             "Saving Checkpoint {save_step}
            // Time: {time:.5}, Step: {h:.8}
            // Norm: {norm:.5e}
            // Nodes: {}",
            //             mesh.num_nodes()
            //         );

            log::trace!(
                "Saving Checkpoint {save_step}, Time: {time:.5}, Step: {h:.8}, Norm: {norm:.5e}, Nodes: {}",
                mesh.num_nodes()
            );
            // Output current system to disk
            let mut systems = SystemCheckpoint::default();
            systems.save_system_ser(fields.as_slice());

            mesh.export_vtu(
                absolute.join(format!("{}_{save_step}.vtu", config.name)),
                &systems,
                ExportVtuConfig {
                    title: "evbrill".to_string(),
                    ghost: false,
                    stride: visualize_stride,
                },
            )?;

            // let l2_norm = mesh.l2_norm(fields.field(Field::Constraint(Constraint::Theta)).into());
            // let max_norm = mesh.max_norm(fields.field(Field::Constraint(Constraint::Theta)).into());

            // let space = mesh.block_space(0);
            // let index = space.index_from_vertex([0; 2]);

            // let origin = fields.field(Field::Constraint(Constraint::Theta))[index];

            // errors.push((time, l2_norm, max_norm, origin));

            // let mut error_csv = String::new();

            // for (time, l2_norm, max, origin) in errors.iter() {
            //     error_csv.write_fmt(format_args!("{time}, {l2_norm}, {max}, {origin},\n"))?;
            // }

            // let mut file = std::fs::File::create("output/negcritical_errors.txt")?;
            // file.write_all(error_csv.as_bytes())?;

            save_step += 1;
        }

        // Compute step
        integrator.step(
            h,
            &mut FieldEvolution {
                mesh: &mut mesh,
                system: fields.system(),
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

    Ok(does_disperse)
}

fn main() -> ExitCode {
    match evolution() {
        Ok(true) => ExitCode::from(0),
        Ok(false) => ExitCode::from(2),
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
