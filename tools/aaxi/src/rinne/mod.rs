//! This crate contains general configuration and paramter data types used by critgen, idgen, and evgen.
//! These types are shared across crates, and thus moved here to prevent redundent definition.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use crate::{
    config::{Config, GaugeCondition, Source},
    history::{RunHistory, RunRecord},
    misc,
};
use aeon::{
    prelude::*,
    solver::{Integrator, Method, SolverCallback},
};
use console::style;
use datasize::DataSize;
use eyre::eyre;
use indicatif::{HumanBytes, HumanCount, HumanDuration, MultiProgress, ProgressBar};

mod eqs;
mod garfinkle;
mod systems;

use eqs::{DynamicalData, DynamicalDerivs, ScalarFieldData, ScalarFieldDerivs, evolution};
pub use systems::*;

// *******************************
// Initial Data ******************
// *******************************

struct IterCallback<'a> {
    config: &'a Config,
    pb: ProgressBar,
    output: &'a Path,
}

impl<'a> SolverCallback<2, Scalar> for IterCallback<'a> {
    fn callback(
        &self,
        mesh: &Mesh<2>,
        input: SystemSlice<Scalar>,
        output: SystemSlice<Scalar>,
        iteration: usize,
    ) {
        self.pb.set_message(format!("Step: {}", iteration));
        self.pb.inc(1);

        if !self.config.visualize.save_relax {
            return;
        }

        let visualize_interval = self.config.visualize.save_relax_interval;

        if iteration % visualize_interval != 0 {
            return;
        }

        let i = iteration / visualize_interval;

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_field("Solution", input.into_scalar());
        checkpoint.save_field("Derivative", output.into_scalar());
        checkpoint
            .export_vtu(
                self.output.join("initial").join(format!(
                    "{}_level_{}_iter_{}.vtu",
                    self.config.name,
                    mesh.max_level(),
                    i
                )),
                ExportVtuConfig {
                    title: self.config.name.to_string(),
                    ghost: false,
                    stride: self.config.visualize.stride,
                },
            )
            .unwrap();
    }
}

/// Solve rinne's initial data problem using Garfinkles variables.
pub fn initial_data(config: &Config) -> eyre::Result<(Mesh<2>, SystemVec<Fields>)> {
    // Save initial time
    let start = Instant::now();

    // Cache directory
    let output = config.output_dir()?;
    let cache = output.join("cache");
    let init_cache = cache.join(format!("{}_init.dat", config.name));

    'cache: {
        if !config.cache.initial {
            // Don't attempt to load cache
            break 'cache;
        }

        // Attempt to load file
        let Ok(checkpoint) = Checkpoint::<2>::import_dat(&init_cache) else {
            break 'cache;
        };

        let mesh = checkpoint.read_mesh();
        let system = checkpoint.read_system::<Fields>();

        println!(
            "Successfully read cached initial data: {}",
            style(init_cache.display()).cyan()
        );

        return Ok((mesh, system));
    };

    if config.cache.initial {
        println!(
            "Failed to read cached initial data: {}",
            style(init_cache.display()).yellow()
        );
    }

    // Build mesh
    let mut mesh = Mesh::new(
        Rectangle {
            size: [config.domain.radius, config.domain.height],
            origin: [0.0, 0.0],
        },
        config.domain.cell_size,
        config.domain.cell_ghost,
        FaceArray::from_fn(|face| match face.side {
            false => BoundaryClass::Ghost,
            true => BoundaryClass::OneSided,
        }),
    );

    // Perform global refinements
    for _ in 0..config.regrid.global {
        mesh.refine_global();
    }

    // Build fields from sources.
    let fields = Fields {
        scalar_fields: config
            .source
            .iter()
            .flat_map(|source| {
                if let Source::ScalarField { mass, .. } = source {
                    Some(mass.unwrap())
                } else {
                    None
                }
            })
            .collect(),
    };

    // ************************************
    // Visualization

    // Path for all visualization data.
    if config.visualize.save_relax
        || config.visualize.save_relax_levels
        || config.visualize.save_relax_result
    {
        std::fs::create_dir_all(&output.join("initial"))?;
    }

    // ************************************
    // Solve

    // Allocate memory for system and transfer buffers.
    let mut transfer = SystemVec::new(fields.clone());
    let mut system = SystemVec::new(fields.clone());
    system.resize(mesh.num_nodes());

    println!("Relaxing Initial Data");

    // Progress bars for relaxation
    let m = MultiProgress::new();

    let mut step_count = 0;

    loop {
        let pb = m.add(ProgressBar::no_length());
        pb.set_style(misc::spinner_style());
        pb.set_prefix(format!("[Level {}]", mesh.max_level()));
        pb.enable_steady_tick(Duration::from_millis(100));

        match config.order {
            2 => {
                garfinkle::solve_order(
                    Order::<2>,
                    &mut mesh,
                    &config.relax,
                    IterCallback {
                        config,
                        pb: pb.clone(),
                        output: &output,
                    },
                    &config.source,
                    system.as_mut_slice(),
                )?;
            }
            4 => {
                garfinkle::solve_order(
                    Order::<4>,
                    &mut mesh,
                    &config.relax,
                    IterCallback {
                        config,
                        pb: pb.clone(),
                        output: &output,
                    },
                    &config.source,
                    system.as_mut_slice(),
                )?;
            }
            6 => {
                garfinkle::solve_order(
                    Order::<6>,
                    &mut mesh,
                    &config.relax,
                    IterCallback {
                        config,
                        pb: pb.clone(),
                        output: &output,
                    },
                    &config.source,
                    system.as_mut_slice(),
                )?;
            }
            _ => return Err(eyre!("Invalid initial data type and order")),
        };

        pb.finish_with_message(format!(
            "Relaxed in {} steps, {} nodes",
            pb.position(),
            mesh.num_nodes()
        ));
        step_count += pb.position();

        if config.visualize.save_relax_levels {
            let mut checkpoint = Checkpoint::default();
            checkpoint.attach_mesh(&mesh);
            checkpoint.save_system(system.as_slice());
            checkpoint.export_vtu(
                output.join("initial").join(format!(
                    "{}_level{}.vtu",
                    &config.name,
                    mesh.max_level()
                )),
                ExportVtuConfig {
                    title: config.name.clone(),
                    ghost: false,
                    stride: config.visualize.stride,
                },
            )?;
        }

        if mesh.max_level() >= config.limits.max_levels
            || mesh.num_nodes() >= config.limits.max_nodes
        {
            log::error!(
                "Failed to solve initial data, level: {}, nodes: {}",
                mesh.max_level(),
                mesh.num_nodes()
            );
            return Err(eyre!("failed to refine within perscribed limits"));
        }

        mesh.flag_wavelets(
            config.order,
            0.0,
            config.regrid.refine_error,
            system.as_slice(),
        );
        mesh.balance_flags();

        if mesh.requires_regridding() {
            transfer.resize(mesh.num_nodes());
            transfer
                .contigious_mut()
                .clone_from_slice(system.contigious());
            mesh.regrid();
            system.resize(mesh.num_nodes());

            match config.order {
                2 => mesh.transfer_system(Order::<2>, transfer.as_slice(), system.as_mut_slice()),
                4 => mesh.transfer_system(Order::<4>, transfer.as_slice(), system.as_mut_slice()),
                6 => mesh.transfer_system(Order::<6>, transfer.as_slice(), system.as_mut_slice()),
                _ => {}
            };
        } else {
            log::trace!(
                "Sucessfully refined mesh to give accuracy: {:.5e}",
                config.regrid.refine_error
            );
            break;
        }
    }

    if config.visualize.save_relax_result {
        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_system(system.as_slice());
        checkpoint.export_vtu(
            output.join("initial").join(format!("{}.vtu", config.name)),
            ExportVtuConfig {
                title: config.name.clone(),
                ghost: false,
                stride: config.visualize.stride,
            },
        )?;
    }

    m.clear()?;

    println!(
        "Finished relaxing in {}, {} Steps",
        HumanDuration(start.elapsed()),
        HumanCount(step_count),
    );
    println!("Mesh Info...");
    println!("- Num Nodes: {}", mesh.num_nodes());
    println!("- Active Cells: {}", mesh.num_active_cells());
    println!(
        "- RAM usage: ~{}",
        HumanBytes(mesh.estimate_heap_size() as u64)
    );
    println!("Field Info...");
    println!(
        "- RAM usage: ~{}",
        HumanBytes((system.estimate_heap_size() + transfer.estimate_heap_size()) as u64)
    );

    if config.cache.initial {
        // Ensure output directory exists
        std::fs::create_dir_all(cache)?;
        // Create checkpoint
        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_system::<Fields>(system.as_slice());
        checkpoint.export_dat(&init_cache)?;

        println!(
            "Successfully wrote initial data cache: {}",
            style(init_cache.display()).cyan()
        );
    }

    Ok((mesh, system))
}

const ORDER: Order<4> = Order::<4>;

#[derive(Clone)]
pub struct FieldDerivs {
    gauge: GaugeCondition,
}

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

        let num_scalar_fields = output.system().num_scalar_fields();
        let scalar_fields = engine.alloc::<ScalarFieldData>(num_scalar_fields);
        let scalar_field_derivs = engine.alloc::<ScalarFieldDerivs>(num_scalar_fields);

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
            second_derivatives!(
                lapse_f, lapse, lapse_r, lapse_z, lapse_rr, lapse_rz, lapse_zz
            );
            derivatives!(shiftr_f, shiftr, shiftr_r, shiftr_z);
            derivatives!(shiftz_f, shiftz, shiftz_r, shiftz_z);

            // Constraints
            derivatives!(theta_f, theta, theta_r, theta_z);
            derivatives!(zr_f, zr, zr_r, zr_z);
            derivatives!(zz_f, zz, zz_r, zz_z);

            for (i, mass) in output.system().scalar_fields().enumerate() {
                let phi = input.field(Field::ScalarField(ScalarField::Phi, i));
                let pi = input.field(Field::ScalarField(ScalarField::Pi, i));

                let scalar_field = &mut scalar_fields[i];

                scalar_field.phi = phi[index];
                scalar_field.phi_r = engine.derivative(phi, 0, vertex);
                scalar_field.phi_z = engine.derivative(phi, 1, vertex);
                scalar_field.phi_rr = engine.second_derivative(phi, 0, vertex);
                scalar_field.phi_rz = engine.mixed_derivative(phi, 0, 1, vertex);
                scalar_field.phi_zz = engine.second_derivative(phi, 1, vertex);

                scalar_field.pi = pi[index];
                scalar_field.pi_r = engine.derivative(pi, 0, vertex);
                scalar_field.pi_z = engine.derivative(pi, 1, vertex);

                scalar_field.mass = mass;
            }

            let system = DynamicalData {
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

            let mut derivs = DynamicalDerivs::default();
            evolution(
                system,
                scalar_fields,
                pos,
                &mut derivs,
                scalar_field_derivs,
                self.gauge,
            );

            output.field_mut(Field::Metric(Metric::Grr))[index] = derivs.grr_t;
            output.field_mut(Field::Metric(Metric::Grz))[index] = derivs.grz_t;
            output.field_mut(Field::Metric(Metric::Gzz))[index] = derivs.gzz_t;
            output.field_mut(Field::Metric(Metric::S))[index] = derivs.s_t;

            output.field_mut(Field::Metric(Metric::Krr))[index] = derivs.krr_t;
            output.field_mut(Field::Metric(Metric::Krz))[index] = derivs.krz_t;
            output.field_mut(Field::Metric(Metric::Kzz))[index] = derivs.kzz_t;
            output.field_mut(Field::Metric(Metric::Y))[index] = derivs.y_t;

            output.field_mut(Field::Constraint(Constraint::Theta))[index] = derivs.theta_t;
            output.field_mut(Field::Constraint(Constraint::Zr))[index] = derivs.zr_t;
            output.field_mut(Field::Constraint(Constraint::Zz))[index] = derivs.zz_t;

            output.field_mut(Field::Gauge(Gauge::Lapse))[index] = derivs.lapse_t;
            output.field_mut(Field::Gauge(Gauge::Shiftr))[index] = derivs.shiftr_t;
            output.field_mut(Field::Gauge(Gauge::Shiftz))[index] = derivs.shiftz_t;

            for i in 0..num_scalar_fields {
                let derivs = &scalar_field_derivs[i];
                output.field_mut(Field::ScalarField(ScalarField::Phi, i))[index] = derivs.phi;
                output.field_mut(Field::ScalarField(ScalarField::Pi, i))[index] = derivs.pi;
            }
        }
    }
}

pub fn evolve_data(
    config: &Config,
    history: &mut RunHistory,
    mut mesh: Mesh<2>,
    mut fields: SystemVec<Fields>,
) -> eyre::Result<()> {
    // Save initial time
    let start = Instant::now();

    let output = config.output_dir()?;
    // Create output folder.
    std::fs::create_dir_all(output.join("evolve"))?;
    // Cache system
    let system = fields.system().clone();

    // Integrate
    let mut integrator = Integrator::new(Method::RK4KO6(config.evolve.dissipation));
    let mut time = 0.0;
    let mut step = 0;

    let mut proper_time = 0.0;

    let mut save_step = 0;
    let mut steps_since_regrid = 0;

    let max_steps = config.evolve.max_steps;
    let max_time = config.evolve.max_time;
    let max_proper_time = config.evolve.max_proper_time;
    let cfl = config.evolve.cfl;
    let regrid_steps = config.regrid.flag_interval;
    let lower = config.regrid.coarsen_error;
    let upper = config.regrid.refine_error;
    let max_level = config.limits.max_levels;

    let mut time_since_save = 0.0;
    let save_interval = if config.visualize.save_evolve {
        config.visualize.save_evolve_interval
    } else {
        f64::MAX
    };
    let visualize_stride = config.visualize.stride;

    // Does the simulation disperse?
    let mut disperse = true;

    println!("Evolving data");

    // Setup progress bars
    let m = MultiProgress::new();
    // Max nodes
    let node_pb = m.add(ProgressBar::new(config.limits.max_nodes as u64));
    node_pb.set_style(misc::node_style());
    node_pb.enable_steady_tick(Duration::from_millis(100));
    node_pb.set_prefix("[Node] ");
    // Max levels
    let level_pb = m.add(ProgressBar::new(config.limits.max_levels as u64));
    level_pb.set_style(misc::level_style());
    level_pb.enable_steady_tick(Duration::from_millis(100));
    level_pb.set_prefix("[Level]");
    // Step spinner
    let step_pb = m.add(ProgressBar::no_length());
    step_pb.set_style(misc::spinner_style());
    step_pb.enable_steady_tick(Duration::from_millis(100));
    step_pb.set_prefix("[Step]");

    while time < max_time && proper_time < max_proper_time {
        assert!(fields.len() == mesh.num_nodes());
        // Fill boundaries
        mesh.fill_boundary(ORDER, FieldConditions, fields.as_mut_slice());

        // Check norm for NaN
        let norm = mesh.l2_norm_system(fields.as_slice());

        if norm.is_nan() || norm >= 1e60 {
            println!("{}", style(format!("Norm diverges: {:.5e}", norm)).red());
            disperse = false;
            break;
        }

        if step >= max_steps {
            println!(
                "{}",
                style(format!("Steps exceded maximum allocated steps: {}", step)).red()
            );
            disperse = false;
            break;
        }

        if mesh.num_nodes() >= config.limits.max_nodes {
            println!(
                "{}",
                style(format!(
                    "Nodes exceded maximum allocated nodes: {}",
                    mesh.num_nodes()
                ))
                .red()
            );
            disperse = false;
            break;
        }

        let ram_usage = fields.estimate_heap_size()
            + integrator.estimate_heap_size()
            + mesh.estimate_heap_size();

        if ram_usage >= config.limits.max_memory {
            println!(
                "{}",
                style(format!(
                    "RAM usage exceded maximum allocated bytes: {}",
                    HumanBytes(ram_usage as u64),
                ))
                .red()
            );
            disperse = false;
            break;
        }

        // Get step size
        let h = mesh.min_spacing() * cfl;

        // Periodically regrid mesh.
        if steps_since_regrid >= regrid_steps {
            steps_since_regrid = 0;

            mesh.flag_wavelets(4, lower, upper, fields.as_slice());
            mesh.limit_level_range_flags(1, max_level);
            mesh.balance_flags();
            mesh.regrid();

            // Copy system into tmp scratch space (provieded by dissipation).
            let scratch = integrator.scratch(fields.contigious().len());
            scratch.copy_from_slice(fields.contigious());
            fields.resize(mesh.num_nodes());
            mesh.transfer_system(
                ORDER,
                SystemSlice::from_contiguous(&scratch, &system),
                fields.as_mut_slice(),
            );

            continue;
        }

        // Periodically save output data
        if time_since_save >= save_interval {
            time_since_save -= save_interval;

            // Output current system to disk
            let mut checkpoint = Checkpoint::default();
            checkpoint.attach_mesh(&mesh);
            checkpoint.save_system(fields.as_slice());
            checkpoint.export_vtu(
                output
                    .join("evolve")
                    .join(format!("{}_{save_step}.vtu", config.name)),
                ExportVtuConfig {
                    title: config.name.clone(),
                    ghost: false,
                    stride: visualize_stride,
                },
            )?;

            save_step += 1;
        }

        // Compute step
        integrator.step(
            &mut mesh,
            ORDER,
            FieldConditions,
            FieldDerivs {
                gauge: config.evolve.gauge,
            },
            h,
            fields.as_mut_slice(),
        );

        let lapse = mesh.bottom_left_value(fields.field(Field::Gauge(Gauge::Lapse)));

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;

        proper_time += h * lapse;

        node_pb.set_position(mesh.num_nodes() as u64);
        level_pb.set_position(mesh.max_level() as u64);

        step_pb.inc(1);
        step_pb.set_message(format!(
            "Step: {}, Max Level {}, Proper Time {:.8}, Lapse {:.5e}",
            step,
            mesh.max_level(),
            proper_time,
            lapse
        ));

        // There is a chance that lapse is now NaN, which should trigger collapse,
        // but might be interpreted as disspersion because `NaN > max_proper_time`
        // Check again
        if lapse.is_nan() || lapse.is_infinite() || lapse.abs() == 0.0 {
            println!(
                "{}",
                style(format!("Lapse Collapses, α = {:.5e}", lapse)).red()
            );
            disperse = false;
            break;
        }

        // Serialize those values
        history.write_record(RunRecord {
            step,
            time,
            proper_time,
            lapse,
        })?;
    }

    m.clear()?;

    // Flush record at end of execution.
    history.flush()?;

    if disperse {
        println!("{}", style(format!("System disperses")).cyan());
    } else {
        println!("{}", style(format!("System collapses")).cyan());
    }

    println!(
        "Final evolution takes {}, {} steps",
        HumanDuration(start.elapsed()),
        HumanCount(step as u64),
    );
    println!("Mesh Info...");
    println!("- Num Nodes: {}", mesh.num_nodes());
    println!("- Active Cells: {}", mesh.num_active_cells());
    println!(
        "- RAM usage: ~{}",
        HumanBytes(mesh.estimate_heap_size() as u64)
    );
    println!("Field Info...");
    println!(
        "- RAM usage: ~{}",
        HumanBytes(
            (fields.estimate_heap_size()
                + integrator.estimate_heap_size()
                + mesh.estimate_heap_size()) as u64
        )
    );

    Ok(())
}
