use crate::{
    eqs::{
        DynamicalData, DynamicalDerivs, GaugeCondition, ScalarFieldData, ScalarFieldDerivs,
        evolution,
    },
    horizon::{self, ApparentHorizonFinder, HorizonError, HorizonProjection, HorizonStatus},
    misc,
    run::{
        config::Config,
        history::{RunHistory, RunRecord},
        interval::IntervalTracker,
        status::{Status, Strategy},
    },
    systems::{Constraint, Field, FieldConditions, Fields, Gauge, Metric, ScalarField},
};
use aeon::{
    prelude::*,
    solver::{Integrator, Method, SolverCallback},
};
use console::style;
use datasize::DataSize as _;
use eyre::eyre;
use indicatif::{HumanBytes, HumanCount, HumanDuration, MultiProgress, ProgressBar};
use std::{
    convert::Infallible,
    path::Path,
    time::{Duration, Instant},
};

const ORDER: Order<4> = Order::<4>;

#[derive(Clone)]
struct FieldDerivs {
    gauge: GaugeCondition,
}

impl Function<2> for FieldDerivs {
    type Input = Fields;
    type Output = Fields;
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Infallible> {
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

        Ok(())
    }
}

pub struct HorizonCallback<'a> {
    directory: &'a Path,
    positions: &'a mut Vec<[f64; 2]>,
    config: &'a Config,
    pb: &'a mut ProgressBar,
}

impl<'a> SolverCallback<1, Scalar> for HorizonCallback<'a> {
    type Error = std::io::Error;

    fn callback(
        &mut self,
        surface: &Mesh<1>,
        radius: SystemSlice<Scalar>,
        _output: SystemSlice<Scalar>,
        iteration: usize,
    ) -> Result<(), Self::Error> {
        self.pb.set_position(iteration as u64);

        if !self.config.visualize.horizon_relax {
            return Ok(());
        }

        let interval = self.config.visualize.horizon_relax_interval.unwrap_steps();

        if iteration % interval != 0 {
            return Ok(());
        }

        let save_index = iteration / interval;
        let radius = radius.field(());

        self.positions.resize(surface.num_nodes(), [0.0; 2]);
        horizon::compute_position_from_radius(surface, radius, &mut self.positions);

        let checkpoint_file = self.directory.join(format!("horizon_{}.vtu", save_index));

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&surface);
        checkpoint.set_embedding(&self.positions);
        checkpoint.save_field("radius", radius);
        checkpoint.export_vtu(
            checkpoint_file,
            ExportVtuConfig {
                title: "horizon".into(),
                ghost: false,
                stride: ExportStride::PerVertex,
            },
        )?;

        Ok(())
    }
}

pub fn evolve_data(
    config: &Config,
    history: &mut RunHistory,
    mut mesh: Mesh<2>,
    mut fields: SystemVec<Fields>,
) -> eyre::Result<Status> {
    // Save initial time
    let start = Instant::now();

    let output = config.output_dir()?;
    // Create output folder.
    std::fs::create_dir_all(output.join("evolve"))?;
    // If we are outputing horizon data, create a folder for that too
    if config.horizon.search && config.visualize.horizon_relax {
        std::fs::create_dir_all(output.join("horizon"))?;
    }

    // Cache system
    let system = fields.system().clone();

    // Integrate
    let mut integrator = Integrator::new(Method::RK4KO6(config.evolve.dissipation));

    let mut step = 0;
    let mut coord_time = 0.0;
    let mut proper_time = 0.0;

    let max_levels = config.limits.max_levels;
    let max_nodes = config.limits.max_nodes;
    let max_memory = config.limits.max_memory;

    let max_steps = config.evolve.max_steps;
    let max_coord_time = config.evolve.max_coord_time;
    let max_proper_time = config.evolve.max_proper_time;
    let cfl = config.evolve.cfl;

    // ************************
    // Regridding

    let mut regrid_tracker = IntervalTracker::new();
    let regrid_interval = config.evolve.regrid_interval;
    let lower = config.evolve.coarsen_error;
    let upper = config.evolve.refine_error;

    // *************************
    // Visualization

    let visualize = config.visualize.evolve;
    let mut visualize_tracker = IntervalTracker::new();
    let visualize_interval = config.visualize.evolve_interval;
    let visualize_stride = config.visualize.stride;
    let mut visualize_index = 0;

    // ***************************
    // Apparent horizons

    let mut search_tracker = IntervalTracker::new();
    let search_interval = config.horizon.search_interval;
    let mut search_index = 0;
    let mut search_positions = Vec::<[f64; 2]>::new();
    let mut search_surface = if config.horizon.search {
        let mut surface = horizon::surface();

        for _ in 0..config.horizon.global_refine {
            surface.refine_global();
        }
        let radius = vec![0.0; surface.num_nodes()];

        Some((surface, radius))
    } else {
        None
    };

    let mut finder = ApparentHorizonFinder::new();
    finder.solver.cfl = config.horizon.relax.cfl;
    finder.solver.dampening = config.horizon.relax.dampening;
    finder.solver.max_steps = config.horizon.relax.max_steps;
    finder.solver.tolerance = config.horizon.relax.tolerance;
    finder.solver.adaptive = true;

    let mut horizon_field = Vec::new();

    // **************************
    // Evolve

    println!("Evolving data");

    // Setup progress bars
    let m = MultiProgress::new();
    // Max nodes
    let node_pb = m.add(ProgressBar::new(config.limits.max_nodes as u64));
    node_pb.set_style(misc::node_style());
    node_pb.enable_steady_tick(Duration::from_millis(100));
    node_pb.set_prefix("[Node]  ");
    // Max levels
    let level_pb = m.add(ProgressBar::new(config.limits.max_levels as u64));
    level_pb.set_style(misc::level_style());
    level_pb.enable_steady_tick(Duration::from_millis(100));
    level_pb.set_prefix("[Level] ");
    // Memory usage
    let memory_pb = m.add(ProgressBar::new(config.limits.max_memory as u64));
    memory_pb.set_style(misc::byte_style());
    memory_pb.enable_steady_tick(Duration::from_millis(100));
    memory_pb.set_prefix("[Memory]");
    // Step spinner
    let step_pb = m.add(ProgressBar::no_length());
    step_pb.set_style(misc::spinner_style());
    step_pb.enable_steady_tick(Duration::from_millis(100));
    step_pb.set_prefix("[Step] ");

    let status: Status = 'evolve: loop {
        assert!(fields.len() == mesh.num_nodes());

        // ****************************
        // Coordinate time

        if coord_time > max_coord_time {
            println!(
                "{}",
                style(format!(
                    "Evolution reached max coordinate time: {}",
                    coord_time
                ))
                .green()
            );

            break 'evolve config
                .error_handler
                .on_max_evolve_coord_time
                .status_or_crash(|| eyre!("evolution reached max coordinate time"))?;
        }

        // ******************************
        // Proper time

        if proper_time > max_proper_time {
            println!(
                "{}",
                style(format!(
                    "Evolution reached max proper time: {}",
                    proper_time
                ))
                .green()
            );

            break 'evolve config
                .error_handler
                .on_max_evolve_proper_time
                .status_or_crash(|| eyre!("evolution reached max proper time"))?;
        }

        // *******************************
        // Steps

        if step > max_steps {
            println!(
                "{}",
                style(format!("Evolution reached max steps: {}", step)).green()
            );

            break 'evolve config
                .error_handler
                .on_max_evolve_steps
                .status_or_crash(|| eyre!("evolution reached max steps"))?;
        }

        // *********************************
        // Max nodes, levels, and memory

        let memory_usage = fields.estimate_heap_size()
            + integrator.estimate_heap_size()
            + mesh.estimate_heap_size();

        if mesh.num_nodes() > max_nodes {
            println!(
                "{}",
                style(format!(
                    "Nodes exceded maximum allocated nodes: {}",
                    mesh.num_nodes()
                ))
                .red()
            );

            break 'evolve config
                .error_handler
                .on_max_nodes
                .status_or_crash(|| eyre!("nodes excede maximum allocated nodes"))?;
        }

        if memory_usage > max_memory {
            println!(
                "{}",
                style(format!(
                    "RAM usage exceded maximum allocated bytes: {}",
                    HumanBytes(memory_usage as u64),
                ))
                .red()
            );

            break 'evolve config
                .error_handler
                .on_max_nodes
                .status_or_crash(|| eyre!("evolution exceded maximum allowed memory"))?;
        }

        if mesh.num_levels() > max_levels {
            if let Some(status) = config
                .error_handler
                .on_max_levels
                .execute(|| eyre!("evolution exceded maximum allowed levels"))?
            {
                println!(
                    "{}",
                    style(format!(
                        "Evolution exceded maxmimum allowed levels: {}",
                        mesh.num_levels(),
                    ))
                    .red()
                );

                break 'evolve status;
            }
        }

        // ******************************

        // Fill boundaries
        mesh.fill_boundary(ORDER, FieldConditions, fields.as_mut_slice());

        // *******************************
        // Norm

        let norm = mesh.l2_norm_system(fields.as_slice());

        if norm.is_nan() || norm >= 1e60 {
            println!("{}", style(format!("Norm diverges: {:.5e}", norm)).red());

            break 'evolve config
                .error_handler
                .on_norm_diverge
                .status_or_crash(|| eyre!("norm diverges during evolution"))?;
        }

        // *******************************

        // Get step size
        let h = mesh.min_spacing() * cfl;

        // ****************************************
        // Periodically regrid mesh

        let mut regrid_flag = false;

        regrid_tracker.every(regrid_interval, || {
            mesh.flag_wavelets(4, lower, upper, fields.as_slice());
            mesh.balance_flags();

            if config.error_handler.on_max_levels == Strategy::Ignore {
                mesh.limit_level_range_flags(0, max_levels - 1);
                mesh.balance_flags();
            }

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

            regrid_flag = true;
        });

        if regrid_flag {
            continue;
        }

        // ******************************************
        // Periodically save output data

        visualize_tracker.try_every(visualize_interval, || -> std::io::Result<()> {
            if !visualize {
                return Ok(());
            }

            horizon_field.resize(mesh.num_nodes(), 0.0);

            mesh.evaluate(
                4,
                HorizonProjection,
                fields.as_slice(),
                SystemSliceMut::from_scalar(&mut horizon_field),
            )
            .unwrap();

            // Output current system to disk
            let mut checkpoint = Checkpoint::default();
            checkpoint.attach_mesh(&mesh);
            checkpoint.save_field("Horizon", &horizon_field);
            checkpoint.save_system(fields.as_slice());
            checkpoint.export_vtu(
                output
                    .join("evolve")
                    .join(format!("{}_{visualize_index}.vtu", config.name)),
                ExportVtuConfig {
                    title: config.name.clone(),
                    ghost: false,
                    stride: visualize_stride,
                },
            )?;

            visualize_index += 1;

            Ok(())
        })?;

        // ***********************************
        // Periodically run horizon search

        let mut search_status = None;

        search_tracker.try_every(search_interval, || {
            let Some((surface, surface_radius)) = search_surface.as_mut() else {
                return Ok::<_, eyre::Report>(());
            };

            surface_radius.fill(config.horizon.search_initial_radius);

            let horizon_dir = output.join("horizon").join(format!("{}", search_index));

            if config.visualize.horizon_relax {
                std::fs::create_dir_all(&horizon_dir).map_err(eyre::Report::new)?;
            }

            let mut horizon_pb = m.add(ProgressBar::new(config.horizon.relax.max_steps as u64));
            horizon_pb.set_style(misc::node_style());
            horizon_pb.enable_steady_tick(Duration::from_millis(100));
            horizon_pb.set_prefix("- [Horizon Search]");

            let result = finder.search_with_callback(
                &mesh,
                fields.as_slice(),
                ORDER,
                surface,
                HorizonCallback {
                    directory: horizon_dir.as_path(),
                    positions: &mut search_positions,
                    config: &config,
                    pb: &mut horizon_pb,
                },
                surface_radius,
            );

            horizon_pb.finish_and_clear();

            if config.visualize.horizon_relax {
                horizon_field.resize(mesh.num_nodes(), 0.0);

                mesh.evaluate(
                    4,
                    HorizonProjection,
                    fields.as_slice(),
                    SystemSliceMut::from_scalar(&mut horizon_field),
                )?;

                // Output current system to disk, for reference in horizon search.
                let mut checkpoint = Checkpoint::default();
                checkpoint.attach_mesh(&mesh);
                checkpoint.save_system(fields.as_slice());
                checkpoint.save_field("Horizon", &horizon_field);
                checkpoint.export_vtu(
                    horizon_dir.join(format!("{}.vtu", config.name)),
                    ExportVtuConfig {
                        title: config.name.clone(),
                        ghost: false,
                        stride: ExportStride::PerCell,
                    },
                )?;
            }

            search_status = match result {
                Ok(HorizonStatus::Converged) => config
                    .horizon
                    .on_search_converged
                    .execute(|| eyre!("horizon search converged"))?,
                Ok(HorizonStatus::ConvergedToZero) => config
                    .horizon
                    .on_search_converged_to_zero
                    .execute(|| eyre!("horizon search converged to zero"))?,
                Err(HorizonError::NormDiverged) => config
                    .horizon
                    .on_search_diverged
                    .execute(|| eyre!("horizon search norm diverged"))?,
                Err(HorizonError::SurfaceNotContained(pos)) => config
                    .horizon
                    .on_search_diverged
                    .execute(|| eyre!("horizon search surface not contained: {:?}", pos))?,
                Err(HorizonError::ReachedMaxSteps) => config
                    .horizon
                    .on_max_search_steps
                    .execute(|| eyre!("horizon search reached max steps without converging"))?,
                Err(other) => return Err(eyre::Report::new(other)),
            };

            search_index += 1;

            Ok(())
        })?;

        if let Some(status) = search_status {
            break 'evolve status;
        }

        // ***********************************
        // Step

        integrator
            .step(
                &mut mesh,
                ORDER,
                FieldConditions,
                FieldDerivs {
                    gauge: config.evolve.gauge,
                },
                h,
                fields.as_mut_slice(),
            )
            .unwrap();

        let lapse = mesh.bottom_left_value(fields.field(Field::Gauge(Gauge::Lapse)));

        step += 1;
        coord_time += h;
        proper_time += h * lapse;

        let proper_time_delta = h * lapse;
        let coord_time_delta = h;
        regrid_tracker.update(proper_time_delta, coord_time_delta, 1);
        visualize_tracker.update(proper_time_delta, coord_time_delta, 1);
        search_tracker.update(proper_time_delta, coord_time_delta, 1);

        node_pb.set_position(mesh.num_nodes() as u64);
        level_pb.set_position(mesh.num_levels() as u64);
        memory_pb.set_position(memory_usage as u64);

        step_pb.inc(1);
        step_pb.set_message(format!(
            "Step: {}, Proper Time {:.8}, Lapse {:.5e}",
            step, proper_time, lapse
        ));

        // There is a chance that lapse is now NaN, which should trigger collapse,
        // but might be interpreted as disspersion because `NaN > max_proper_time`
        // Check again
        if lapse.is_nan() || lapse.is_infinite() || lapse.abs() == 0.0 {
            println!("{}", style(format!("lapse collapses: {:.5e}", norm)).red());

            break 'evolve config
                .error_handler
                .on_norm_diverge
                .status_or_crash(|| eyre!("norm diverges during evolution"))?;
        }

        // Serialize those values
        history.write_record(RunRecord {
            step,
            time: coord_time,
            proper_time,
            lapse,
        })?;
    };

    m.clear()?;

    // Flush record at end of execution.
    history.flush()?;

    // Print status of run
    match status {
        Status::Disperse => println!("{}", style(format!("System disperses")).cyan()),
        Status::Collapse => println!("{}", style(format!("System collapses")).cyan()),
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
        HumanBytes((fields.estimate_heap_size() + integrator.estimate_heap_size()) as u64)
    );

    Ok(status)
}
