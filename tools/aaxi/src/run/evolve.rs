use crate::{
    eqs::{
        DynamicalData, DynamicalDerivs, GaugeCondition, ScalarFieldData, ScalarFieldDerivs,
        evolution,
    },
    horizon::{self, ApparentHorizonFinder, HorizonError, HorizonProjection, HorizonStatus},
    run::{
        config::Config,
        history::{History, HistoryInfo},
        interval::{Interval, IntervalTracker},
        status::{Status, Strategy},
    },
    systems::*,
};
use aeon::{
    prelude::*,
    solver::{Integrator, Method, SolverCallback},
};
use aeon_app::progress;
use console::style;
use datasize::DataSize as _;
use eyre::eyre;
use indicatif::{HumanBytes, HumanCount, HumanDuration, MultiProgress, ProgressBar};
use std::{
    convert::Infallible,
    path::Path,
    time::{Duration, Instant},
};

#[derive(Clone)]
struct FieldDerivs<'a> {
    gauge: GaugeCondition,
    masses: &'a [f64],
}

impl<'a> Function<2> for FieldDerivs<'a> {
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: ImageRef,
        mut output: ImageMut,
    ) -> Result<(), Infallible> {
        let grr_f = input.channel(GRR_CH);
        let grz_f = input.channel(GRZ_CH);
        let gzz_f = input.channel(GZZ_CH);
        let s_f = input.channel(S_CH);

        let krr_f = input.channel(KRR_CH);
        let krz_f = input.channel(KRZ_CH);
        let kzz_f = input.channel(KZZ_CH);
        let y_f = input.channel(Y_CH);

        let lapse_f = input.channel(LAPSE_CH);
        let shiftr_f = input.channel(SHIFTR_CH);
        let shiftz_f = input.channel(SHIFTZ_CH);

        let theta_f = input.channel(THETA_CH);
        let zr_f = input.channel(ZR_CH);
        let zz_f = input.channel(ZZ_CH);

        let num_scalar_fields = self.masses.len();
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

            for (i, &mass) in self.masses.iter().enumerate() {
                let phi = input.channel(phi_ch(i));
                let pi = input.channel(pi_ch(i));

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

            output.channel_mut(GRR_CH)[index] = derivs.grr_t;
            output.channel_mut(GRZ_CH)[index] = derivs.grz_t;
            output.channel_mut(GZZ_CH)[index] = derivs.gzz_t;
            output.channel_mut(S_CH)[index] = derivs.s_t;

            output.channel_mut(KRR_CH)[index] = derivs.krr_t;
            output.channel_mut(KRZ_CH)[index] = derivs.krz_t;
            output.channel_mut(KZZ_CH)[index] = derivs.kzz_t;
            output.channel_mut(Y_CH)[index] = derivs.y_t;

            output.channel_mut(THETA_CH)[index] = derivs.theta_t;
            output.channel_mut(ZR_CH)[index] = derivs.zr_t;
            output.channel_mut(ZZ_CH)[index] = derivs.zz_t;

            output.channel_mut(LAPSE_CH)[index] = derivs.lapse_t;
            output.channel_mut(SHIFTR_CH)[index] = derivs.shiftr_t;
            output.channel_mut(SHIFTZ_CH)[index] = derivs.shiftz_t;

            for i in 0..num_scalar_fields {
                let derivs = &scalar_field_derivs[i];
                output.channel_mut(phi_ch(i))[index] = derivs.phi;
                output.channel_mut(pi_ch(i))[index] = derivs.pi;
            }
        }

        Ok(())
    }
}

pub struct HorizonCallback<'a> {
    directory: &'a Path,
    positions: &'a mut Vec<[f64; 2]>,
    config: &'a Config,
    pb: Option<&'a mut ProgressBar>,
}

impl<'a> SolverCallback<1> for HorizonCallback<'a> {
    type Error = std::io::Error;

    fn callback(
        &mut self,
        surface: &Mesh<1>,
        radius: ImageRef,
        _output: ImageRef,
        iteration: usize,
    ) -> Result<(), Self::Error> {
        if let Some(pb) = &self.pb {
            pb.set_position(iteration as u64);
        }

        if !self.config.visualize.horizon_relax {
            return Ok(());
        }

        let interval = self.config.visualize.horizon_relax_interval.unwrap_steps();

        if iteration % interval != 0 {
            return Ok(());
        }

        let save_index = iteration / interval;
        let radius = radius.channel(0);

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

struct Progress {
    m: MultiProgress,
    node_pb: ProgressBar,
    level_pb: ProgressBar,
    memory_pb: ProgressBar,
    step_pb: ProgressBar,
}

struct Incremental {
    tracker: IntervalTracker,
    interval: Interval,
    max_nodes: usize,
    max_levels: usize,
    max_memory: usize,
}

enum Spinners {
    Progress(Progress),
    Incremental(Incremental),
}

impl Spinners {
    fn progress(max_nodes: usize, max_levels: usize, max_memory: usize) -> Self {
        // Setup progress bars
        let m = MultiProgress::new();
        // Max nodes
        let node_pb = m.add(ProgressBar::new(max_nodes as u64));
        node_pb.set_style(progress::node_style());
        node_pb.enable_steady_tick(Duration::from_millis(100));
        node_pb.set_prefix("[Node]  ");
        // Max levels
        let level_pb = m.add(ProgressBar::new(max_levels as u64));
        level_pb.set_style(progress::level_style());
        level_pb.enable_steady_tick(Duration::from_millis(100));
        level_pb.set_prefix("[Level] ");
        // Memory usage
        let memory_pb = m.add(ProgressBar::new(max_memory as u64));
        memory_pb.set_style(progress::byte_style());
        memory_pb.enable_steady_tick(Duration::from_millis(100));
        memory_pb.set_prefix("[Memory]");
        // Step spinner
        let step_pb = m.add(ProgressBar::no_length());
        step_pb.set_style(progress::spinner_style());
        step_pb.enable_steady_tick(Duration::from_millis(100));
        step_pb.set_prefix("[Step] ");

        Self::Progress(Progress {
            m,
            node_pb,
            level_pb,
            memory_pb,
            step_pb,
        })
    }

    fn incremental(
        max_nodes: usize,
        max_levels: usize,
        max_memory: usize,
        interval: Interval,
    ) -> Self {
        Self::Incremental(Incremental {
            interval,
            tracker: IntervalTracker::new(),
            max_nodes,
            max_levels,
            max_memory,
        })
    }

    fn multiprogress(&self) -> Option<&MultiProgress> {
        if let Self::Progress(progress) = self {
            Some(&progress.m)
        } else {
            None
        }
    }

    fn update(
        &mut self,
        step: usize,
        proper_time: f64,
        coord_time: f64,
        proper_time_delta: f64,
        coord_time_delta: f64,
        num_nodes: usize,
        num_levels: usize,
        memory_usage: usize,
        lapse: f64,
    ) {
        match self {
            Spinners::Progress(progress) => {
                progress.node_pb.set_position(num_nodes as u64);
                progress.level_pb.set_position(num_levels as u64);
                progress.memory_pb.set_position(memory_usage as u64);

                progress.step_pb.inc(1);
                progress.step_pb.set_message(format!(
                    "Step: {}, Proper Time {:.8}, Lapse {:.5e}",
                    step, proper_time, lapse
                ));
            }
            Spinners::Incremental(inc) => {
                inc.tracker.every(inc.interval, || {
                    let node_percentage = num_nodes as f64 / inc.max_nodes as f64 * 100.0;
                    let level_percentage = num_levels as f64 / inc.max_levels as f64 * 100.0;
                    let memory_percentage = memory_usage as f64 / inc.max_memory as f64 * 100.0;

                    log::info!(
                        "Step: {}, Proper Time: {:.8}, Coord Time: {:.8}",
                        step,
                        proper_time,
                        coord_time
                    );
                    log::info!(
                        "- Nodes: {} ({:.2}%), Levels: {} ({:.2}%), Ram Usage: {} ({:.2}%)",
                        num_nodes,
                        node_percentage,
                        num_levels,
                        level_percentage,
                        memory_usage,
                        memory_percentage
                    );
                    log::info!("- Lapse: {:.6e}", lapse)
                });
                inc.tracker.update(proper_time_delta, coord_time_delta, 1);
            }
        }
    }

    fn clear(&self) -> eyre::Result<()> {
        if let Self::Progress(progress) = self {
            progress.m.clear()?;
        }

        Ok(())
    }
}

pub fn evolve_data(config: &Config, mut mesh: Mesh<2>, mut fields: Image) -> eyre::Result<Status> {
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
    let scalar_fields = config.scalar_field_masses();
    let num_scalar_fields = scalar_fields.len();
    let num_channels = num_channels(num_scalar_fields);

    // Integrate
    let mut integrator = Integrator::new(Method::RK4KO6(config.evolve.dissipation));

    let mut step = 0;
    let mut coord_time = 0.0;
    let mut proper_time = 0.0;

    let max_levels = config.limits.max_levels;
    let max_nodes = config.limits.max_nodes;
    let max_memory = config.limits.max_memory;
    let max_wall_time = config.limits.max_wall_time;

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
    // Origin Output

    let mut history_tracker = IntervalTracker::new();
    let history_interval = config.history.evolve_interval;
    let mut history = if config.history.evolve {
        History::output(&output.join("evolve_history.csv"))?
    } else {
        History::empty()
    };

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

    log::info!("Evolving data");

    // Create spinner outputs
    let mut spinners = match config.logging {
        crate::run::config::Logging::Progress => Spinners::progress(
            config.limits.max_nodes,
            config.limits.max_levels,
            config.limits.max_memory,
        ),
        crate::run::config::Logging::Incremental {
            evolve: interval, ..
        } => Spinners::incremental(
            config.limits.max_nodes,
            config.limits.max_levels,
            config.limits.max_memory,
            interval,
        ),
    };

    let status: Status = 'evolve: loop {
        assert!(fields.num_nodes() == mesh.num_nodes());

        // ****************************
        // Coordinate time

        if coord_time > max_coord_time {
            log::warn!(
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
            log::warn!(
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
            log::warn!(
                "{}",
                style(format!("Evolution reached max steps: {}", step)).green()
            );

            break 'evolve config
                .error_handler
                .on_max_evolve_steps
                .status_or_crash(|| eyre!("evolution reached max steps"))?;
        }

        // *********************************
        // Max nodes, levels, memory, wall time

        let memory_usage = fields.estimate_heap_size()
            + integrator.estimate_heap_size()
            + mesh.estimate_heap_size();

        if start.elapsed().as_secs() as usize > max_wall_time {
            log::warn!(
                "{}",
                style(format!(
                    "Wall time exceded maximum alotted wall time: {:?}",
                    start.elapsed()
                ))
                .red()
            );
            break 'evolve config
                .error_handler
                .on_max_wall_time
                .status_or_crash(|| eyre!("wall time exceded maximum alotted wall time"))?;
        }

        if mesh.num_nodes() > max_nodes {
            log::warn!(
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
            log::warn!(
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
                log::warn!(
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
        mesh.fill_boundary(4, FieldConditions, fields.as_mut());

        // *******************************
        // Norm

        let norm = mesh.l2_norm_system(fields.as_ref());

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
            mesh.flag_wavelets(4, lower, upper, fields.as_ref());
            mesh.balance_flags();

            if config.error_handler.on_max_levels == Strategy::Ignore {
                mesh.limit_level_range_flags(0, max_levels - 1);
                mesh.balance_flags();
            }

            mesh.regrid();

            // Copy system into tmp scratch space (provieded by dissipation).
            let scratch = integrator.scratch(fields.storage().len());
            scratch.copy_from_slice(fields.storage());
            fields.resize(mesh.num_nodes());
            mesh.transfer_system(
                4,
                ImageRef::from_storage(&scratch, num_channels),
                fields.as_mut(),
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
                fields.as_ref(),
                ImageMut::from(horizon_field.as_mut_slice()),
            )
            .unwrap();

            // Output current system to disk
            let mut checkpoint = Checkpoint::default();
            checkpoint.attach_mesh(&mesh);
            checkpoint.save_field("Horizon", &horizon_field);
            save_image(&mut checkpoint, fields.as_ref());
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

        history_tracker.try_every(history_interval, || -> std::io::Result<()> {
            history.write_record(HistoryInfo {
                proper_time,
                coord_time,
                nodes: mesh.num_nodes(),
                dofs: mesh.num_dofs(),
                levels: mesh.num_levels(),
                alpha: mesh.bottom_left_value(fields.channel(LAPSE_CH)),
                grr: mesh.bottom_left_value(fields.channel(GRR_CH)),
                grz: mesh.bottom_left_value(fields.channel(GRZ_CH)),
                gzz: mesh.bottom_left_value(fields.channel(GZZ_CH)),
                theta: mesh.bottom_left_value(fields.channel(THETA_CH)),
            })?;

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

            let mut pb = spinners.multiprogress().map(|m| {
                let pb = m.add(ProgressBar::new(config.horizon.relax.max_steps as u64));
                pb.set_style(progress::node_style());
                pb.enable_steady_tick(Duration::from_millis(100));
                pb.set_prefix("- [Horizon Search]");
                pb
            });

            let result = finder.search_with_callback(
                &mesh,
                fields.as_ref(),
                4,
                surface,
                HorizonCallback {
                    directory: horizon_dir.as_path(),
                    positions: &mut search_positions,
                    config: &config,
                    pb: pb.as_mut(),
                },
                surface_radius,
            );

            pb.as_ref().map(ProgressBar::finish_and_clear);

            if config.visualize.horizon_relax {
                horizon_field.resize(mesh.num_nodes(), 0.0);

                mesh.evaluate(
                    4,
                    HorizonProjection,
                    fields.as_ref(),
                    horizon_field.as_mut_slice().into(),
                )?;

                // Output current system to disk, for reference in horizon search.
                let mut checkpoint = Checkpoint::default();
                checkpoint.attach_mesh(&mesh);
                save_image(&mut checkpoint, fields.as_ref());
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
                4,
                FieldConditions,
                FieldDerivs {
                    gauge: config.evolve.gauge,
                    masses: &scalar_fields,
                },
                h,
                fields.as_mut(),
            )
            .unwrap();

        let lapse = mesh.bottom_left_value(fields.channel(LAPSE_CH));

        step += 1;
        coord_time += h;
        proper_time += h * lapse;

        let proper_time_delta = h * lapse;
        let coord_time_delta = h;
        regrid_tracker.update(proper_time_delta, coord_time_delta, 1);
        visualize_tracker.update(proper_time_delta, coord_time_delta, 1);
        history_tracker.update(proper_time_delta, coord_time_delta, 1);
        search_tracker.update(proper_time_delta, coord_time_delta, 1);

        spinners.update(
            step,
            proper_time,
            coord_time,
            proper_time_delta,
            coord_time_delta,
            mesh.num_nodes(),
            mesh.num_levels(),
            memory_usage,
            lapse,
        );

        // There is a chance that lapse is now NaN, which should trigger collapse,
        // but might be interpreted as disspersion because `NaN > max_proper_time`
        // Check again
        if lapse.is_nan() || lapse.is_infinite() || lapse <= 0.0 {
            log::warn!("{}", style(format!("lapse collapses: {:.5e}", norm)).red());

            break 'evolve config
                .error_handler
                .on_norm_diverge
                .status_or_crash(|| eyre!("norm diverges during evolution"))?;
        }

        // Check if lapse is too small, which indicates collapse
        if lapse < 1e-7 {
            log::warn!("{}", style(format!("lapse too small: {:.5e}", lapse)).red());

            break 'evolve config
                .error_handler
                .on_min_lapse
                .status_or_crash(|| eyre!("lapse is too small during evolution"))?;
        }
    };

    spinners.clear()?;

    // Flush history file
    history.flush()?;

    // Print status of run
    match status {
        Status::Disperse => log::info!("{}", style(format!("Status: system disperses")).cyan()),
        Status::Collapse => log::info!("{}", style(format!("Status: system collapses")).cyan()),
    }

    log::info!(
        "Final evolution takes {}, {} steps",
        HumanDuration(start.elapsed()),
        HumanCount(step as u64),
    );
    log::info!(
        "Mesh Data: (Nodes: {}; Cells: {}; RAM usage: ~{}",
        mesh.num_nodes(),
        mesh.num_active_cells(),
        HumanBytes(mesh.estimate_heap_size() as u64)
    );
    log::info!(
        "Field Data: (RAM usage: ~{})",
        HumanBytes((fields.estimate_heap_size() + integrator.estimate_heap_size()) as u64)
    );

    Ok(status)
}
