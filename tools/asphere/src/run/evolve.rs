use crate::{
    run::{SimulationInfo, Status, Subrun, config::Config},
    system::{
        CONFORMAL_CH, ConstraintRhs, FieldConditions, LAPSE_CH, NUM_CHANNELS, PSI_CH, TimeDerivs,
        find_mass, save_image,
    },
};
use aeon::{
    image::ImageMut,
    prelude::*,
    solver::{Integrator, Method},
};
use aeon_app::progress;
use console::style;
use datasize::DataSize as _;
use indicatif::{HumanBytes, HumanCount, HumanDuration, MultiProgress, ProgressBar};
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::{
    path::Path,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, serde::Serialize)]
pub struct DiagnosticInfo {
    proper_time: f64,
    time: f64,
    nodes: usize,
    dofs: usize,
    levels: usize,
    alpha: f64,
    mass: f64,
    psi: f64,
    constraint: f64,
}

pub fn save_csv_table<T: serde::Serialize>(records: &[T], path: &Path) -> eyre::Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    for record in records {
        writer.serialize(record)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn evolve_data(config: &Config, mesh: Mesh<1>, system: Image) -> eyre::Result<SimulationInfo> {
    // // Load diagnostics
    // let mut diagnostics = Diagnostics::default();
    // // Evolve
    // let result = evolve_data_with_diagnostics(config, &mut diagnostics, mesh, system);
    // // Flush diagnostics
    // diagnostics.flush(config)?;
    // // Bubble up result
    // result

    evolve_data_full(config, mesh, system, None)
}

pub fn evolve_data_full(
    config: &Config,
    mut mesh: Mesh<1>,
    mut system: Image,
    subrun: Option<&Subrun>,
) -> eyre::Result<SimulationInfo> {
    // Get start time of evolution
    let start = Instant::now();
    // Get output directory
    let absolute = config.directory()?;

    // Path for initial visualization data.
    if config.visualize.save_evolve || config.diagnostic.save_evolve {
        std::fs::create_dir_all(&absolute.join("evolve"))?;
    }

    let mut integrator = Integrator::new(Method::RK4KO6(config.evolve.dissipation));
    let mut time = 0.0;
    let mut step = 0;

    let mut proper_time = 0.0;

    let mut save_step = 0;
    let mut steps_since_regrid = 0;
    let mut time_since_save = 0.0;
    let mut fixed_grid = false;

    // Create progress bars, if we are not performing a subrun
    let bars = if subrun.is_none() {
        let m = MultiProgress::new();
        let node_pb = m.add(ProgressBar::new(config.limits.max_nodes as u64));
        node_pb.set_style(progress::node_style());
        node_pb.set_prefix("[Nodes] ");
        node_pb.enable_steady_tick(Duration::from_millis(100));
        let memory_pb = m.add(ProgressBar::new(config.limits.max_memory as u64));
        memory_pb.set_style(progress::byte_style());
        memory_pb.set_prefix("[Memory]");
        memory_pb.enable_steady_tick(Duration::from_millis(100));
        let level_pb = m.add(ProgressBar::new(config.limits.max_levels as u64));
        level_pb.set_style(progress::level_style());
        level_pb.set_prefix("[Level] ");
        level_pb.enable_steady_tick(Duration::from_millis(100));
        // Step spinner
        let step_pb = m.add(ProgressBar::no_length());
        step_pb.set_style(progress::spinner_style());
        step_pb.set_prefix("[Step] ");
        step_pb.enable_steady_tick(Duration::from_millis(100));
        // Output progress bars.
        Some((m, node_pb, memory_pb, level_pb, step_pb))
    } else {
        None
    };

    let subbar = subrun.map(|m| {
        let pb = m
            .multi
            .add(ProgressBar::new(config.evolve.max_steps as u64));
        pb.set_style(progress::run_style());
        pb.set_prefix(format!("[P = {:.16}]", m.parameter));
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    });

    let mut disperse = true;
    let mut mass_queue = AllocRingBuffer::new(50);

    let mut min_alpha = f64::INFINITY;
    let mut min_alpha_mass = 0.0;
    let mut min_alpha_proper_time = 0.0;

    let mut deriv_buffer = Vec::<f64>::default();
    let mut deriv_buffers: [Vec<f64>; 5] = Default::default();
    let mut constraint_buffers: [Vec<f64>; 5] = Default::default();
    let mut buffers_filled: [bool; 5] = [false; 5];
    let mut buffer_index = 0;

    let mut constraint: f64 = 0.0;
    let mut constraint_linf: f64 = 0.0;
    let mut max_constraint: f64 = 0.0;
    let mut max_constraint_linf: f64 = 0.0;
    // let mut constraint_output_index: usize = 0;

    let mut max_nodes = 0;
    let mut max_dofs = 0;

    let mut diagnostic = Vec::new();
    let mut collapse_msg = "".to_string();

    while proper_time < config.evolve.max_proper_time {
        assert!(system.num_nodes() == mesh.num_nodes());
        mesh.fill_boundary(4, FieldConditions, system.as_mut());

        // Fill current derive buffer
        deriv_buffers[buffer_index % 5].resize(mesh.num_nodes(), 0.0);
        constraint_buffers[buffer_index % 5].resize(mesh.num_nodes(), 0.0);
        deriv_buffers[buffer_index % 5].copy_from_slice(system.channel(CONFORMAL_CH));
        mesh.evaluate(
            4,
            ConstraintRhs,
            system.as_ref(),
            ImageMut::from(constraint_buffers[buffer_index % 5].as_mut_slice()),
        )
        .unwrap();

        buffers_filled[buffer_index % 5] = true;

        // Check Norm
        let norm = mesh.l2_norm_system(system.as_ref());

        if norm.is_nan() || norm >= 1e60 {
            // println!("Evolution collapses, norm: {}", norm);
            collapse_msg = format!("Norm blows up: {}", norm);
            disperse = false;
            break;
        }

        max_constraint = max_constraint.max(constraint);
        max_constraint_linf = max_constraint_linf.max(constraint_linf);
        max_nodes = max_nodes.max(mesh.num_nodes());
        max_dofs = max_dofs.max(mesh.num_dofs());
        if constraint >= config.evolve.max_constraint {
            collapse_msg = format!("Max constraint reached: {}", max_constraint);
            disperse = false;
            break;
        }

        if step >= config.evolve.max_steps {
            collapse_msg = format!("Evolution exceded maximum allocated steps: {}", step);
            disperse = false;
            break;
        }

        if mesh.num_nodes() >= config.limits.max_nodes {
            collapse_msg = format!(
                "Evolution exceded maximum allocated nodes: {}",
                mesh.num_nodes()
            );
            disperse = false;
            break;
        }

        let memory_usage = system.estimate_heap_size()
            + integrator.estimate_heap_size()
            + mesh.estimate_heap_size();

        let h = mesh.min_spacing() * config.evolve.cfl;

        // Perform regridding to a fixed value if configured as such, and if at a late enough time
        if config.regrid.fix_grid && proper_time >= config.regrid.fix_grid_time && !fixed_grid {
            fixed_grid = true;

            // Loop through regridding steps until we reach the desired level
            loop {
                // Check if we are at the desired level
                let mut min_level = usize::MAX;
                let mut max_level = usize::MIN;
                for cell in mesh.tree().active_cell_indices() {
                    let ll = mesh.tree().active_level(cell);
                    let cc = mesh.tree().active_bounds(cell).center()[0]; // get 1st element because we only have 1 dimension
                    if cc < config.regrid.fix_grid_radius {
                        if ll < min_level {
                            min_level = ll;
                        }
                        if ll > max_level {
                            max_level = ll;
                        }
                    }
                }
                if min_level == config.regrid.fix_grid_level
                    && max_level == config.regrid.fix_grid_level
                {
                    break;
                }
                // // If radius is smaller than the smallest cell, what do we do?
                let mut refine_inner = false;
                let mut coarsen_inner = false;
                if min_level == usize::MAX || max_level == usize::MIN {
                    // If too refined, then coarsen the innermost
                    if mesh.num_levels() > config.regrid.fix_grid_level {
                        coarsen_inner = true;
                    // If too coarse, then refine the innermost
                    } else if mesh.num_levels() < config.regrid.fix_grid_level {
                        refine_inner = true;
                    // If at max refinement, then just stop
                    } else {
                        break;
                    }
                }

                // Perform constraint assessment
                if buffers_filled.iter().all(|&f| f) {
                    deriv_buffer.resize(mesh.num_nodes(), 0.0);
                    deriv_buffer.fill(0.0);

                    const WEIGHTS: [f64; 5] = [-1. / 12., 2. / 3., 0.0, -2. / 3., 1. / 12.];

                    for si in 0..5 {
                        let part = deriv_buffers[(buffer_index - si) % 5].as_slice();

                        assert!(part.len() == deriv_buffer.len());

                        for i in 0..deriv_buffer.len() {
                            deriv_buffer[i] += WEIGHTS[si] * part[i]
                        }
                    }

                    for i in 0..deriv_buffer.len() {
                        deriv_buffer[i] /= h;
                    }

                    let constraint_buffer = constraint_buffers[(buffer_index - 2) % 5].as_slice();
                    assert!(deriv_buffer.len() == constraint_buffer.len());

                    for i in 0..deriv_buffer.len() {
                        deriv_buffer[i] -= constraint_buffer[i];
                    }

                    constraint = mesh.l2_norm(&deriv_buffer);
                    constraint_linf = mesh.max_norm(&deriv_buffer);
                }

                // Take one regridding step towards desired level
                if refine_inner {
                    mesh.refine_innermost();
                } else if coarsen_inner {
                    mesh.coarsen_innermost();
                } else {
                    mesh.regrid_in_radius(
                        config.regrid.fix_grid_radius,
                        config.regrid.fix_grid_level,
                    );
                }

                // Copy system into tmp scratch space (provided by dissipation).
                let scratch = integrator.scratch(system.storage().len());
                scratch.copy_from_slice(system.storage());
                system.resize(mesh.num_nodes());
                mesh.transfer_system(
                    4,
                    ImageRef::from_storage(&scratch, NUM_CHANNELS),
                    system.as_mut(),
                );

                buffers_filled.fill(false);
            }

            continue;
        }

        // Perform normal regridding step
        if steps_since_regrid > config.regrid.flag_interval && !fixed_grid {
            steps_since_regrid = 0;

            // Perform constraint assessment
            if buffers_filled.iter().all(|&f| f) {
                deriv_buffer.resize(mesh.num_nodes(), 0.0);
                deriv_buffer.fill(0.0);

                const WEIGHTS: [f64; 5] = [-1. / 12., 2. / 3., 0.0, -2. / 3., 1. / 12.];

                for si in 0..5 {
                    let part = deriv_buffers[(buffer_index - si) % 5].as_slice();

                    assert!(part.len() == deriv_buffer.len());

                    for i in 0..deriv_buffer.len() {
                        deriv_buffer[i] += WEIGHTS[si] * part[i]
                    }
                }

                for i in 0..deriv_buffer.len() {
                    deriv_buffer[i] /= h;
                }

                let constraint_buffer = constraint_buffers[(buffer_index - 2) % 5].as_slice();
                assert!(deriv_buffer.len() == constraint_buffer.len());

                for i in 0..deriv_buffer.len() {
                    deriv_buffer[i] -= constraint_buffer[i];
                }

                // let mut checkpoint = Checkpoint::default();
                // checkpoint.attach_mesh(&mesh);
                // save_image(&mut checkpoint, system.as_ref());
                // checkpoint.save_field("Constraint", &deriv_buffer);
                // checkpoint.export_vtu(
                //     absolute
                //         .join("evolve")
                //         .join(format!("constraint_{constraint_output_index}.vtu")),
                //     ExportVtuConfig {
                //         title: "Masslesss Scalar Field Evolution".to_string(),
                //         ghost: false,
                //         stride: config.visualize.stride,
                //     },
                // )?;
                // checkpoint.export_csv(
                //     absolute
                //         .join("evolve")
                //         .join(format!("constraint_{constraint_output_index}.csv")),
                //     ExportStride::PerVertex,
                // )?;

                constraint = mesh.l2_norm(&deriv_buffer);
                constraint_linf = mesh.max_norm(&deriv_buffer);

                // constraint_output_index += 1;
            }

            mesh.flag_wavelets(
                4,
                config.regrid.coarsen_error,
                config.regrid.refine_error,
                system.as_ref(),
            );
            mesh.limit_level_range_flags(1, config.limits.max_levels - 1);
            mesh.balance_flags();
            mesh.regrid();

            log::trace!(
                "Regridded Mesh at time: {proper_time:.5}, Num Levels {}, Num Nodes {}, Step: {}",
                mesh.num_levels(),
                mesh.num_nodes(),
                step,
            );

            // Copy system into tmp scratch space (provided by dissipation).
            let scratch = integrator.scratch(system.storage().len());
            scratch.copy_from_slice(system.storage());
            system.resize(mesh.num_nodes());
            mesh.transfer_system(
                4,
                ImageRef::from_storage(&scratch, NUM_CHANNELS),
                system.as_mut(),
            );

            buffers_filled.fill(false);

            continue;
        }

        if config.visualize.save_evolve && time_since_save >= config.visualize.save_evolve_interval
        {
            time_since_save -= config.visualize.save_evolve_interval;

            log::trace!(
                "Saving Checkpoint {save_step}, Time: {time:.5}, Dilated Time: {proper_time:.5}, Step: {step}, Norm: {norm:.5e}, Nodes: {}",
                mesh.num_nodes()
            );

            // Output current system to disk
            let mut checkpoint = Checkpoint::default();
            checkpoint.attach_mesh(&mesh);
            save_image(&mut checkpoint, system.as_ref());
            checkpoint.export_vtu(
                absolute
                    .join("evolve")
                    .join(format!("{}_{save_step}.vtu", config.name)),
                ExportVtuConfig {
                    title: "Masslesss Scalar Field Evolution".to_string(),
                    ghost: false,
                    stride: config.visualize.stride,
                },
            )?;

            save_step += 1;
        }

        // Compute lapse and mass before running diagnostic
        let alpha = mesh.bottom_left_value(system.channel(LAPSE_CH));
        let psi = mesh.bottom_left_value(system.channel(PSI_CH));
        let mass = find_mass(&mesh, system.as_ref());

        if config.diagnostic.save_evolve && step % config.diagnostic.save_evolve_interval == 0 {
            diagnostic.push(DiagnosticInfo {
                proper_time,
                time,
                nodes: mesh.num_nodes(),
                dofs: mesh.num_dofs(),
                alpha,
                mass,
                psi,
                levels: mesh.num_levels(),
                constraint,
            });
        }

        if alpha.abs() <= min_alpha {
            min_alpha = alpha.abs();
            min_alpha_mass = mass;
            min_alpha_proper_time = proper_time;
        }

        // Crash if min lapse achieved
        if alpha <= config.evolve.min_lapse {
            collapse_msg = format!("Minimum Lapse achieved {}", alpha);
            disperse = false;
            break;
        }

        // Compute step
        integrator
            .step(
                &mut mesh,
                4,
                FieldConditions,
                TimeDerivs,
                h,
                system.as_mut(),
            )
            .unwrap();

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;

        proper_time += h * alpha;

        buffer_index += 1;

        if let Some((_, node_pb, memory_pb, level_pb, step_pb)) = &bars {
            node_pb.set_position(mesh.num_nodes() as u64);
            level_pb.set_position(mesh.num_levels() as u64);
            memory_pb.set_position(memory_usage as u64);
            step_pb.inc(1);
            step_pb.set_message(format!(
                "Step: {}, τ: {:.8}, M: {:.8e}, α: {:.8e}, ℂ: {:.4e}",
                step, proper_time, mass, alpha, constraint
            ));
        }

        if let Some(bar) = &subbar {
            bar.set_position(step as u64);
            bar.set_message(format!(
                "τ: {:.8}, α: {:.8e}, ℂ: {:.4e}",
                proper_time, alpha, constraint
            ));
        }

        mass_queue.push(mass);

        let norm = mesh.l2_norm_system(system.as_ref());

        if norm.is_nan() || norm >= 1e60 || alpha.is_nan() || alpha <= 0.0 {
            collapse_msg = format!(
                "Invalid Fields after update. Norm: {}, Alpha: {}",
                norm, alpha
            );
            disperse = false;
            break;
        }
    }

    if let Some((m, _, _, _, _)) = bars {
        m.clear()?;
    }

    if config.diagnostic.save_evolve {
        save_csv_table(
            &diagnostic,
            &absolute.join("evolve").join("diagnostics.csv"),
        )?;
    }

    let alpha = min_alpha;
    let mut mass = min_alpha_mass;

    if !disperse {
        proper_time = min_alpha_proper_time;
    }

    const USE_MIN_ALPHA_MASS: bool = false;

    if !USE_MIN_ALPHA_MASS {
        mass = *mass_queue.front().unwrap();
    }

    // let alpha = mesh.bottom_left_value(system.field(Field::Lapse));

    if let Some(bar) = &subbar {
        let status = if disperse {
            style("Disperses").green()
        } else {
            style("Collapses").red()
        };

        let mass = if disperse { 0.0 } else { mass };
        bar.abandon_with_message(format!(
            "{}, τ: {}, M: {}, ℂ: {:.4e}",
            status, proper_time, mass, max_constraint,
        ));
    }

    // Only log if not running as subrun
    if subrun.is_none() {
        println!(
            "Final evolution takes {}, {} steps",
            HumanDuration(start.elapsed()),
            HumanCount(step as u64),
        );

        let status = if disperse {
            style("Disperses").green()
        } else {
            style("Collapses").red()
        };
        println!("Run Status: {}, Mass: {}, Lapse: {}", status, mass, alpha);
        if !disperse {
            println!("Reason for Collapse: {}", collapse_msg);
        }

        println!(
            "Nodes: {}, Dofs: {}, Max constraint: {}",
            max_nodes, max_dofs, max_constraint
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
            HumanBytes((system.estimate_heap_size() + integrator.estimate_heap_size()) as u64)
        );
    }

    Ok(match disperse {
        true => SimulationInfo {
            status: Status::Disperse,
            mass: 0.0,
        },
        false => SimulationInfo {
            status: Status::Collapse,
            mass,
        },
    })
}
