use crate::{
    misc,
    run::config::Config,
    system::{Field, FieldConditions, Fields, TimeDerivs, find_mass},
};
use aeon::{
    prelude::*,
    solver::{Integrator, Method},
};
use circular_queue::CircularQueue;
use console::style;
use datasize::DataSize as _;
use eyre::eyre;
use indicatif::{HumanBytes, HumanCount, HumanDuration, MultiProgress, ProgressBar};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

struct Snapshot {
    mass: f64,
    alpha: f64,
    phi: f64,
    level: usize,
    nodes: usize,
}

#[derive(Default)]
struct Diagnostics {
    times: Vec<f64>,
    data: Vec<Snapshot>,
}

impl Diagnostics {
    fn append(&mut self, time: f64, data: Snapshot) {
        self.times.push(time);
        self.data.push(data);
    }

    fn flush(&self, config: &Config) -> eyre::Result<()> {
        if !config.diagnostic.save {
            return Ok(());
        }

        let directory = config.directory()?;

        let serial_id = config.diagnostic.serial_id.unwrap();

        let file1 = format!("Mass-{}-{}", 4, serial_id);
        let file2 = format!("Al-{}-{}", 4, serial_id);
        let file3 = format!("Level-{}-{}", 4, serial_id);
        let file4 = format!("Origin-{}.txt", serial_id);

        let mut data1 = String::new();
        let mut data2 = String::new();
        let mut data3 = String::new();
        let mut data4 = String::new();

        for (&time, data) in self.times.iter().zip(self.data.iter()) {
            writeln!(data1, "{} {}", time, data.mass)?;
            writeln!(data2, "{} {}", time, data.alpha)?;
            writeln!(data3, "{} {} {}", time, data.level, data.nodes)?;
            writeln!(
                data4,
                "{} {} {} {} {} {}",
                time, 0.0, data.alpha, data.alpha, data.phi, 0.0
            )?;
        }

        std::fs::write(directory.join(file1), data1)?;
        std::fs::write(directory.join(file2), data2)?;
        std::fs::write(directory.join(file3), data3)?;
        std::fs::write(directory.join(file4), data4)?;

        Ok(())
    }
}

pub fn evolve_data(config: &Config, mesh: Mesh<1>, system: SystemVec<Fields>) -> eyre::Result<()> {
    // Load diagnostics
    let mut diagnostics = Diagnostics::default();
    // Evolve
    let result = evolve_data_with_diagnostics(config, &mut diagnostics, mesh, system);
    // Flush diagnostics
    diagnostics.flush(config)?;
    // Bubble up result
    result
}

fn evolve_data_with_diagnostics(
    config: &Config,
    diagnostics: &mut Diagnostics,
    mut mesh: Mesh<1>,
    mut system: SystemVec<Fields>,
) -> eyre::Result<()> {
    // Get start time of evolution
    let start = Instant::now();
    // Get output directory
    let absolute = config.directory()?;

    // Create output dir.
    std::fs::create_dir_all(&absolute)?;
    // Path for initial visualization data.
    if config.visualize.save_evolve {
        std::fs::create_dir_all(&absolute.join("evolve"))?;
    }

    let mut integrator = Integrator::new(Method::RK4KO6(config.evolve.dissipation));
    let mut time = 0.0;
    let mut step = 0;

    let mut proper_time = 0.0;

    let mut save_step = 0;
    let mut steps_since_regrid = 0;
    let mut time_since_save = 0.0;

    diagnostics.append(
        proper_time,
        Snapshot {
            mass: find_mass(&mesh, system.as_slice()),
            alpha: mesh.bottom_left_value(system.field(Field::Lapse)),
            phi: mesh.bottom_left_value(system.field(Field::Phi)),
            level: mesh.num_levels(),
            nodes: mesh.num_nodes(),
        },
    );

    println!("Evolving Data");

    // Create progress bars
    let m = MultiProgress::new();
    let node_pb = m.add(ProgressBar::new(config.limits.max_nodes as u64));
    node_pb.set_style(misc::node_style());
    node_pb.set_prefix("[Nodes] ");
    node_pb.enable_steady_tick(Duration::from_millis(100));
    let memory_pb = m.add(ProgressBar::new(config.limits.max_memory as u64));
    memory_pb.set_style(misc::byte_style());
    memory_pb.set_prefix("[Memory]");
    memory_pb.enable_steady_tick(Duration::from_millis(100));
    let level_pb = m.add(ProgressBar::new(config.limits.max_levels as u64));
    level_pb.set_style(misc::level_style());
    level_pb.set_prefix("[Level] ");
    level_pb.enable_steady_tick(Duration::from_millis(100));
    // Step spinner
    let step_pb = m.add(ProgressBar::no_length());
    step_pb.set_style(misc::spinner_style());
    step_pb.set_prefix("[Step] ");
    step_pb.enable_steady_tick(Duration::from_millis(100));

    let mut disperse = true;

    let mut mass_queue = CircularQueue::with_capacity(20);

    while proper_time < config.evolve.max_proper_time {
        assert!(system.len() == mesh.num_nodes());
        mesh.fill_boundary(Order::<4>, FieldConditions, system.as_mut_slice());

        // Check Norm
        let norm = mesh.l2_norm_system(system.as_slice());

        if norm.is_nan() || norm >= 1e60 {
            println!("Evolution collapses, norm: {}", norm);
            disperse = false;
            break;
        }

        if step >= config.evolve.max_steps {
            println!("Evolution exceded maximum allocated steps: {}", step);
            disperse = false;
            break;
        }

        if mesh.num_nodes() >= config.limits.max_nodes {
            println!(
                "Evolution exceded maximum allocated nodes: {}",
                mesh.num_nodes()
            );
            disperse = false;
            break;
        }

        let memory_usage = system.estimate_heap_size()
            + integrator.estimate_heap_size()
            + mesh.estimate_heap_size();

        // if mesh.max_level() >= MAX_LEVELS {
        //     log::trace!(
        //         "Evolution collapses, Reached maximum allowed level of refinement: {}",
        //         mesh.max_level()
        //     );
        //     return Err(anyhow!(
        //         "reached maximum allowed level of refinement: {}",
        //         mesh.max_level()
        //     ));
        // }

        let h = mesh.min_spacing() * config.evolve.cfl;

        if steps_since_regrid > config.regrid.flag_interval {
            steps_since_regrid = 0;

            mesh.flag_wavelets(
                4,
                config.regrid.coarsen_error,
                config.regrid.refine_error,
                system.as_slice(),
            );
            mesh.limit_level_range_flags(1, config.limits.max_levels + 1);
            mesh.balance_flags();
            mesh.regrid();

            log::trace!(
                "Regrided Mesh at time: {proper_time:.5}, Max Level {}, Num Nodes {}, Step: {}",
                mesh.num_levels(),
                mesh.num_nodes(),
                step,
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
            checkpoint.save_system(system.as_slice());
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

        // Compute step
        integrator
            .step(
                &mut mesh,
                Order::<4>,
                FieldConditions,
                TimeDerivs,
                h,
                system.as_mut_slice(),
            )
            .unwrap();

        let alpha = mesh.bottom_left_value(system.field(Field::Lapse));
        let mass = find_mass(&mesh, system.as_slice());
        if config.diagnostic.save && step % config.diagnostic.save_interval.unwrap() == 0 {
            diagnostics.append(
                proper_time,
                Snapshot {
                    mass,
                    alpha,
                    phi: mesh.bottom_left_value(system.field(Field::Phi)),
                    level: mesh.num_levels(),
                    nodes: mesh.num_nodes(),
                },
            );
        }

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;

        proper_time += h * alpha;

        node_pb.set_position(mesh.num_nodes() as u64);
        level_pb.set_position(mesh.num_levels() as u64);
        memory_pb.set_position(memory_usage as u64);
        step_pb.inc(1);
        step_pb.set_message(format!(
            "Step: {}, Proper Time {:.8}, Mass {:.8e}",
            step, proper_time, mass
        ));
        mass_queue.push(mass);

        let norm = mesh.l2_norm_system(system.as_slice());

        if norm.is_nan() || norm >= 1e60 || alpha.is_nan() {
            println!("Evolution collapses after step, norm: {}", norm);
            return Err(eyre!("exceded max allotted steps for evolution: {}", step));
        }
    }

    m.clear()?;

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
    println!("Run Status: {}", status);

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

    // for mass in mass_queue.iter() {
    //     println!("Previous Mass: {:.8e}", mass);
    // }

    Ok(())
}
