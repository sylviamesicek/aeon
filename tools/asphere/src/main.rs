//! An executable for creating general initial data for numerical relativity simulations in 2D.
#![allow(unused_assignments)]

use aeon::{
    prelude::*,
    solver::{Integrator, Method},
};
use aeon_config::{ConfigVars, Transform as _};
use circular_queue::CircularQueue;
use clap::{Arg, ArgMatches, Command, arg, value_parser};
use console::style;
use core::f64;
use datasize::DataSize;
use eyre::{Context as _, eyre};
use indicatif::{HumanBytes, HumanCount, HumanDuration, MultiProgress, ProgressBar};
use std::{
    collections::HashMap,
    path::PathBuf,
    time::{Duration, Instant},
};
use std::{fmt::Write as _, num::ParseFloatError};

mod cole;
mod config;
mod misc;
mod system;

use cole::*;
use config::*;
use system::*;

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

        let serial_id = config.diagnostic.serial_id;

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

/// Solve for initial conditions and adaptively refine mesh
fn initial_data(config: &Config) -> eyre::Result<(Mesh<1>, SystemVec<Fields>)> {
    // Save initial time
    let start = Instant::now();
    // Get output directory
    let absolute = config.directory()?;

    eyre::ensure!(
        config.sources.len() == 1,
        "asphere currently only supports a single source term"
    );

    // Retrieve primary source
    let source = config.sources[0].clone();

    // Create output dir.
    std::fs::create_dir_all(&absolute)?;
    // Path for initial visualization data.
    if config.visualize.save_initial || config.visualize.save_initial_levels {
        std::fs::create_dir_all(&absolute.join("initial"))?;
    }

    // Build mesh
    let mut mesh = Mesh::new(
        Rectangle {
            size: [config.domain.radius],
            origin: [0.0],
        },
        6,
        3,
        FaceArray::from_sides([BoundaryClass::Ghost], [BoundaryClass::OneSided]),
    );
    // Perform global refinements
    for _ in 0..config.regrid.global {
        mesh.refine_global();
    }

    // Create system of fields
    let mut system = SystemVec::new(Fields);

    println!("Relaxing Initial Data");

    // Create progress bars
    let m = MultiProgress::new();
    let node_pb = m.add(ProgressBar::new(config.limits.max_nodes as u64));
    node_pb.set_style(misc::node_style());
    node_pb.set_prefix("[Nodes] ");
    let memory_pb = m.add(ProgressBar::new(config.limits.max_memory as u64));
    memory_pb.set_style(misc::byte_style());
    memory_pb.set_prefix("[Memory]");
    let level_pb = m.add(ProgressBar::new(config.limits.max_levels as u64));
    level_pb.set_style(misc::level_style());
    level_pb.set_prefix("[Level] ");

    // Adaptively solve and refine until we satisfy error requirement
    loop {
        // Resize system to current mesh
        system.resize(mesh.num_nodes());

        node_pb.set_length(mesh.num_nodes() as u64);
        memory_pb.set_length((mesh.estimate_heap_size() + system.estimate_heap_size()) as u64);
        level_pb.set_length(mesh.max_level() as u64);

        // Set initial data for scalar field.
        let scalar_field = generate_initial_scalar_field(
            &mut mesh,
            source.amplitude.unwrap(),
            source.sigma.unwrap(),
        );

        // Fill system using scalar field.
        mesh.evaluate(
            4,
            InitialData,
            (&scalar_field).into(),
            system.as_mut_slice(),
        )
        .unwrap();

        // Solve for conformal and lapse
        solve_constraints(&mut mesh, system.as_mut_slice());
        // Compute norm
        // let l2_norm: f64 = mesh.l2_norm_system(system.as_slice());
        // log::info!("Scalar Field Norm {}", l2_norm);
        // Save visualization
        if config.visualize.save_initial_levels {
            let mut checkpoint = Checkpoint::default();
            checkpoint.attach_mesh(&mesh);
            checkpoint.save_system(system.as_slice());
            checkpoint.export_vtu(
                absolute.join("initial").join(format!(
                    "{}_level{}.vtu",
                    config.name,
                    mesh.max_level()
                )),
                ExportVtuConfig {
                    title: "Massless Scalar Field Initial Data".to_string(),
                    ghost: false,
                    stride: 1,
                },
            )?;
        }

        if mesh.num_nodes() >= config.limits.max_nodes {
            log::error!(
                "Failed to solve initial data, level: {}, nodes: {}",
                mesh.max_level(),
                mesh.num_nodes()
            );
            return Err(eyre!("failed to refine within perscribed limits"));
        }

        mesh.flag_wavelets(4, 0.0, config.regrid.refine_error, system.as_slice());
        mesh.limit_level_range_flags(1, config.limits.max_levels);
        mesh.balance_flags();

        if !mesh.requires_regridding() {
            log::trace!(
                "Sucessfully refined mesh to give accuracy: {:.5e}",
                config.regrid.refine_error
            );
            break;
        }

        mesh.regrid();
    }

    m.clear()?;

    println!("Finished relaxing in {}", HumanDuration(start.elapsed()),);
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
        HumanBytes(system.estimate_heap_size() as u64)
    );

    if config.visualize.save_initial {
        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_system(system.as_slice());
        checkpoint.export_vtu(
            absolute.join("initial.vtu"),
            ExportVtuConfig {
                title: "Massless Scalar Field Initial".to_string(),
                ghost: false,
                stride: config.visualize.stride.into_int(),
            },
        )?;

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_system(system.as_slice());
        checkpoint.export_vtu(
            absolute
                .join("initial")
                .join(format!("{}.vtu", config.name)),
            ExportVtuConfig {
                title: "Massless Scalar Field Initial Data".to_string(),
                ghost: false,
                stride: config.visualize.stride.into_int(),
            },
        )?;
    }

    Ok((mesh, system))
}

fn evolve_data(
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
            level: mesh.max_level(),
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
            mesh.limit_level_range_flags(1, config.limits.max_levels);
            mesh.balance_flags();
            mesh.regrid();

            log::trace!(
                "Regrided Mesh at time: {proper_time:.5}, Max Level {}, Num Nodes {}, Step: {}",
                mesh.max_level(),
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
                    stride: 1,
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
        if config.diagnostic.save && step % config.diagnostic.save_interval == 0 {
            diagnostics.append(
                proper_time,
                Snapshot {
                    mass,
                    alpha,
                    phi: mesh.bottom_left_value(system.field(Field::Phi)),
                    level: mesh.max_level(),
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
        level_pb.set_position(mesh.max_level() as u64);
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

fn run(config: &Config, diagnostics: &mut Diagnostics) -> eyre::Result<()> {
    // Solve for initial data
    let (mesh, system) = initial_data(config)?;
    // Run evolution
    evolve_data(config, diagnostics, mesh, system)?;

    Ok(())
}

// Main function that can return an error
fn main() -> eyre::Result<()> {
    // Set up nice colored error handing.
    color_eyre::install()?;
    // Load configuration
    let command = Command::new("asphere")
        .about("A program for simulating GR in spherical symmetry.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("0.1.0")
        .config_args();
    // Find matches
    let matches = command.get_matches();
    // Load configuration file.
    let (config, vars) = parse_config(&matches)?;
    let config = config.transform(&vars)?;

    // Check that there is only one source term.
    eyre::ensure!(
        config.sources.len() == 1,
        "asphere currently only supports one source term"
    );

    let _source = config.sources[0].clone();

    // Basic info dumping
    println!("Simulation: {}", style(&config.name).green());
    println!(
        "Output Directory: {}",
        style(config.directory()?.display()).green()
    );
    println!("Domain: {:.5}", config.domain.radius);
    println!("Sources...");
    for source in &config.sources {
        source.println();
    }

    // Diagnostic object
    let mut diagnostics = Diagnostics::default();
    // Run simulation
    let result = run(&config, &mut diagnostics);
    // Write diagnostics to file
    diagnostics.flush(&config)?;

    // match result {
    //     Ok(_) => eprintln!("A = {:.15} Disperses", source.amplitude.unwrap()),
    //     Err(_) => eprintln!("A = {:.15} Collapses", source.amplitude.unwrap()),
    // }

    // Bubble up result
    result
}

// ******************************
// Helpers **********************
// ******************************

fn parse_config(matches: &ArgMatches) -> eyre::Result<(Config, ConfigVars)> {
    // First check if we are running the cole subcommand.
    if let Some(matches) = matches.subcommand_matches("cole") {
        let amplitude = matches
            .get_one::<String>("amp")
            .ok_or(eyre!("Failed to find amplitude positional argument"))?
            .parse::<f64>()
            .map_err(|_| eyre!("Failed to parse amplitude as float"))?
            .clone();

        let serial_id = matches
            .get_one::<String>("ser")
            .ok_or(eyre!("Failed to find serial_id positional argument"))?
            .parse::<usize>()
            .map_err(|_| eyre!("Failed to parse serial_id as int"))?
            .clone();

        return Ok((cole_config(amplitude, serial_id), ConfigVars::new()));
    }

    // Compute config path.
    let config_path = matches
        .get_one::<PathBuf>("config")
        .cloned()
        .ok_or_else(|| eyre!("failed to specify config argument"))?;
    let config_path = misc::abs_or_relative(&config_path)?;

    // Parse config file from toml.
    let config =
        misc::import_from_toml::<Config>(&config_path).context("Failed to parse config file")?;

    // Read positional arguments
    let positional_args: Vec<&str> = matches
        .get_many::<String>("positional")
        .into_iter()
        // .ok_or(eyre::eyre!("Unable to parse positional arguments"))?
        .flat_map(|v| v.into_iter().map(|s| s.as_str()))
        .collect();

    let vars = ConfigVars {
        positional: positional_args
            .into_iter()
            .map(|sbuf| sbuf.parse::<f64>())
            .collect::<Result<Vec<f64>, ParseFloatError>>()?,
        named: HashMap::new(),
    };

    Ok((config, vars))
}

/// Extension trait for defining helper methods on `clap::Command`.
trait CommandExt {
    fn config_args(self) -> Self;
}

impl CommandExt for Command {
    fn config_args(self) -> Self {
        self.subcommand_negates_reqs(true)
            .subcommand(
                Command::new("cole")
                    .arg(
                        Arg::new("amp")
                            .value_name("FLOAT")
                            .required(true)
                            .help("Amplitude of massless scalar field to simulate"),
                    )
                    .arg(
                        Arg::new("ser")
                            .value_name("INT")
                            .required(true)
                            .help("Serialization number for massless scalar field data"),
                    ),
            )
            .arg(
                arg!(-c --config <FILE> "Sets a custom config file")
                    .required(true)
                    .value_parser(value_parser!(PathBuf)),
            )
            .arg(
                arg!(<positional> ... "positional arguments referenced in config file")
                    .trailing_var_arg(true)
                    .required(false),
            )
    }
}
