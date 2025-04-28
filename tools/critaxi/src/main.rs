use eyre::{anyhow, Result, WrapErr};
use serde::{Deserialize, Serialize};
use sharedaxi::{
    import_from_path_arg, import_from_toml, CritConfig, EVConfig, GaugeCondition, IDConfig,
    Logging, Source, Visualize,
};
use std::{
    collections::HashMap,
    ffi::OsStr,
    path::PathBuf,
    process::{Command, ExitCode, ExitStatus},
};

#[derive(Serialize, Deserialize)]
enum InitialStatus {
    Success,
    Failure,
}

#[derive(Serialize, Deserialize)]
enum EvolutionStatus {
    Disperse,
    Collapse,
    Failure,
}

type InitialCache = HashMap<String, InitialStatus>;
type EvolutionCache = HashMap<String, EvolutionStatus>;

fn float_to_string(value: f64) -> String {
    format!("{:?}", value)
}

fn critical_search() -> Result<()> {
    let matches = clap::Command::new("critaxi")
        .about("A program for searching for critical points in a range using idgen and evgen.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("v0.1.0")
        .arg(
            clap::Arg::new("path")
                .help("Path of config file for searching for critical points")
                .value_name("PATH")
                .required(true),
        )
        .get_matches();

    // Load configuration
    let config = import_from_path_arg::<CritConfig>(&matches)?;

    // Load header data and defaults
    let output = PathBuf::from(
        config
            .output_dir
            .clone()
            .unwrap_or_else(|| format!("{}_output", &config.name)),
    );
    let name = config.name.clone();

    // Compute log filter level.
    let level = config.logging.filter();

    // Build enviornment logger.
    env_logger::builder().filter_level(level).init();
    // Find currect working directory
    let dir = std::env::current_dir().context("Failed to find current working directory")?;
    let absolute = if output.is_absolute() {
        output
    } else {
        dir.join(output)
    };

    // Log Header data.
    log::info!("Running Critical Search: {}", &config.name);
    log::info!("Logging Level: {} ", level);
    log::info!(
        "Output Directory: {}",
        absolute
            .to_str()
            .ok_or(anyhow!("Failed to find absolute output directory"))?
    );

    eyre::ensure!(
        config.domain.radius > 0.0 && config.domain.height > 0.0,
        "Domain must have positive non-zero radius and height"
    );

    log::info!(
        "Domain is {:.5} by {:.5}",
        config.domain.radius,
        config.domain.height
    );

    eyre::ensure!(
        config.domain.cell.subdivisions >= 2 * config.domain.cell.ghost,
        "Domain cell nodes must be >= 2 * padding"
    );

    let stride = config.domain.cell.subdivisions;

    eyre::ensure!(config.start < config.end);
    std::fs::create_dir_all(&absolute)?;
    std::fs::create_dir_all(&absolute.join("config"))?;
    std::fs::create_dir_all(&absolute.join("evolve"))?;
    std::fs::create_dir_all(&absolute.join("initial"))?;

    let mut initial_cache = InitialCache::default();
    let mut evolution_cache = EvolutionCache::default();

    if config.cache_initial {
        initial_cache =
            import_from_toml::<InitialCache>(absolute.join("initial").join("cache.toml"))
                .unwrap_or(InitialCache::default());
    }

    if config.cache_evolve {
        evolution_cache =
            import_from_toml::<EvolutionCache>(absolute.join("evolve").join("cache.toml"))
                .unwrap_or(EvolutionCache::default());
    }

    let mut range = config.start..config.end;

    for _ in 0..config.bifurcations {
        let diff = range.end - range.start;
        let step = diff / (config.subsearches - 1) as f64;

        let subsearches = (0..config.subsearches)
            .map(|i| range.start + step * i as f64)
            .collect::<Vec<_>>();

        log::info!(
            "Bifurcating range {} to {} with {} subsearches",
            range.start,
            range.end,
            config.subsearches
        );

        for amplitude in subsearches.iter() {
            let idconfig = IDConfig {
                name: "initial".to_string(),
                output_dir: Some(
                    absolute
                        .join("initial")
                        .join(format!("{}_{:?}", name, amplitude))
                        .as_os_str()
                        .to_string_lossy()
                        .into_owned(),
                ),
                logging: Logging {
                    level: Logging::TRACE,
                },
                order: 4,

                visualize_levels: false,
                visualize_result: true,
                visualize_relax: false,
                visualize_every: 1,
                visualize_stride: stride,

                max_nodes: config.limits.max_nodes,
                max_error: config.solver.tolerance,
                max_levels: config.regrid.max_levels,

                refine_global: 1,

                domain: config.domain.clone(),
                source: vec![Source::ScalarField {
                    amplitude: *amplitude,
                    sigma: (5.35, 5.35),
                    mass: 0.0,
                }],

                solver: config.solver.clone(),
            };

            let config_path = absolute
                .join("config")
                .join(format!("initial_{:?}.toml", amplitude));

            std::fs::write(config_path, toml::to_string_pretty(&idconfig)?)?;

            let evconfig = EVConfig {
                name: "evolve".to_string(),
                output_dir: Some(
                    absolute
                        .join("evolve")
                        .join(format!("{}_{:?}", name, amplitude))
                        .as_os_str()
                        .to_string_lossy()
                        .into_owned(),
                ),
                order: 4,
                diss_order: 6,
                logging: Logging {
                    level: Logging::TRACE,
                },
                cfl: 0.1,
                dissipation: 0.5,
                max_time: config.limits.max_coord_time,
                max_proper_time: config.limits.max_proper_time,
                max_steps: config.limits.max_steps,
                max_nodes: config.limits.max_nodes,
                regrid: config.regrid.clone(),
                visualize: Some(Visualize {
                    save_interval: 0.05,
                    stride,
                }),
                gauge: GaugeCondition::Harmonic,
            };

            let config_path = absolute
                .join("config")
                .join(format!("evolve_{:?}.toml", amplitude));

            std::fs::write(config_path, toml::to_string_pretty(&evconfig)?)?;
        }

        for amplitude in subsearches.iter() {
            if initial_cache.contains_key(&float_to_string(*amplitude)) {
                log::trace!("Using cached initial data for amplitude: {:?}", amplitude);
                continue;
            }
            log::trace!(
                "Launching initial data solver for amplitude: {:?}",
                amplitude
            );

            let config_path = absolute
                .join("config")
                .join(format!("initial_{:?}.toml", amplitude));
            let status = execute_idaxi(config_path.as_ref())?;

            if status.success() {
                log::info!(
                    "Successfully generated initial data for amplitude: {:?}",
                    amplitude
                );
                initial_cache.insert(float_to_string(*amplitude), InitialStatus::Success);
            } else {
                log::error!(
                    "Failed to generate initial data for amplitude: {}",
                    amplitude
                );
                initial_cache.insert(float_to_string(*amplitude), InitialStatus::Failure);
            }
        }

        std::fs::write(
            absolute.join("initial").join("cache.toml"),
            toml::to_string_pretty(&initial_cache)?,
        )?;

        for amplitude in subsearches.iter() {
            if let Some(InitialStatus::Failure) = initial_cache.get(&float_to_string(*amplitude)) {
                return Err(anyhow!(
                    "initial data generation failed for amplitude: {}",
                    amplitude
                ));
            }
        }

        for amplitude in subsearches.iter() {
            if evolution_cache.contains_key(&float_to_string(*amplitude)) {
                log::trace!("Using cached evolution data for amplitude: {:?}", amplitude);
                continue;
            }
            log::trace!("Launching evolution for amplitude: {}", amplitude);

            let path = absolute
                .join("initial")
                .join(format!("{}_{:?}", config.name, amplitude))
                .join("initial.dat");
            let config = absolute
                .join("config")
                .join(format!("evolve_{:?}.toml", amplitude));

            let status = execute_evaxi(config.as_ref(), path.as_ref())?;

            let Some(code) = status.code() else {
                log::error!(
                    "Failed to load exit code for evolution with amplitude {}",
                    amplitude
                );
                continue;
            };

            match code {
                0 => {
                    log::info!("Amplitude: {} dispersed", amplitude);
                    evolution_cache.insert(float_to_string(*amplitude), EvolutionStatus::Disperse);
                }
                2 => {
                    log::info!("Amplitude: {} collapsed", amplitude);
                    evolution_cache.insert(float_to_string(*amplitude), EvolutionStatus::Collapse);
                }
                1 => {
                    log::error!("Error occured in evolution for amplitude {}", amplitude);

                    evolution_cache.insert(float_to_string(*amplitude), EvolutionStatus::Failure);
                }
                _ => {
                    log::error!(
                        "Unknown error occured in evolution for amplitude {}",
                        amplitude
                    );
                    evolution_cache.insert(amplitude.to_string(), EvolutionStatus::Failure);
                }
            }
        }

        std::fs::write(
            absolute.join("evolve").join("cache.toml"),
            toml::to_string_pretty(&evolution_cache)?,
        )?;

        for amplitude in subsearches.iter() {
            if let Some(EvolutionStatus::Failure) =
                evolution_cache.get(&float_to_string(*amplitude))
            {
                log::error!("Evolution Failed for amplitude: {}", amplitude);
                // return Err(anyhow!("evolution failed for amplitude: {}", amplitude));
            }
        }

        let mut dispersion = 0;
        for (i, amplitude) in subsearches.iter().enumerate() {
            match evolution_cache.get(&float_to_string(*amplitude)) {
                Some(EvolutionStatus::Disperse) => dispersion = i,
                Some(EvolutionStatus::Collapse | EvolutionStatus::Failure) => break,
                _ => return Err(anyhow!("evolution failed for amplitude: {}", amplitude)),
            }
        }

        let mut collapse = subsearches.len() - 1;
        for (i, amplitude) in subsearches.iter().enumerate().rev() {
            match evolution_cache.get(&float_to_string(*amplitude)) {
                Some(EvolutionStatus::Disperse) => break,
                Some(EvolutionStatus::Collapse | EvolutionStatus::Failure) => collapse = i,
                _ => return Err(anyhow!("evolution failed for amplitude: {}", amplitude)),
            }
        }

        if dispersion + 1 != collapse {
            return Err(anyhow!("multiple critical points detected"));
        }

        range.start = subsearches[dispersion];
        range.end = subsearches[collapse];
    }

    Ok(())
}

fn execute_idaxi(config: &OsStr) -> Result<ExitStatus> {
    let current_dir = std::env::current_dir()?;

    if let Ok(status) = Command::new(current_dir.join("idaxi")).arg(config).status() {
        return Ok(status);
    }

    let mut current_exe = std::env::current_exe()?;

    if current_exe.pop() {
        if let Ok(status) = Command::new(current_exe.join("idaxi")).arg(config).status() {
            return Ok(status);
        }
    }

    if let Ok(status) = Command::new("idaxi").arg(config).status() {
        return Ok(status);
    }

    Err(anyhow!("Failed to find idaxi executable"))
}

fn execute_evaxi(config: &OsStr, path: &OsStr) -> Result<ExitStatus> {
    let current_dir = std::env::current_dir()?;

    if let Ok(status) = Command::new(current_dir.join("evaxi"))
        .arg("--config")
        .arg(config)
        .arg(path)
        .status()
    {
        return Ok(status);
    }

    let mut current_exe = std::env::current_exe()?;

    if current_exe.pop() {
        if let Ok(status) = Command::new(current_exe.join("evaxi"))
            .arg("--config")
            .arg(config)
            .arg(path)
            .status()
        {
            return Ok(status);
        }
    }

    if let Ok(status) = Command::new("evaxi")
        .arg("--config")
        .arg(config)
        .arg(path)
        .status()
    {
        return Ok(status);
    }

    Err(anyhow!("Failed to find evaxi executable"))
}

fn main() -> ExitCode {
    match critical_search() {
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
