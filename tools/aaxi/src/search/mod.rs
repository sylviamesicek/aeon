use crate::{
    CommandExt as _, Hpc, parse_define_args, parse_invoke_arg,
    run::{self, RunHistory, Status},
};
#[cfg(feature = "mpi")]
use crate::{WORKER_STATUS_HALT, WORKER_STATUS_RUN};
use aeon_app::{
    config::{Transform, VarDefs},
    file,
};
use clap::{ArgMatches, Command};
use eyre::eyre;
#[cfg(feature = "mpi")]
use mpi::traits::*;

mod config;
mod history;

use config::Config as SearchConfig;
use history::SearchHistory;
use run::Config as RunConfig;

pub fn search(matches: &ArgMatches, hpc: Hpc) -> eyre::Result<()> {
    #[cfg(feature = "mpi")]
    println!("Root   (Rank 0): launching root in search mode");

    let vars = parse_define_args(matches)?;
    let invoke = parse_invoke_arg(matches)?;

    let config_run_file = std::env::current_dir()?.join(format!("{}.toml", invoke));
    let config_search_file = std::env::current_dir()?.join(format!("{}.search.toml", invoke));

    // Load search configuration
    let search_config = file::import_toml::<SearchConfig>(&config_search_file)?;
    let search_config = search_config.transform(&vars)?;
    // Load run configuration
    let run_config = file::import_toml::<RunConfig>(&config_run_file)?;

    // Find search directory
    let search_dir = search_config.search_dir()?;
    std::fs::create_dir_all(&search_dir)?;
    // Load history file
    let history_file = search_dir.join("history.csv");
    let mut history =
        SearchHistory::load_csv(&history_file).unwrap_or_else(|_| SearchHistory::new());

    // Setup range to search
    let mut start = search_config.start();
    let mut end = search_config.end();
    // Name of parameter
    let parameter = search_config.parameter.clone();

    // Coordinate run configuration across mpi processes
    #[cfg(feature = "mpi")]
    {
        let mut config_buffer = bincode::encode_to_vec::<RunConfig, _>(
            run_config.clone(),
            bincode::config::standard(),
        )?;
        let mut config_buffer_len: usize = config_buffer.len();
        hpc.root.broadcast_into(&mut config_buffer_len);
        assert_eq!(config_buffer_len, config_buffer.len());
        hpc.root.broadcast_into(&mut config_buffer);
        println!(
            "Root   (Rank 0): broadcasted config buffer with length {}",
            config_buffer_len
        );

        // println!("Root (Rank 0): broadcasting config File");
        let mut parameter_buffer = parameter.clone().into_bytes();
        let mut parameter_len: usize = parameter_buffer.len();
        hpc.root.broadcast_into(&mut parameter_len);
        hpc.root.broadcast_into(&mut parameter_buffer);
        println!(
            "Root   (Rank 0): broadcasted parameter buffer with length {}",
            parameter_len
        );

        let mut vars_buffer =
            bincode::encode_to_vec::<VarDefs, _>(vars.clone(), bincode::config::standard())?;
        let mut vars_buffer_len = vars_buffer.len();
        hpc.root.broadcast_into(&mut vars_buffer_len);
        hpc.root.broadcast_into(&mut vars_buffer);
        println!(
            "Root   (Rank 0): broadcasted var defs buffer with length {}",
            vars_buffer_len
        );
    }

    let info = launch_fleet(
        &run_config,
        &parameter,
        &[start, end],
        &vars,
        &history,
        hpc.clone(),
    )?;

    let start_status = info[0];
    let end_status = info[1];

    history.insert(start, start_status);
    history.insert(end, end_status);

    match (start_status, end_status) {
        (Status::Disperse, Status::Disperse) => {
            return Err(eyre!("both sides of the parameter range disperse"));
        }
        (Status::Collapse, Status::Collapse) => {
            return Err(eyre!("both sides of the parameter range collapse"));
        }
        _ => {}
    }

    // Cache history file (only if we are root to avoid conflicts).
    history.save_csv(&history_file)?;

    let mut depth = 0;
    loop {
        // Have we reached minimum tolerance
        let tolerance = (end - start).abs();
        if tolerance <= search_config.min_error {
            println!("Reached minimum critical parameter error {:.4e}", tolerance);
            break;
        }

        println!(
            "Searching range: {} to {}, diff: {:.4e}",
            start, end, tolerance
        );

        let amplitudes = (0..search_config.parallel)
            .map(|i| (i + 1) as f64 / (search_config.parallel + 1) as f64)
            .map(|w| (1.0 - w) * start + w * end)
            .collect::<Vec<_>>();

        // Execute iterations independently
        let status = launch_fleet(
            &run_config,
            &parameter,
            &amplitudes,
            &vars,
            &history,
            hpc.clone(),
        )?;

        // Insert results into history cache
        for (w, s) in amplitudes.iter().zip(status.iter()) {
            history.insert(*w, *s);
        }

        println!("Iteration Status Results: {:?}", status);

        // Imagine vector of all statuses, including start and end
        let (sindex, eindex) = double_headed_search(start_status, &status, end_status);
        match (sindex, eindex) {
            (Some(idx), None) if idx == status.len() - 1 => {}
            (None, Some(0)) => {}
            (Some(start), Some(end)) if start + 1 == end => {}
            _ => return Err(eyre!("range contains multiple critical points")),
        }

        if let Some(sidx) = sindex {
            assert!(status[sidx] == start_status);
            start = amplitudes[sidx];
        }

        if let Some(eidx) = eindex {
            assert!(status[eidx] == end_status);
            end = amplitudes[eidx]
        }

        // Cache history file.
        history.save_csv(&history_file)?;

        depth += 1;
        // Check maximum depth
        if depth >= search_config.max_depth {
            println!("Reached maximum critical search depth");
            break;
        }
    }

    #[cfg(feature = "mpi")]
    {
        let mut status: i32 = WORKER_STATUS_HALT;
        hpc.root.broadcast_into(&mut status);
    }

    println!(
        "Final search range: {} to {}, diff: {:.4e}",
        start,
        end,
        (end - start).abs()
    );

    Ok(())
}

#[cfg(feature = "mpi")]
fn launch_fleet(
    config: &RunConfig,
    parameter: &str,
    amplitudes: &[f64],
    vars: &VarDefs,
    history: &SearchHistory,
    hpc: Hpc,
) -> eyre::Result<Vec<Status>> {
    let mut execute = Vec::new();
    let mut imap = Vec::new();
    let mut stat = vec![Status::Collapse; amplitudes.len()];

    for (i, a) in amplitudes.iter().enumerate() {
        if let Some(s) = history.status(*a) {
            stat[i] = s;
        } else {
            execute.push(*a);
            imap.push(i);
        }
    }

    // Some stuff
    let mut cursor = 0;
    while cursor < execute.len() {
        let mut exec = vec![0.0; hpc.world.size() as usize];
        let mut result = vec![0i32; hpc.world.size() as usize];

        let mut num_workers: i32 = (execute.len() - cursor).min(hpc.world.size() as usize) as i32;
        assert!(num_workers >= 1);

        for i in 0..hpc.world.size() as usize {
            if cursor + i < execute.len() {
                exec[i] = execute[cursor + i];
            } else {
                exec[i] = 0.0;
            }
        }

        let mut status: i32 = WORKER_STATUS_RUN;
        hpc.root.broadcast_into(&mut status);
        println!("Root   (Rank 0): broadcasted run op code");
        hpc.root.broadcast_into(&mut num_workers);
        println!("Root   (Rank 0): broadcasted num workers {}", num_workers);

        let mut root_exec: f64 = 0.0;
        hpc.root.scatter_into_root(&exec, &mut root_exec);
        println!("Root   (Rank 0): scattered amplitudes {:?}", exec);

        // Perform iteration

        println!(
            "Root   (Rank 0): running simulation for param = {:.8}",
            root_exec
        );

        // Transform config
        let mut vars = vars.clone();
        vars.defs
            .insert(parameter.to_string(), root_exec.to_string());
        let config = config.transform(&vars)?;

        // Run simulation
        let mut history = RunHistory::empty();
        let status = run::run_simulation(&config, &mut history)?;
        let mut code = status as i32;
        // Gather results
        hpc.root.gather_into_root(&mut code, &mut result);

        for (&s, &i) in result.iter().zip(imap.iter()) {
            stat[cursor + i] = Status::from(s);
        }

        cursor += hpc.world.size() as usize;
    }

    Ok(stat)
}

#[cfg(feature = "mpi")]
pub fn search_worker(hpc: Hpc) -> eyre::Result<()> {
    let rank = hpc.world.rank();

    println!("Worker (Rank {}): launching worker in search mode", rank);

    // Recieve config over network
    let mut config_buffer_len: usize = 0;
    hpc.root.broadcast_into(&mut config_buffer_len);
    let mut config_buffer = vec![0u8; config_buffer_len];
    hpc.root.broadcast_into(&mut config_buffer);
    println!(
        "Worker (Rank {}): received config buffer with length {}",
        rank, config_buffer_len
    );
    // Convert back to config
    let (run_config, _) =
        bincode::decode_from_slice::<RunConfig, _>(&config_buffer, bincode::config::standard())?;

    // Recieve parameter name over network
    let mut parameter_buffer_len: usize = 0;
    hpc.root.broadcast_into(&mut parameter_buffer_len);
    let mut parameter_buffer = vec![0u8; parameter_buffer_len];
    hpc.root.broadcast_into(&mut parameter_buffer);
    println!(
        "Worker (Rank {}): received parameter buffer with length {}",
        rank, parameter_buffer_len
    );
    let parameter = String::from_utf8(parameter_buffer)?;

    // Recieve var definitions over networks
    let mut vars_buffer_len: usize = 0;
    hpc.root.broadcast_into(&mut vars_buffer_len);
    let mut vars_buffer = vec![0u8; vars_buffer_len];
    hpc.root.broadcast_into(&mut vars_buffer);
    println!(
        "Worker (Rank {}): received var defs buffer with length {}",
        rank, vars_buffer_len
    );
    // Convert back to config
    let (vars, _) =
        bincode::decode_from_slice::<VarDefs, _>(&vars_buffer, bincode::config::standard())?;

    loop {
        let mut code: i32 = WORKER_STATUS_RUN;
        hpc.root.broadcast_into(&mut code);
        println!(
            "Worker (Rank {}): work loop received status update {}",
            rank, code
        );

        if code == WORKER_STATUS_RUN {
        } else if code == WORKER_STATUS_HALT {
            println!("Worker (Rank {rank}): halting");
            break;
        } else {
            return Err(eyre!(
                "worker on rank {} received unknown status code {}",
                rank,
                code
            ));
        }

        let mut num_workers: i32 = 0;
        hpc.root.broadcast_into(&mut num_workers);
        println!(
            "Worker (Rank {}): received num workers {}",
            rank, num_workers
        );

        let mut amplitude: f64 = 0.0;
        hpc.root.scatter_into(&mut amplitude);
        println!(
            "Worker (Rank {}): received scattered amplitude {}",
            rank, amplitude
        );

        let mut code = Status::Collapse as i32;

        if hpc.world.rank() < num_workers {
            // Perform iteration
            println!(
                "Worker (Rank {}): running simulation for param = {}",
                rank, amplitude
            );

            let mut vars = vars.clone();
            vars.defs
                .insert(parameter.to_string(), amplitude.to_string());

            let config = run_config.transform(&vars)?;

            let mut history = RunHistory::empty();
            let simulation = run::run_simulation(&config, &mut history)?;
            code = simulation as i32;
        } else {
            // Perform iteration
            println!("Worker (Rank {}): skipping execution", rank);
        }

        hpc.root.gather_into(&code);
    }

    Ok(())
}

#[cfg(not(feature = "mpi"))]
fn launch_fleet(
    config: &RunConfig,
    parameter: &str,
    amplitudes: &[f64],
    vars: &VarDefs,
    history: &SearchHistory,
    _hpc: Hpc,
) -> eyre::Result<Vec<Status>> {
    let mut status = vec![Status::Collapse; amplitudes.len()];

    for (i, &amp) in amplitudes.iter().enumerate() {
        if let Some(s) = history.status(amp) {
            status[i] = s;
            continue;
        }

        let mut vars = vars.clone();
        vars.defs.insert(parameter.to_string(), amp.to_string());

        let config = config.transform(&vars)?;

        let mut history = RunHistory::empty();
        status[i] = run::run_simulation(&config, &mut history)?;
    }

    Ok(status)
}

// /// Run a single iteration of a critical search
// fn search_iteration(
//     vars: &VarDefs,
//     config: &run::Config,
//     parameter: &str,
//     amplitude: f64,
//     history: &SearchHistory,
// ) -> eyre::Result<Status> {
//     if let Some(s) = history.status(amplitude) {
//         return Ok(s);
//     }

//     let mut vars = vars.clone();
//     // Set the parameter variable to the given amplitude
//     vars.defs
//         .insert(parameter.to_string(), amplitude.to_string());
//     // Transform config appropriately
//     let config = config.transform(&vars)?;

//     let mut history = RunHistory::empty();
//     let sim = run::run_simulation(&config, &mut history)?;

//     Ok(sim)
// }

pub trait CommandExt {
    fn search_cmd(self) -> Self;
}

impl CommandExt for Command {
    fn search_cmd(self) -> Self {
        self.subcommand(
            Command::new("search")
                .about("Performs critical search across many MPI nodes")
                .define_args()
                .invoke_arg(),
        )
    }
}

pub fn parse_search_cmd(matches: &ArgMatches) -> Option<&ArgMatches> {
    matches.subcommand_matches("search")
}

/// Returns the last index in values that matches start and the first index in values that matches end.
fn double_headed_search<T: PartialEq>(
    start: T,
    values: &[T],
    end: T,
) -> (Option<usize>, Option<usize>) {
    assert!(start != end);

    if values.len() == 0 {
        return (None, None);
    }

    if values.len() == 1 {
        if start == values[0] {
            return (Some(0), None);
        }

        if end == values[0] {
            return (None, Some(0));
        }

        panic!("Invalid search");
    }

    let sindex = if values[0] != start {
        None
    } else {
        let mut sindex = 0;
        while sindex < values.len() - 1 && values[sindex + 1] == start {
            sindex += 1;
        }
        Some(sindex)
    };

    let eindex = if values[values.len() - 1] != end {
        None
    } else {
        let mut eindex = values.len() - 1;
        while eindex > 0 && values[eindex - 1] == end {
            eindex -= 1;
        }
        Some(eindex)
    };

    (sindex, eindex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_headed() {
        assert_eq!(double_headed_search(false, &[true], true), (None, Some(0)));
        assert_eq!(double_headed_search(false, &[false], true), (Some(0), None));
        assert_eq!(
            double_headed_search(false, &[false, false, true, true], true),
            (Some(1), Some(2))
        );
        assert_eq!(
            double_headed_search(false, &[false, false, true, false, true, true], true),
            (Some(1), Some(4))
        );
    }
}
