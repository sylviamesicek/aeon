use aeon_app::config::{VarDef, VarDefs};
use clap::{Arg, ArgAction, ArgMatches, Command};
use eyre::eyre;
#[cfg(feature = "mpi")]
use mpi::{
    topology::{Process, SimpleCommunicator},
    traits::*,
};
use std::io::Write as _;
use std::marker::PhantomData;

mod eqs;
mod horizon;
mod run;
mod schwarzschild;
mod search;
mod systems;

use run::CommandExt as _;
use schwarzschild::CommandExt as _;
use search::CommandExt as _;

#[derive(Clone)]
/// Struct for managing High Performance Computing datastructures (mainly responsible for coordinating several different MPI processes)
struct Hpc<'a> {
    #[cfg(feature = "mpi")]
    world: &'a mpi::topology::SimpleCommunicator,
    #[cfg(feature = "mpi")]
    root: Process<'a>,

    _marker: PhantomData<&'a ()>,
}

impl<'a> Hpc<'a> {
    #[cfg(feature = "mpi")]
    fn new(world: &'a SimpleCommunicator) -> Self {
        Self {
            world: world,
            root: world.process_at_rank(ROOT_RANK),
            _marker: PhantomData,
        }
    }
}

impl Hpc<'static> {
    #[cfg(not(feature = "mpi"))]
    fn empty() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[cfg(feature = "mpi")]
const WORKER_OP_NONE: i32 = 0;
#[cfg(feature = "mpi")]
const WORKER_OP_SEARCH: i32 = 1;
#[cfg(feature = "mpi")]
const WORKER_STATUS_RUN: i32 = 0;
#[cfg(feature = "mpi")]
const WORKER_STATUS_HALT: i32 = 1;
#[cfg(feature = "mpi")]
const ROOT_RANK: i32 = 0;

fn main() -> eyre::Result<()> {
    // Set up MPI context and such.
    #[cfg(feature = "mpi")]
    let (universe, threading) = mpi::initialize_with_threading(mpi::Threading::Serialized).unwrap();
    #[cfg(feature = "mpi")]
    eyre::ensure!(
        matches!(
            threading,
            mpi::Threading::Serialized | mpi::Threading::Multiple
        ),
        "MPI implementation doesn't support multithreaded processes"
    );

    #[cfg(feature = "mpi")]
    let world = universe.world();
    #[cfg(feature = "mpi")]
    let hpc = Hpc::new(&world);
    #[cfg(not(feature = "mpi"))]
    let hpc = Hpc::empty();

    // Set up nice error handing.
    color_eyre::install()?;

    #[cfg(not(feature = "mpi"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format(move |buf, record| writeln!(buf, "[{}]: {}", record.level(), record.args()))
            .init();
    }

    #[cfg(feature = "mpi")]
    {
        let rank = hpc.world.rank();
        if rank != ROOT_RANK {
            env_logger::builder()
                .filter_level(log::LevelFilter::Info)
                .format(move |buf, record| {
                    writeln!(
                        buf,
                        "[{}; Worker {}]: {}",
                        record.level(),
                        rank,
                        record.args()
                    )
                })
                .init();
        } else {
            env_logger::builder()
                .filter_level(log::LevelFilter::Info)
                .format(move |buf, record| {
                    writeln!(
                        buf,
                        "[{}; Root   {}]: {}",
                        record.level(),
                        rank,
                        record.args()
                    )
                })
                .init();
        }
    }

    // Run as a worker if requested.
    #[cfg(feature = "mpi")]
    if hpc.world.rank() != ROOT_RANK {
        let mut opcode = 0;
        hpc.root.broadcast_into(&mut opcode);

        match opcode {
            WORKER_OP_NONE => {
                log::info!("Performing noop");
                return Ok(());
            }
            WORKER_OP_SEARCH => {
                return search::search_worker(hpc);
            }
            _ => {
                return Err(eyre!(
                    "invalid worker op id received on rank: {}",
                    hpc.world.rank()
                ));
            }
        }
    }

    // Specify cli argument parsing.
    let command = Command::new("aaxi")
        .about("A program for running axisymmetric simulations using numerical relativity")
        .author("Sylvia Mesicek, sylvia.mesicek@gmail.com")
        .version("0.3.0")
        .subcommand_negates_reqs(true)
        .schwarzschild_cmd()
        .run_cmd()
        .search_cmd();
    // Check argument matches
    let matches = command.get_matches();

    // Run schwarzschild subcommand
    if let Some(matches) = schwarzschild::parse_schwarzschild_cmd(&matches) {
        log::info!("Running Schwarzschil subcommand");
        return schwarzschild::schwarzschild(matches);
    }

    // Run search subcommand
    if let Some(matches) = search::parse_search_cmd(&matches) {
        log::info!("Running Search subcommand");
        #[cfg(feature = "mpi")]
        {
            log::info!("Broadcasting search subcommand.");
            let mut worker = WORKER_OP_SEARCH;
            hpc.root.broadcast_into(&mut worker);
        }

        return search::search(matches, hpc);
    }

    // Run default subcommand
    if let Some(matches) = run::parse_run_cmd(&matches) {
        log::info!("Running run subcommand");
        return run::run(matches);
    }

    // Run default subcommand
    log::info!("No subcommand provided");
    Ok(())
}

trait CommandExt {
    /// Implements default define args
    fn define_args(self) -> Self;
    /// Implements 0th positional argument for executing a specific subconfig
    fn invoke_arg(self) -> Self;
}

impl CommandExt for Command {
    fn define_args(self) -> Self {
        self.arg(
            Arg::new("define")
                .long("define")
                .short('D')
                .help("Define variable to be referenced in config files via ${} syntax")
                .required(false)
                .action(ArgAction::Append),
        )
    }

    fn invoke_arg(self) -> Self {
        self.arg(Arg::new("invoke").required(true))
    }
}

/// Loads a series of variable definitions from a set of argument matches
fn parse_define_args(matches: &ArgMatches) -> eyre::Result<VarDefs> {
    // Collection of cli invokation variable definitions.
    let mut vars = VarDefs::new();
    if let Some(defines) = matches.get_many::<String>("define") {
        for def in defines {
            vars.insert(VarDef::parse(def)?);
        }
    }
    Ok(vars)
}

/// Loads the invokation argument from a set of argument matches
fn parse_invoke_arg(matches: &ArgMatches) -> eyre::Result<String> {
    let Some(invoke) = matches.get_one::<String>("invoke") else {
        return Err(eyre!(
            "failed to find 0th positional argument in cli invokation"
        ));
    };

    Ok(invoke.clone())
}
