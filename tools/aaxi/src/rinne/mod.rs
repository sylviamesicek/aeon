//! This crate contains general configuration and paramter data types used by critgen, idgen, and evgen.
//! These types are shared across crates, and thus moved here to prevent redundent definition.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use crate::{
    config::{Config, Source},
    misc,
};
use aeon::{prelude::*, solver::SolverCallback};
use eyre::eyre;
use indicatif::{HumanDuration, MultiProgress, ProgressBar};

mod eqs;
mod garfinkle;
mod systems;

pub use systems::*;

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

        let Some(visualize_interval) = self.config.visualize.save_relax_interval else {
            return;
        };

        if iteration % visualize_interval != 0 {
            return;
        }

        let i = iteration / visualize_interval;

        let mut checkpoint = SystemCheckpoint::default();
        checkpoint.save_field("Solution", input.into_scalar());
        checkpoint.save_field("Derivative", output.into_scalar());

        mesh.export_vtu(
            self.output.join("initial").join(format!(
                "{}_level_{}_iter_{}.vtu",
                self.config.name,
                mesh.max_level(),
                i
            )),
            &checkpoint,
            ExportVtuConfig {
                title: self.config.name.to_string(),
                ghost: false,
                stride: self.config.visualize.stride,
            },
        )
        .unwrap()
    }
}

/// Solve rinne's initial data problem using Garfinkles variables.
pub fn initial_data(config: &Config, output: &Path) -> eyre::Result<(Mesh<2>, SystemVec<Fields>)> {
    println!("Relaxing initial data");
    // Save initial time
    let start = Instant::now();

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
                    Some(mass.as_f64())
                } else {
                    None
                }
            })
            .collect(),
    };

    // ************************************
    // Visualization

    // Path for all visualization data.
    if config.visualize.save_relax_levels
        || config.visualize.save_relax_interval.is_some()
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
                        output,
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
                        output,
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
                        output,
                    },
                    &config.source,
                    system.as_mut_slice(),
                )?;
            }
            _ => return Err(eyre!("Invalid initial data type and order")),
        };

        pb.finish_with_message(format!(
            "relaxed in {} steps, {} nodes",
            pb.position(),
            mesh.num_nodes()
        ));
        step_count += pb.position();

        if config.visualize.save_relax_levels {
            let mut checkpoint = SystemCheckpoint::default();
            checkpoint.save_system_ser(system.as_slice());

            mesh.export_vtu(
                output.join(format!("{}_level{}.vtu", &config.name, mesh.max_level())),
                &checkpoint,
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
        let mut checkpoint = SystemCheckpoint::default();
        checkpoint.save_system_ser(system.as_slice());

        mesh.export_vtu(
            output.join("initial").join(format!("{}.vtu", config.name)),
            &checkpoint,
            ExportVtuConfig {
                title: config.name.clone(),
                ghost: false,
                stride: config.visualize.stride,
            },
        )?;
    }

    m.clear().unwrap();

    println!(
        "Finished relaxation in {}, {} Steps",
        HumanDuration(start.elapsed()),
        step_count
    );
    println!("Mesh Info...");
    println!("- Nodes: {}", mesh.num_nodes());
    println!("- Active Cells: {}", mesh.num_active_cells());

    Ok((mesh, system))
}
