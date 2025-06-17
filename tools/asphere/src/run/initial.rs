use crate::{
    misc,
    run::config::Config,
    system::{Fields, InitialData, generate_initial_scalar_field, solve_constraints},
};
use aeon::prelude::*;
use datasize::DataSize as _;
use eyre::eyre;
use indicatif::{HumanBytes, HumanDuration, MultiProgress, ProgressBar};
use std::time::Instant;

/// Solve for initial conditions and adaptively refine mesh
pub fn initial_data(config: &Config) -> eyre::Result<(Mesh<1>, SystemVec<Fields>)> {
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
        level_pb.set_length(mesh.num_levels() as u64);

        // Set initial data for scalar field.
        let scalar_field = generate_initial_scalar_field(&mut mesh, &source.profile);

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
                    "{}_levels_{}.vtu",
                    config.name,
                    mesh.num_levels()
                )),
                ExportVtuConfig {
                    title: "Massless Scalar Field Initial Data".to_string(),
                    ghost: false,
                    stride: config.visualize.stride,
                },
            )?;
        }

        if mesh.num_nodes() >= config.limits.max_nodes {
            log::error!(
                "Failed to solve initial data, level: {}, nodes: {}",
                mesh.num_levels(),
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
                stride: config.visualize.stride,
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
                stride: config.visualize.stride,
            },
        )?;
    }

    Ok((mesh, system))
}
