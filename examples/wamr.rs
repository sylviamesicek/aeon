//! An example of using wavelet adaptive mesh refinement to
//! compress a function and generate appropriate grids.

use aeon::prelude::*;
use std::f64::consts::PI;

#[derive(Clone)]
pub struct SeedConditions;

impl SystemBoundaryConds<2> for SeedConditions {
    fn kind(&self, _channel: usize, face: Face<2>) -> BoundaryKind {
        if face.side {
            return BoundaryKind::Radiative;
        }

        [BoundaryKind::AntiSymmetric, BoundaryKind::Symmetric][face.axis]
    }

    fn radiative(&self, _channel: usize, _position: [f64; 2]) -> RadiativeParams {
        RadiativeParams::lightlike(0.0)
    }
}

#[derive(Clone)]
struct SeedProjection;

impl Projection<2> for SeedProjection {
    fn project(&self, [rho, z]: [f64; 2]) -> f64 {
        rho * (-(rho * rho + z * z)).exp()
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    std::fs::create_dir_all("output/wamr")?;

    let domain = HyperBox {
        origin: [0., 0.],
        size: [2. * PI, 2. * PI],
    };

    log::info!("Building Base Mesh.");

    // Create mesh
    let mut mesh = Mesh::new(
        domain,
        4,
        2,
        FaceArray::from_fn(|face| match face.side {
            false => BoundaryClass::Ghost,
            true => BoundaryClass::OneSided,
        }),
    );
    mesh.refine_global();

    // Store system from previous iteration.
    let mut system_prev = Image::new(1, mesh.num_nodes());
    mesh.project(4, SeedProjection, system_prev.channel_mut(0));

    let mut errors = Vec::new();

    for i in 0..10 {
        log::info!(
            "Computing {}th iteration with {} nodes.",
            i,
            mesh.num_nodes()
        );

        let mut system = Image::new(1, mesh.num_nodes());
        mesh.project(4, SeedProjection, system.channel_mut(0));
        mesh.fill_boundary(4, SeedConditions, system.as_mut());

        mesh.flag_wavelets(4, 1e-13, 1e-9, system.as_ref());
        mesh.balance_flags();

        // Output

        let mut flag_debug = vec![0; mesh.num_nodes()];
        let mut block_debug = vec![0; mesh.num_nodes()];
        let mut cell_debug = vec![0; mesh.num_nodes()];

        mesh.flags_debug(&mut flag_debug);
        mesh.block_debug(&mut block_debug);
        mesh.cell_debug(&mut cell_debug);

        let diff = system
            .channel(0)
            .iter()
            .zip(system_prev.channel(0).iter())
            .map(|(i, j)| i - j)
            .collect::<Vec<_>>();

        let norm = mesh.l2_norm_system(ImageRef::from(diff.as_slice()));

        if norm.abs() >= 1e-20 {
            errors.push((i, norm));
        }

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_field("Seed", system.channel(0));
        checkpoint.save_field("SeedInterpolated", system_prev.channel(0));
        checkpoint.save_field("SeedDiff", &diff);
        checkpoint.save_int_field("Flags", &flag_debug);
        checkpoint.save_int_field("Blocks", &block_debug);
        checkpoint.save_int_field("Cell", &cell_debug);

        checkpoint.export_vtu(
            format!("output/wamr/wamr{i}.vtu"),
            ExportVtuConfig {
                title: "WAMR".to_string(),
                ghost: false,
                stride: ExportStride::PerVertex,
            },
        )?;

        mesh.regrid();

        // Prolong data from previous system.
        system_prev = Image::new(1, mesh.num_nodes());
        mesh.transfer_system(4, system.as_ref(), system_prev.as_mut());
    }

    for i in 0..errors.len() - 1 {
        let original = errors[i];
        let dest = errors[i + 1];

        log::info!(
            "Ratio between {} and {} iteration is {}",
            original.0,
            dest.0,
            original.1 / dest.1
        );
    }

    Ok(())
}
