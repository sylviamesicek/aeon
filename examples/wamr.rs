//! An example of using wavelet adaptive mesh refinement to
//! compress a function and generate appropriate grids.

use aeon::prelude::*;
use std::f64::consts::PI;

#[derive(Clone)]
pub struct SeedConditions;

impl SystemBoundaryConds<2> for SeedConditions {
    type System = Scalar;

    fn parity(&self, _field: (), face: Face<2>) -> bool {
        [false, true][face.axis]
    }

    fn radiative(&self, _field: (), _position: [f64; 2]) -> RadiativeParams {
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

    let domain = Rectangle {
        origin: [0., 0.],
        size: [2. * PI, 2. * PI],
    };

    log::info!("Building Base Mesh.");

    // Create mesh
    let mut mesh = Mesh::new(domain, 4, 2);
    mesh.set_face_boundary(Face::negative(0), BoundaryKind::Parity);
    mesh.set_face_boundary(Face::negative(1), BoundaryKind::Parity);
    mesh.set_face_boundary(Face::positive(0), BoundaryKind::Radiative);
    mesh.set_face_boundary(Face::positive(1), BoundaryKind::Radiative);

    // Store system from previous iteration.
    let mut system_prev = SystemVec::with_length(mesh.num_nodes(), Scalar);
    mesh.project(4, SeedProjection, system_prev.field_mut(()));

    let mut errors = Vec::new();

    for i in 0..10 {
        log::info!(
            "Computing {}th iteration with {} nodes.",
            i,
            mesh.num_nodes()
        );

        let mut system = SystemVec::with_length(mesh.num_nodes(), Scalar);
        mesh.project(4, SeedProjection, system.field_mut(()));
        mesh.fill_boundary(Order::<4>, SeedConditions, system.as_mut_slice());

        mesh.flag_wavelets(4, 1e-13, 1e-9, system.as_slice());
        mesh.balance_flags();

        // Output

        let mut flag_debug = vec![0; mesh.num_nodes()];
        let mut block_debug = vec![0; mesh.num_nodes()];
        let mut cell_debug = vec![0; mesh.num_nodes()];

        mesh.flags_debug(&mut flag_debug);
        mesh.block_debug(&mut block_debug);
        mesh.cell_debug(&mut cell_debug);

        let diff = system
            .field(())
            .iter()
            .zip(system_prev.field(()).iter())
            .map(|(i, j)| i - j)
            .collect::<Vec<_>>();

        let norm = mesh.l2_norm(SystemSlice::from_scalar(diff.as_slice()));

        if norm.abs() >= 1e-20 {
            errors.push((i, norm));
        }

        let mut systems = SystemCheckpoint::default();
        systems.save_field("Seed", system.field(()));
        systems.save_field("SeedInterpolated", system_prev.field(()));
        systems.save_field("SeedDiff", &diff);
        systems.save_int_field("Flags", &flag_debug);
        systems.save_int_field("Blocks", &block_debug);
        systems.save_int_field("Cell", &cell_debug);

        mesh.export_vtu(
            format!("output/wamr/wamr{i}.vtu"),
            &systems,
            ExportVtuConfig {
                title: "WAMR".to_string(),
                ghost: false,
                stride: 1,
            },
        )?;

        mesh.regrid();

        // Prolong data from previous system.
        system_prev = SystemVec::with_length(mesh.num_nodes(), Scalar);
        mesh.transfer_system(Order::<4>, system.as_slice(), system_prev.as_mut_slice());
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
