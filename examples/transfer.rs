use aeon::{mesh::Gaussian, prelude::*};

const ORDER: usize = 4;
const LOWER: f64 = 1e-10;
const UPPER: f64 = 1e-6;

#[derive(Clone)]
pub struct WaveConditions;

impl SystemBoundaryConds<2> for WaveConditions {
    fn kind(&self, _channel: usize, _face: Face<2>) -> BoundaryKind {
        BoundaryKind::Radiative
    }

    fn radiative(&self, _channel: usize, _position: [f64; 2]) -> RadiativeParams {
        RadiativeParams::lightlike(0.0)
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    std::fs::create_dir_all("output/transfer")?;

    log::info!("Intializing Mesh.");

    // Generate initial mesh
    let mut mesh = Mesh::new(
        HyperBox::from_aabb([-10., -10.], [10., 10.]),
        4,
        2,
        FaceArray::splat(BoundaryClass::OneSided),
    );
    mesh.refine_global();

    // Allocate space for system
    let mut system = Image::new(1, 0);

    // Comparison
    let mut transfered = Image::new(1, 0);
    let mut error = Image::new(1, 0);

    log::info!("Performing Initial Adaptive Mesh Refinement.");

    // Perform initial adaptive refinement.
    for i in 0..15 {
        log::info!("Iteration: {i}");

        let profile = Gaussian {
            amplitude: 1.0,
            sigma: [1.0, 1.0],
            center: [0.5, 0.],
        };

        system.resize(mesh.num_nodes());

        mesh.project(4, profile, system.channel_mut(0));
        mesh.fill_boundary(ORDER, WaveConditions, system.as_mut());

        mesh.flag_wavelets(4, LOWER, UPPER, system.as_ref());

        mesh.limit_level_range_flags(1, 10);
        mesh.balance_flags();

        // Save data to file.

        let mut flags = vec![0; mesh.num_nodes()];
        mesh.flags_debug(&mut flags);

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_field("Wave", system.storage());
        checkpoint.save_int_field("Flags", &flags);

        if error.num_nodes() == mesh.num_nodes() {
            for i in 0..mesh.num_nodes() {
                error.storage_mut()[i] = system.storage()[i] - transfered.storage()[i]
            }

            checkpoint.save_field("Transfered", transfered.storage());
            checkpoint.save_field("Error", error.storage());
        }

        let path = format!("output/transfer/iteration{i}.vtu");
        checkpoint.export_vtu(
            path.as_str(),
            ExportVtuConfig {
                title: "Initial Wave Mesh".to_string(),
                ghost: false,
                stride: ExportStride::PerVertex,
            },
        )?;

        if i == 14 {
            log::info!("Failed to regrid completely within 15 iterations.");
            break;
        }

        if mesh.requires_regridding() {
            let refine = mesh.num_refine_cells();
            let coarsen = mesh.num_coarsen_cells();

            log::info!("Refining {refine} cells, coarsening {coarsen} cells.");
            mesh.regrid();

            transfered.resize(mesh.num_nodes());
            error.resize(mesh.num_nodes());

            mesh.transfer_system(ORDER, system.as_ref(), transfered.as_mut());

            continue;
        } else {
            log::info!("Regridded within range in {} iterations.", i + 1);
            break;
        }
    }

    Ok(())
}
