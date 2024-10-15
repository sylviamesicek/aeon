use aeon::{fd::Gaussian, prelude::*};

const ORDER: Order<4> = Order::<4>;
// const REGRID_SKIP: usize = 10;

const LOWER: f64 = 1e-10;
const UPPER: f64 = 1e-6;

/// The quadrant domain the function is being projected on.
#[derive(Clone)]
pub struct Quadrant;

impl Boundary<2> for Quadrant {
    fn kind(&self, _face: Face<2>) -> BoundaryKind {
        BoundaryKind::Radiative
    }
}

#[derive(Clone)]
pub struct WaveConditions;

impl Conditions<2> for WaveConditions {
    type System = Scalar;

    fn radiative(&self, _field: Self::System, _position: [f64; 2]) -> f64 {
        0.0
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    std::fs::create_dir_all("output/transfer")?;

    log::info!("Intializing Mesh.");

    // Generate initial mesh
    let mut mesh = Mesh::new(Rectangle::from_aabb([-10., -10.], [10., 10.]), 4, 3);
    // Allocate space for system
    let mut system = SystemVec::new();

    // Comparison
    let mut transfered = SystemVec::new();
    let mut error = SystemVec::<Scalar>::new();

    log::info!("Performing Initial Adaptive Mesh Refinement.");

    // Perform initial adaptive refinement.
    for i in 0..15 {
        log::info!("Iteration: {i}");

        let profile = Gaussian {
            amplitude: 1.0,
            sigma: 1.0,
            center: [0.5, 0.],
        };

        system.resize(mesh.num_nodes());

        mesh.project(ORDER, Quadrant, profile, system.as_mut_slice());
        mesh.fill_boundary(ORDER, Quadrant, WaveConditions, system.as_mut_slice());

        mesh.flag_wavelets(LOWER, UPPER, Quadrant, system.as_slice());
        mesh.set_regrid_level_limit(10);

        mesh.balance_flags();

        // Save data to file.

        let mut flags = vec![0; mesh.num_nodes()];
        mesh.flags_debug(&mut flags);

        let mut systems = SystemCheckpoint::default();
        systems.save_field("Wave", system.contigious());
        systems.save_int_field("Flags", &flags);

        if error.len() == mesh.num_nodes() {
            for i in 0..mesh.num_nodes() {
                error.contigious_mut()[i] = system.contigious()[i] - transfered.contigious()[i]
            }

            systems.save_field("Transfered", transfered.contigious());
            systems.save_field("Error", error.contigious());
        }

        let path = format!("output/transfer/iteration{i}.vtu");
        mesh.export_vtu(
            path.as_str(),
            ExportVtuConfig {
                title: "Initial Wave Mesh".to_string(),
                ghost: false,
                systems,
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

            mesh.transfer_system(
                ORDER,
                Quadrant,
                system.as_slice(),
                transfered.as_mut_slice(),
            );

            continue;
        } else {
            log::info!("Regridded within range in {} iterations.", i + 1);
            break;
        }
    }

    Ok(())
}
