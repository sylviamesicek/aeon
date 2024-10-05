use std::f64::consts::PI;

use aeon::prelude::*;

#[derive(Clone)]
pub struct Quadrant;

impl Boundary<2> for Quadrant {
    fn kind(&self, face: Face<2>) -> BoundaryKind {
        if face.side == false {
            BoundaryKind::Parity
        } else {
            BoundaryKind::Radiative
        }
    }
}

#[derive(Clone)]
pub struct SeedConditions;

impl Conditions<2> for SeedConditions {
    type System = Scalar;

    fn parity(&self, _field: Self::System, face: Face<2>) -> bool {
        [false, true][face.axis]
    }

    fn radiative(&self, _field: Self::System, _position: [f64; 2]) -> f64 {
        0.0
    }
}

impl Conditions<2> for Quadrant {
    type System = Scalar;

    fn parity(&self, _field: Self::System, _face: Face<2>) -> bool {
        false
    }
}

#[derive(Clone)]
struct SeedProjection;

impl Projection<2> for SeedProjection {
    type Output = Scalar;

    fn project(&self, [rho, z]: [f64; 2]) -> SystemValue<Self::Output> {
        SystemValue::new([rho * (-(rho * rho + z * z)).exp()])
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    std::fs::create_dir_all("output/wavelet")?;

    let domain = Rectangle {
        origin: [0., 0.],
        size: [2. * PI, 2. * PI],
    };

    log::info!("Building Base Mesh.");

    // Create mesh
    let mut mesh = Mesh::new(domain, 4, 2);

    // Store system from previous iteration.
    let mut system_prev = SystemVec::with_length(mesh.num_nodes());
    mesh.project(
        Order::<4>,
        Quadrant,
        SeedProjection,
        system_prev.as_mut_slice(),
    );

    let mut errors = Vec::new();

    for i in 0..10 {
        log::info!(
            "Computing {}th iteration with {} nodes.",
            i,
            mesh.num_nodes()
        );

        let mut system = SystemVec::with_length(mesh.num_nodes());
        mesh.project(Order::<4>, Quadrant, SeedProjection, system.as_mut_slice());
        mesh.fill_boundary(Order::<4>, Quadrant, SeedConditions, system.as_mut_slice());

        mesh.flag_wavelets(1e-13, 1e-9, Quadrant, system.as_slice());
        mesh.balance_flags();

        // Output

        let mut flag_debug = vec![0; mesh.num_nodes()];
        let mut block_debug = vec![0; mesh.num_nodes()];
        let mut cell_debug = vec![0; mesh.num_nodes()];

        mesh.flags_debug(&mut flag_debug);
        mesh.block_debug(&mut block_debug);
        mesh.cell_debug(&mut cell_debug);

        let diff = system
            .field(Scalar)
            .iter()
            .zip(system_prev.field(Scalar).iter())
            .map(|(i, j)| i - j)
            .collect::<Vec<_>>();

        let norm = mesh.norm(diff.as_slice().into());

        if norm.abs() >= 1e-20 {
            errors.push((i, norm));
        }

        let mut systems = SystemCheckpoint::default();
        systems.save_field("Seed", system.field(Scalar));
        systems.save_field("SeedInterpolated", system_prev.field(Scalar));
        systems.save_field("SeedDiff", &diff);
        systems.save_int_field("Flags", &flag_debug);
        systems.save_int_field("Blocks", &block_debug);
        systems.save_int_field("Cell", &cell_debug);

        mesh.export_vtk(
            format!("output/wavelet/wamr{i}.vtu"),
            ExportVtkConfig {
                title: "WAMR".to_string(),
                ghost: false,
                systems,
            },
        )?;

        mesh.regrid();

        // Prolong data from previous system.
        system_prev = SystemVec::with_length(mesh.num_nodes());
        mesh.transfer_system(
            Order::<4>,
            Quadrant,
            system.as_slice(),
            system_prev.as_mut_slice(),
        );
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
