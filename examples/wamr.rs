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

    let mut mesh = Mesh::new(domain, 4, 2);

    for i in 0..10 {
        log::info!(
            "Computing {}th iteration with {} nodes.",
            i,
            mesh.num_nodes()
        );

        let mut system = SystemVec::with_length(mesh.num_nodes());
        mesh.project(Order::<4>, Quadrant, SeedProjection, system.as_mut_slice());
        mesh.fill_boundary(Order::<4>, Quadrant, SeedConditions, system.as_mut_slice());

        mesh.wavelet(1e-9, Quadrant, system.as_slice());
        mesh.balance_flags();

        // if mesh.num_cells() > 96 {
        //     let region = Region::new([Side::Right, Side::Right]);
        //     let neighbor = mesh.tree().neighbor_in_region(96, region);

        //     let n1 = mesh.tree().neighbor_after_refinement(
        //         96,
        //         AxisMask::pack([true, true]),
        //         Face::positive(0),
        //     );

        //     let n1o = mesh.tree().neighbor(96, Face::positive(0));

        //     dbg!(n1, n1o);

        //     let n2 = mesh.tree().neighbor_after_refinement(
        //         n1,
        //         AxisMask::pack([false, true]),
        //         Face::positive(1),
        //     );

        //     dbg!(n2);

        //     for face in region.adjacent_faces() {
        //         dbg!(face);
        //     }

        //     dbg!(neighbor);
        // }

        let mut flag_debug = vec![0; mesh.num_nodes()];
        let mut block_debug = vec![0; mesh.num_nodes()];
        let mut cell_debug = vec![0; mesh.num_nodes()];

        mesh.flags_debug(&mut flag_debug);
        mesh.block_debug(&mut block_debug);
        mesh.cell_debug(&mut cell_debug);

        let mut systems = SystemCheckpoint::default();
        systems.save_system(system.as_slice());
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

        // mesh.refine_just_blocks();

        let mut debug = String::new();
        mesh.write_debug(&mut debug);

        std::fs::write("output/mesh.txt", debug).unwrap();

        // for cell in 0..mesh.num_cells() {
        //     println!("Cell {:?} Flag {}", cell, mesh.refine_flags()[cell])
        // }

        mesh.refine();
    }

    Ok(())
}
