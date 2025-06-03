use std::{convert::Infallible, path::PathBuf};

use aeon::{mesh::Gaussian, prelude::*};
use clap::{ArgMatches, Command, arg, value_parser};

use crate::{
    eqs, misc,
    systems::{Constraint, Field, FieldConditions, Fields, Gauge, Metric, ScalarField},
};

struct SchwarzschildData {
    mass: f64,
}

impl Function<2> for SchwarzschildData {
    type Input = Empty;
    type Output = Fields;
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        _: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Self::Error> {
        let mass = self.mass;

        assert!(output.len() == engine.num_nodes());

        output.field_mut(Field::Metric(Metric::Grz)).fill(0.0);
        output.field_mut(Field::Metric(Metric::S)).fill(0.0);
        output.field_mut(Field::Metric(Metric::Krr)).fill(0.0);
        output.field_mut(Field::Metric(Metric::Krz)).fill(0.0);
        output.field_mut(Field::Metric(Metric::Kzz)).fill(0.0);
        output.field_mut(Field::Metric(Metric::Y)).fill(0.0);
        output.field_mut(Field::Gauge(Gauge::Shiftr)).fill(0.0);
        output.field_mut(Field::Gauge(Gauge::Shiftz)).fill(0.0);
        output
            .field_mut(Field::Constraint(Constraint::Theta))
            .fill(0.0);
        output
            .field_mut(Field::Constraint(Constraint::Zr))
            .fill(0.0);
        output
            .field_mut(Field::Constraint(Constraint::Zz))
            .fill(0.0);

        for i in 0..output.system().num_scalar_fields() {
            output
                .field_mut(Field::ScalarField(ScalarField::Phi, i))
                .fill(0.0);
            output
                .field_mut(Field::ScalarField(ScalarField::Pi, i))
                .fill(0.0);
        }

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            let [rho, z] = engine.position(vertex);

            let mut radius = (rho * rho + z * z).sqrt();

            if radius <= mass / 10.0 {
                radius = mass / 10.0
            }

            let lapse = (mass - 2.0 * radius) / (mass + 2.0 * radius);
            let conformal = (1.0 + mass / (2.0 * radius)).powi(4);

            if conformal >= 1e4 {
                println!("What");
            }

            output.field_mut(Field::Metric(Metric::Grr))[index] = conformal;
            output.field_mut(Field::Metric(Metric::Gzz))[index] = conformal;
            output.field_mut(Field::Gauge(Gauge::Lapse))[index] = lapse;
        }

        Ok(())
    }
}

pub fn schwarzschild(matches: &ArgMatches) -> eyre::Result<()> {
    let output_directory = matches
        .get_one::<PathBuf>("directory")
        .cloned()
        .ok_or_else(|| eyre::eyre!("failed to specify directory argument"))?;

    let output = misc::abs_or_relative(&output_directory)?;

    // Create output directory
    std::fs::create_dir_all(&output)?;

    let domain = Rectangle {
        origin: [0., 0.],
        size: [10.0, 10.0],
    };

    let mut mesh = Mesh::new(
        domain,
        6,
        3,
        FaceArray::from_fn(|face| match face.side {
            false => BoundaryClass::Ghost,
            true => BoundaryClass::OneSided,
        }),
    );
    mesh.refine_global();

    // Run 10 refinements to reach a suitable mesh.
    for _ in 0..10 {
        let mut system = SystemVec::new(Fields {
            scalar_fields: Vec::new(),
        });
        system.resize(mesh.num_nodes());
        system.contigious_mut().fill(0.0);

        mesh.evaluate(
            4,
            SchwarzschildData { mass: 1.0 },
            SystemSlice::empty(),
            system.as_mut_slice(),
        )
        .unwrap();
        mesh.fill_boundary(Order::<4>, FieldConditions, system.as_mut_slice());

        // Perform amr
        mesh.flag_wavelets(4, 1e-13, 1e-6, system.as_slice());
        mesh.balance_flags();

        let mut flag_debug = vec![0; mesh.num_nodes()];
        let mut block_debug = vec![0; mesh.num_nodes()];
        let mut cell_debug = vec![0; mesh.num_nodes()];

        mesh.flags_debug(&mut flag_debug);
        mesh.block_debug(&mut block_debug);
        mesh.cell_debug(&mut cell_debug);

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_system(system.as_slice());
        checkpoint.save_int_field("Flags", &flag_debug);
        checkpoint.save_int_field("Blocks", &block_debug);
        checkpoint.save_int_field("Cell", &cell_debug);

        checkpoint.export_vtu(
            output.join(format!("schwarzschild{}.vtu", mesh.max_level())),
            ExportVtuConfig {
                title: "schwarzschild".into(),
                ghost: false,
                stride: 1,
            },
        )?;

        if !mesh.requires_regridding() {
            println!("Refined to requisite tolerance");
            break;
        }

        // Otherwise regrid and loop
        mesh.regrid();
    }

    Ok(())
}

pub trait CommandExt {
    fn schwarzschild_args(self) -> Self;
}

impl CommandExt for Command {
    fn schwarzschild_args(self) -> Self {
        self.subcommand(
            Command::new("schwarzschild")
                .name("schwarzschild")
                .about("Subcommand for simulating schwarzschild spacetime")
                .arg(
                    arg!(--directory <FILE> "Sets the output directory")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                ),
        )
    }
}
