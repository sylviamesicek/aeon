use crate::eqs;
use crate::run::config::{Config, Initial, Source};
use crate::systems::{Constraint, Field, FieldConditions, Fields, Gauge, Metric, ScalarField};
use aeon::prelude::*;
use aeon::{
    kernel::Kernels,
    mesh::{Gaussian, Mesh},
    solver::{HyperRelaxSolver, SolverCallback},
    system::System,
};
use aeon_app::progress;
use console::style;
use datasize::DataSize as _;
use eyre::eyre;
use indicatif::{HumanBytes, HumanCount, HumanDuration, MultiProgress, ProgressBar};
use reborrow::ReborrowMut;
use std::convert::Infallible;
use std::error::Error;
use std::path::Path;
use std::time::{Duration, Instant};

// *************************
// Garfinkle variables *****
// *************************

#[derive(Clone)]
struct ContextSystem {
    /// scalar field masses.
    pub scalar_fields: Vec<f64>,
}

impl ContextSystem {
    /// Returns the number of scalar fields.
    pub fn num_scalar_fields(&self) -> usize {
        self.scalar_fields.len()
    }

    /// Returns an interator over the scalar fields in this system.
    pub fn scalar_fields(&self) -> impl Iterator<Item = f64> + '_ {
        self.scalar_fields.iter().cloned()
    }
}

impl System for ContextSystem {
    const NAME: &'static str = "Context";

    type Label = Context;

    fn count(&self) -> usize {
        1 + self.num_scalar_fields()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        std::iter::once(Context::Seed).chain((0..self.num_scalar_fields()).map(Context::Phi))
    }

    fn label_index(&self, label: Self::Label) -> usize {
        match label {
            Context::Seed => 0,
            Context::Phi(id) => id + 1,
        }
    }

    fn label_name(&self, label: Self::Label) -> String {
        match label {
            Context::Seed => "Seed".to_string(),
            Context::Phi(id) => format!("Phi{id}"),
        }
    }

    fn label_from_index(&self, mut index: usize) -> Self::Label {
        if index < 1 {
            return Context::Seed;
        }
        index -= 1;

        Context::Phi(index)
    }
}

/// Context system label.
#[derive(Clone, Copy)]
enum Context {
    Seed,
    Phi(usize),
}

// ***********************
// Boundary conditions ***
// ***********************

/// Boundary conditions for psi.
#[derive(Clone)]
struct PsiCondition;

impl BoundaryConds<2> for PsiCondition {
    fn kind(&self, face: Face<2>) -> BoundaryKind {
        if face.side {
            return BoundaryKind::Radiative;
        }

        BoundaryKind::Symmetric
    }

    fn radiative(&self, _position: [f64; 2]) -> RadiativeParams {
        RadiativeParams::lightlike(1.0)
    }
}

/// Boundary Conditions for context variables.
#[derive(Clone)]
struct ContextConditions;

impl SystemBoundaryConds<2> for ContextConditions {
    type System = ContextSystem;

    fn kind(&self, label: <Self::System as System>::Label, face: Face<2>) -> BoundaryKind {
        if face.side {
            return BoundaryKind::Radiative;
        }

        let axes = match label {
            Context::Seed => [BoundaryKind::AntiSymmetric, BoundaryKind::Symmetric],
            Context::Phi(_) => [BoundaryKind::Symmetric, BoundaryKind::Symmetric],
        };

        axes[face.axis]
    }

    fn radiative(&self, _field: Context, _position: [f64; 2]) -> RadiativeParams {
        RadiativeParams::lightlike(0.0)
    }
}

/// Seed function projections.
#[derive(Clone)]
struct SeedProjection<'a>(&'a [Source]);

impl<'a> Projection<2> for SeedProjection<'a> {
    fn project(&self, [rho, z]: [f64; 2]) -> f64 {
        let rho2 = rho * rho;
        let z2 = z * z;

        let mut result = 0.0;

        for source in self.0 {
            match source {
                Source::Brill { amplitude, sigma } => {
                    let srho2 = sigma.0.unwrap() * sigma.0.unwrap();
                    let sz2 = sigma.1.unwrap() * sigma.1.unwrap();

                    result += rho * amplitude.unwrap() * (-rho2 / srho2 - z2 / sz2).exp()
                }
                _ => {}
            }
        }

        result
    }
}

/// Hamiltonian elliptic equation.
#[derive(Clone)]
struct Hamiltonian<'a> {
    context: SystemSlice<'a, ContextSystem>,
}

impl<'a> Function<2> for Hamiltonian<'a> {
    type Input = Scalar;
    type Output = Scalar;
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: SystemSlice<Self::Input>,
        output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Infallible> {
        let context = self.context.slice(engine.node_range());
        let seed = context.field(Context::Seed);

        let psi = input.into_scalar();
        let dest = output.into_scalar();

        // Iterate over vertices in block.
        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);

            let [rho, _] = engine.position(vertex);

            let psi_r = engine.derivative(psi, 0, vertex);
            let psi_rr = engine.second_derivative(psi, 0, vertex);
            let psi_zz = engine.second_derivative(psi, 1, vertex);

            let seed_r = engine.derivative(seed, 0, vertex);
            let seed_rr = engine.second_derivative(seed, 0, vertex);
            let seed_zz = engine.second_derivative(seed, 1, vertex);

            let laplacian = if rho.abs() <= 10e-10 {
                2.0 * psi_rr + psi_zz
            } else {
                psi_rr + psi_r / rho + psi_zz
            };

            let mut source = 0.0;

            for (i, mass) in context.system().scalar_fields().enumerate() {
                let mass2 = mass * mass;
                let phi = context.field(Context::Phi(i));

                let phi2 = phi[index] * phi[index];
                let phi_r = engine.derivative(phi, 0, vertex);
                let phi_z = engine.derivative(phi, 1, vertex);

                let kinetic = 0.5 * (phi_r * phi_r + phi_z * phi_z);
                let potential =
                    0.5 * (2.0 * rho * seed[index]).exp() * psi[index].powi(4) * mass2 * phi2;

                source += kinetic + potential;
            }

            dest[index] = laplacian
                + psi[index] / 4.0 * (rho * seed_rr + 2.0 * seed_r + rho * seed_zz)
                + psi[index] / 4.0 * source * eqs::KAPPA;
        }
        Ok(())
    }
}

/// Generate fields from Garfinkle variables.
#[derive(Clone)]
struct FieldsFromGarfinkle<'a> {
    psi: &'a [f64],
    context: SystemSlice<'a, ContextSystem>,
}

impl<'a> Function<2> for FieldsFromGarfinkle<'a> {
    type Input = Empty;
    type Output = Fields;
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        _: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Infallible> {
        let psi = &self.psi[engine.node_range()];
        let context = self.context.slice(engine.node_range());
        let seed = context.field(Context::Seed);

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            let [rho, _] = engine.position(vertex);

            let conformal = psi[index].powi(4) * (2.0 * rho * seed[index]).exp();

            output.field_mut(Field::Metric(Metric::Grr))[index] = conformal;
            output.field_mut(Field::Metric(Metric::Gzz))[index] = conformal;
            output.field_mut(Field::Metric(Metric::S))[index] = -seed[index];
        }

        Ok(())
    }
}

fn solve_order<const ORDER: usize, S: SolverCallback<2, Scalar> + Send + Sync>(
    order: Order<ORDER>,
    mesh: &mut Mesh<2>,
    initial: &Initial,
    callback: S,
    sources: &[Source],
    mut system: SystemSliceMut<Fields>,
) -> eyre::Result<()>
where
    Order<ORDER>: Kernels,
    S::Error: Error + Send + Sync + 'static,
{
    let num_scalar_field = system.system().num_scalar_fields();

    // Retrieve number of nodes in current version of mesh.
    let num_nodes = mesh.num_nodes();

    let mut psi = vec![0.0; num_nodes];
    let mut context = SystemVec::with_length(
        num_nodes,
        ContextSystem {
            scalar_fields: system.system().scalar_fields.clone(),
        },
    );

    // Compute seed values.
    mesh.project(
        ORDER,
        SeedProjection(sources),
        context.field_mut(Context::Seed),
    );

    // Compute scalar field values.
    let mut scalar_field_index = 0;
    for source in sources {
        if let Source::ScalarField {
            amplitude, sigma, ..
        } = source
        {
            mesh.project(
                ORDER,
                Gaussian {
                    amplitude: amplitude.unwrap(),
                    sigma: [sigma.0.unwrap(), sigma.1.unwrap()],
                    center: [0.0; 2],
                },
                context.field_mut(Context::Phi(scalar_field_index)),
            );

            scalar_field_index += 1;
        }
    }

    // Fill boundary conditions for context fields.
    mesh.fill_boundary(order, ContextConditions, context.as_mut_slice());

    // Initial Guess for Psi
    psi.fill(1.0);

    log::trace!(
        "Relaxing. Num Levels {}, Nodes: {}",
        mesh.num_levels(),
        mesh.num_nodes()
    );

    let mut solver = HyperRelaxSolver::new();
    solver.dampening = initial.relax.dampening;
    solver.max_steps = initial.relax.max_steps;
    solver.tolerance = initial.relax.tolerance;
    solver.cfl = initial.relax.cfl;
    solver.adaptive = true;

    solver.solve_with_callback(
        mesh,
        order,
        ScalarConditions(PsiCondition),
        callback,
        Hamiltonian {
            context: context.as_slice(),
        },
        SystemSliceMut::from_scalar(&mut psi),
    )?;

    mesh.evaluate(
        ORDER,
        FieldsFromGarfinkle {
            psi: &psi,
            context: context.as_slice(),
        },
        SystemSlice::empty(),
        system.rb_mut(),
    )
    .unwrap();

    // Copy scalar fields
    for i in 0..num_scalar_field {
        system
            .field_mut(Field::ScalarField(ScalarField::Phi, i))
            .copy_from_slice(context.field(Context::Phi(i)));
        system
            .field_mut(Field::ScalarField(ScalarField::Pi, i))
            .fill(0.0);
    }

    // Metric
    system.field_mut(Field::Metric(Metric::Grz)).fill(0.0);
    system.field_mut(Field::Metric(Metric::Krr)).fill(0.0);
    system.field_mut(Field::Metric(Metric::Kzz)).fill(0.0);
    system.field_mut(Field::Metric(Metric::Krz)).fill(0.0);
    system.field_mut(Field::Metric(Metric::Y)).fill(0.0);
    // Constraint
    system
        .field_mut(Field::Constraint(Constraint::Theta))
        .fill(0.0);
    system
        .field_mut(Field::Constraint(Constraint::Zr))
        .fill(0.0);
    system
        .field_mut(Field::Constraint(Constraint::Zz))
        .fill(0.0);
    // Gauge
    system.field_mut(Field::Gauge(Gauge::Lapse)).fill(1.0);
    system.field_mut(Field::Gauge(Gauge::Shiftr)).fill(0.0);
    system.field_mut(Field::Gauge(Gauge::Shiftz)).fill(0.0);

    mesh.fill_boundary(order, FieldConditions, system.rb_mut());

    Ok(())
}

// *******************************
// Initial Data ******************
// *******************************

struct IterCallback<'a> {
    config: &'a Config,
    pb: ProgressBar,
    output: &'a Path,
}

impl<'a> SolverCallback<2, Scalar> for IterCallback<'a> {
    type Error = std::io::Error;

    fn callback(
        &mut self,
        mesh: &Mesh<2>,
        input: SystemSlice<Scalar>,
        output: SystemSlice<Scalar>,
        iteration: usize,
    ) -> Result<(), Self::Error> {
        self.pb.set_message(format!("Step: {}", iteration));
        self.pb.inc(1);

        if !self.config.visualize.initial_relax {
            return Ok(());
        }

        let visualize_interval = self.config.visualize.initial_relax_interval.unwrap_steps();

        if iteration % visualize_interval != 0 {
            return Ok(());
        }

        let i = iteration / visualize_interval;

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_field("Solution", input.into_scalar());
        checkpoint.save_field("Derivative", output.into_scalar());
        checkpoint.export_vtu(
            self.output.join("initial").join(format!(
                "{}_levels_{}_iter_{}.vtu",
                self.config.name,
                mesh.num_levels(),
                i
            )),
            ExportVtuConfig {
                title: self.config.name.to_string(),
                ghost: false,
                stride: self.config.visualize.stride,
            },
        )?;

        Ok(())
    }
}

/// Solve rinne's initial data problem using Garfinkles variables.
pub fn initial_data(config: &Config) -> eyre::Result<(Mesh<2>, SystemVec<Fields>)> {
    // Save initial time
    let start = Instant::now();

    // Cache directory
    let output = config.output_dir()?;
    let cache = output.join("cache");
    let init_cache = cache.join(format!("{}_init.dat", config.name));

    'cache: {
        if !config.cache.initial {
            // Don't attempt to load cache
            break 'cache;
        }

        // Attempt to load file
        let Ok(checkpoint) = Checkpoint::<2>::import_dat(&init_cache) else {
            break 'cache;
        };

        let mesh = checkpoint.read_mesh();
        let system = checkpoint.read_system::<Fields>();

        println!(
            "Successfully read cached initial data: {}",
            style(init_cache.display()).cyan()
        );

        return Ok((mesh, system));
    };

    if config.cache.initial {
        println!(
            "Failed to read cached initial data: {}",
            style(init_cache.display()).yellow()
        );
    }

    // Build mesh
    let mut mesh = Mesh::new(
        HyperBox {
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
    for _ in 0..config.domain.global_refine {
        // Ensure we don't go above max levels
        if mesh.num_levels() >= config.limits.max_levels {
            break;
        }

        mesh.refine_global();
    }

    // Build fields from sources.
    let fields = Fields {
        scalar_fields: config
            .sources
            .iter()
            .flat_map(|source| {
                if let Source::ScalarField { mass, .. } = source {
                    Some(mass.unwrap())
                } else {
                    None
                }
            })
            .collect(),
    };

    // ************************************
    // Visualization

    // Path for all visualization data.
    if config.visualize.initial || config.visualize.initial_levels || config.visualize.initial_relax
    {
        std::fs::create_dir_all(&output.join("initial"))?;
    }

    // ************************************
    // Solve

    // Allocate memory for system and transfer buffers.
    let mut transfer = SystemVec::new(fields.clone());
    let mut system = SystemVec::new(fields.clone());
    system.resize(mesh.num_nodes());

    println!("Relaxing Initial Data");

    // Progress bars for relaxation
    let m = MultiProgress::new();

    let mut step_count = 0;

    loop {
        let pb = m.add(ProgressBar::no_length());
        pb.set_style(progress::spinner_style());
        pb.set_prefix(format!("[Level {}]", mesh.num_levels()));
        pb.enable_steady_tick(Duration::from_millis(100));

        match config.order {
            2 => {
                solve_order(
                    Order::<2>,
                    &mut mesh,
                    &config.initial,
                    IterCallback {
                        config,
                        pb: pb.clone(),
                        output: &output,
                    },
                    &config.sources,
                    system.as_mut_slice(),
                )?;
            }
            4 => {
                solve_order(
                    Order::<4>,
                    &mut mesh,
                    &config.initial,
                    IterCallback {
                        config,
                        pb: pb.clone(),
                        output: &output,
                    },
                    &config.sources,
                    system.as_mut_slice(),
                )?;
            }
            6 => {
                solve_order(
                    Order::<6>,
                    &mut mesh,
                    &config.initial,
                    IterCallback {
                        config,
                        pb: pb.clone(),
                        output: &output,
                    },
                    &config.sources,
                    system.as_mut_slice(),
                )?;
            }
            _ => return Err(eyre!("Invalid initial data type and order")),
        };

        pb.finish_with_message(format!(
            "Relaxed in {} steps, {} nodes",
            pb.position(),
            mesh.num_nodes()
        ));
        step_count += pb.position();

        if config.visualize.initial_levels {
            let mut checkpoint = Checkpoint::default();
            checkpoint.attach_mesh(&mesh);
            checkpoint.save_system(system.as_slice());
            checkpoint.export_vtu(
                output.join("initial").join(format!(
                    "{}_levels_{}.vtu",
                    &config.name,
                    mesh.num_levels()
                )),
                ExportVtuConfig {
                    title: config.name.clone(),
                    ghost: false,
                    stride: config.visualize.stride,
                },
            )?;
        }

        mesh.flag_wavelets(
            config.order,
            config.initial.coarsen_error,
            config.initial.refine_error,
            system.as_slice(),
        );
        mesh.balance_flags();

        // Check if we are done regridding
        if !mesh.requires_regridding() {
            log::trace!(
                "Sucessfully refined mesh to given accuracy: {:.5e}",
                config.initial.refine_error
            );
            break;
        }

        if mesh.num_levels() >= config.limits.max_levels
            || mesh.num_nodes() >= config.limits.max_nodes
        {
            log::error!(
                "Failed to solve initial data, levels: {}, nodes: {}",
                mesh.num_levels(),
                mesh.num_nodes()
            );
            return Err(eyre!("failed to refine within perscribed limits"));
        }

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
    }

    if config.visualize.initial {
        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_system(system.as_slice());
        checkpoint.export_vtu(
            output.join("initial").join(format!("{}.vtu", config.name)),
            ExportVtuConfig {
                title: config.name.clone(),
                ghost: false,
                stride: config.visualize.stride,
            },
        )?;
    }

    m.clear()?;

    println!(
        "Finished relaxing in {}, {} Steps",
        HumanDuration(start.elapsed()),
        HumanCount(step_count),
    );
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
        HumanBytes((system.estimate_heap_size() + transfer.estimate_heap_size()) as u64)
    );

    if config.cache.initial {
        // Ensure output directory exists
        std::fs::create_dir_all(cache)?;
        // Create checkpoint
        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_system::<Fields>(system.as_slice());
        checkpoint.export_dat(&init_cache)?;

        println!(
            "Successfully wrote initial data cache: {}",
            style(init_cache.display()).cyan()
        );
    }

    Ok((mesh, system))
}
