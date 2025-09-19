use crate::eqs;
use crate::run::config::{Config, Initial, Logging, Source};
use crate::run::interval::Interval;
use crate::systems::{self, FieldConditions, save_image};
use aeon::kernel::ScalarConditions;
use aeon::prelude::*;
use aeon::{
    mesh::{Gaussian, Mesh},
    solver::{HyperRelaxSolver, SolverCallback},
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

mod garfinkle {
    pub const S_CH: usize = 0;

    pub fn phi_ch(i: usize) -> usize {
        1 + 2 * i
    }

    pub fn num_channels(scalar_fields: usize) -> usize {
        1 + scalar_fields
    }
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
    fn kind(&self, channel: usize, face: Face<2>) -> BoundaryKind {
        if face.side {
            return BoundaryKind::Radiative;
        }

        let axes = match channel {
            garfinkle::S_CH => [BoundaryKind::AntiSymmetric, BoundaryKind::Symmetric],
            _ => [BoundaryKind::Symmetric, BoundaryKind::Symmetric],
        };

        axes[face.axis]
    }

    fn radiative(&self, _channel: usize, _position: [f64; 2]) -> RadiativeParams {
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
    context: ImageRef<'a>,
    scalar_fields: &'a [f64],
}

impl<'a> Function<2> for Hamiltonian<'a> {
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: ImageRef,
        mut output: ImageMut,
    ) -> Result<(), Infallible> {
        let context = self.context.slice(engine.node_range());
        let seed = context.channel(garfinkle::S_CH);

        let psi = input.channel(0);
        let dest = output.channel_mut(0);

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

            for (i, &mass) in self.scalar_fields.iter().enumerate() {
                let mass2 = mass * mass;
                let phi = context.channel(garfinkle::phi_ch(i));

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
    scalar_fields: &'a [f64],
}

impl<'a> Function<2> for FieldsFromGarfinkle<'a> {
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        context: ImageRef,
        mut output: ImageMut,
    ) -> Result<(), Infallible> {
        let psi = &self.psi[engine.node_range()];
        let seed = context.channel(garfinkle::S_CH);

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            let [rho, _] = engine.position(vertex);

            let conformal = psi[index].powi(4) * (2.0 * rho * seed[index]).exp();

            output.channel_mut(systems::GRR_CH)[index] = conformal;
            output.channel_mut(systems::GRZ_CH)[index] = 0.0;
            output.channel_mut(systems::GZZ_CH)[index] = conformal;
            output.channel_mut(systems::S_CH)[index] = -seed[index];

            output.channel_mut(systems::KRR_CH)[index] = 0.0;
            output.channel_mut(systems::KRZ_CH)[index] = 0.0;
            output.channel_mut(systems::KZZ_CH)[index] = 0.0;
            output.channel_mut(systems::Y_CH)[index] = 0.0;

            output.channel_mut(systems::THETA_CH)[index] = 0.0;
            output.channel_mut(systems::ZR_CH)[index] = 0.0;
            output.channel_mut(systems::ZZ_CH)[index] = 0.0;

            output.channel_mut(systems::LAPSE_CH)[index] = 1.0;
            output.channel_mut(systems::SHIFTR_CH)[index] = 0.0;
            output.channel_mut(systems::SHIFTZ_CH)[index] = 0.0;

            for (i, _) in self.scalar_fields.iter().enumerate() {
                output.channel_mut(systems::phi_ch(i))[index] =
                    context.channel(garfinkle::phi_ch(i))[index];
                output.channel_mut(systems::pi_ch(i))[index] = 0.0;
            }
        }

        Ok(())
    }
}

fn solve_order<S: SolverCallback<2> + Send + Sync>(
    order: usize,
    mesh: &mut Mesh<2>,
    initial: &Initial,
    callback: S,
    sources: &[Source],
    mut system: ImageMut,
) -> eyre::Result<()>
where
    S::Error: Error + Send + Sync + 'static,
{
    let num_scalar_field = sources
        .iter()
        .filter(|&source| matches!(source, Source::ScalarField { .. }))
        .count();

    // Retrieve number of nodes in current version of mesh.
    let num_nodes = mesh.num_nodes();

    let mut psi = vec![0.0; num_nodes];
    let mut context = Image::new(garfinkle::num_channels(num_scalar_field), num_nodes);

    // Compute seed values.
    mesh.project(
        order,
        SeedProjection(sources),
        context.channel_mut(garfinkle::S_CH),
    );

    // Compute scalar field values.
    let mut scalar_fields = Vec::new();
    let mut scalar_field_index = 0;
    for source in sources {
        if let Source::ScalarField {
            amplitude,
            sigma,
            mass,
        } = source
        {
            mesh.project(
                order,
                Gaussian {
                    amplitude: amplitude.unwrap(),
                    sigma: [sigma.0.unwrap(), sigma.1.unwrap()],
                    center: [0.0; 2],
                },
                context.channel_mut(garfinkle::phi_ch(scalar_field_index)),
            );

            scalar_fields.push(mass.unwrap());
            scalar_field_index += 1;
        }
    }

    // Fill boundary conditions for context fields.
    mesh.fill_boundary(order, ContextConditions, context.as_mut());

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
            context: context.as_ref(),
            scalar_fields: &scalar_fields,
        },
        psi.as_mut_slice().into(),
    )?;

    mesh.evaluate(
        order,
        FieldsFromGarfinkle {
            psi: &psi,
            scalar_fields: &scalar_fields,
        },
        context.as_ref(),
        system.rb_mut(),
    )
    .unwrap();

    mesh.fill_boundary(order, FieldConditions, system.rb_mut());

    Ok(())
}

// *******************************
// Initial Data ******************
// *******************************

#[derive(Clone)]
enum Spinners {
    Progress(ProgressBar),
    Incremental(Interval),
}

struct IterCallback<'a> {
    config: &'a Config,
    spinners: Spinners,
    output: &'a Path,
    step_count: &'a mut usize,
}

impl<'a> SolverCallback<2> for IterCallback<'a> {
    type Error = std::io::Error;

    fn callback(
        &mut self,
        mesh: &Mesh<2>,
        input: ImageRef,
        output: ImageRef,
        iteration: usize,
    ) -> Result<(), Self::Error> {
        *self.step_count += 1;

        match &self.spinners {
            Spinners::Progress(pb) => {
                pb.set_message(format!("Step: {}", iteration));
                pb.inc(1);
            }
            Spinners::Incremental(interval) => match interval {
                Interval::Steps { steps } => {
                    if iteration % steps == 0 {
                        println!(
                            "Initial Relax; levels: {}, iteration: {}",
                            mesh.num_levels(),
                            iteration
                        );
                    }
                }
                _ => unimplemented!("Incremental logging interval for initial data must be steps"),
            },
        }

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
        checkpoint.save_field("Solution", input.channel(0));
        checkpoint.save_field("Derivative", output.channel(0));
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
pub fn initial_data(config: &Config) -> eyre::Result<(Mesh<2>, Image)> {
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
        let system = checkpoint.read_image("Data");

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

    // ************************************
    // Visualization

    // Path for all visualization data.
    if config.visualize.initial || config.visualize.initial_levels || config.visualize.initial_relax
    {
        std::fs::create_dir_all(&output.join("initial"))?;
    }

    // ************************************
    // Solve

    let num_scalar_field = config
        .sources
        .iter()
        .filter(|&source| matches!(source, Source::ScalarField { .. }))
        .count();

    let num_channels = systems::num_channels(num_scalar_field);

    // Allocate memory for system and transfer buffers.
    let mut transfer = Image::new(num_channels, 0);
    let mut system = Image::new(num_channels, 0);
    system.resize(mesh.num_nodes());

    println!("Relaxing Initial Data");

    // Progress bars for relaxation
    let m = MultiProgress::new();

    let mut step_count = 0;

    loop {
        let spinners = match config.logging {
            Logging::Progress => Spinners::Progress({
                let pb = m.add(ProgressBar::no_length());
                pb.set_style(progress::spinner_style());
                pb.set_prefix(format!("[Level {}]", mesh.num_levels()));
                pb.enable_steady_tick(Duration::from_millis(100));

                pb
            }),
            Logging::Incremental { initial, .. } => Spinners::Incremental(initial),
        };

        solve_order(
            config.order,
            &mut mesh,
            &config.initial,
            IterCallback {
                config,
                spinners: spinners.clone(),
                output: &output,
                step_count: &mut step_count,
            },
            &config.sources,
            system.as_mut(),
        )?;

        match spinners {
            Spinners::Progress(pb) => {
                pb.finish_with_message(format!(
                    "Relaxed in {} steps, {} nodes",
                    pb.position(),
                    mesh.num_nodes()
                ));
            }
            Spinners::Incremental(..) => {}
        }

        if config.visualize.initial_levels {
            let mut checkpoint = Checkpoint::default();
            checkpoint.attach_mesh(&mesh);
            save_image(&mut checkpoint, system.as_ref());
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
            system.as_ref(),
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
        transfer.storage_mut().clone_from_slice(system.storage());
        mesh.regrid();
        system.resize(mesh.num_nodes());

        mesh.transfer_system(config.order, transfer.as_ref(), system.as_mut());
    }

    if config.visualize.initial {
        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        save_image(&mut checkpoint, system.as_ref());
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
        HumanCount(step_count as u64),
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
        checkpoint.save_image("Data", system.as_ref());
        checkpoint.export_dat(&init_cache)?;

        println!(
            "Successfully wrote initial data cache: {}",
            style(init_cache.display()).cyan()
        );
    }

    Ok((mesh, system))
}
