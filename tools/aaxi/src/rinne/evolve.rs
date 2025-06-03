use crate::{
    config::{Config, GaugeCondition},
    history::{RunHistory, RunRecord, RunStatus},
    misc,
    rinne::{
        Constraint, Field, FieldConditions, Fields, Gauge, Metric, ScalarField,
        eqs::{DynamicalData, DynamicalDerivs, ScalarFieldData, ScalarFieldDerivs, evolution},
        horizon::ApparentHorizonFinder,
    },
};
use aeon::{
    prelude::*,
    solver::{Integrator, Method},
};
use console::style;
use datasize::DataSize as _;
use indicatif::{HumanBytes, HumanCount, HumanDuration, MultiProgress, ProgressBar};
use std::{
    convert::Infallible,
    time::{Duration, Instant},
};

const ORDER: Order<4> = Order::<4>;

#[derive(Clone)]
struct FieldDerivs {
    gauge: GaugeCondition,
}

impl Function<2> for FieldDerivs {
    type Input = Fields;
    type Output = Fields;
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Infallible> {
        let grr_f = input.field(Field::Metric(Metric::Grr));
        let grz_f = input.field(Field::Metric(Metric::Grz));
        let gzz_f = input.field(Field::Metric(Metric::Gzz));
        let s_f = input.field(Field::Metric(Metric::S));

        let krr_f = input.field(Field::Metric(Metric::Krr));
        let krz_f = input.field(Field::Metric(Metric::Krz));
        let kzz_f = input.field(Field::Metric(Metric::Kzz));
        let y_f = input.field(Field::Metric(Metric::Y));

        let lapse_f = input.field(Field::Gauge(Gauge::Lapse));
        let shiftr_f = input.field(Field::Gauge(Gauge::Shiftr));
        let shiftz_f = input.field(Field::Gauge(Gauge::Shiftz));

        let theta_f = input.field(Field::Constraint(Constraint::Theta));
        let zr_f = input.field(Field::Constraint(Constraint::Zr));
        let zz_f = input.field(Field::Constraint(Constraint::Zz));

        let num_scalar_fields = output.system().num_scalar_fields();
        let scalar_fields = engine.alloc::<ScalarFieldData>(num_scalar_fields);
        let scalar_field_derivs = engine.alloc::<ScalarFieldDerivs>(num_scalar_fields);

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let pos = engine.position(vertex);
            let index = engine.index_from_vertex(vertex);

            macro_rules! derivatives {
                ($field:ident, $value:ident, $dr:ident, $dz:ident) => {
                    let $value = $field[index];
                    let $dr = engine.derivative($field, 0, vertex);
                    let $dz = engine.derivative($field, 1, vertex);
                };
            }

            macro_rules! second_derivatives {
                ($field:ident, $value:ident, $dr:ident, $dz:ident, $drr:ident, $drz:ident, $dzz:ident) => {
                    let $value = $field[index];
                    let $dr = engine.derivative($field, 0, vertex);
                    let $dz = engine.derivative($field, 1, vertex);

                    let $drr = engine.second_derivative($field, 0, vertex);
                    let $drz = engine.mixed_derivative($field, 0, 1, vertex);
                    let $dzz = engine.second_derivative($field, 1, vertex);
                };
            }

            // Metric
            second_derivatives!(grr_f, grr, grr_r, grr_z, grr_rr, grr_rz, grr_zz);
            second_derivatives!(gzz_f, gzz, gzz_r, gzz_z, gzz_rr, gzz_rz, gzz_zz);
            second_derivatives!(grz_f, grz, grz_r, grz_z, grz_rr, grz_rz, grz_zz);

            // S
            second_derivatives!(s_f, s, s_r, s_z, s_rr, s_rz, s_zz);

            // K
            derivatives!(krr_f, krr, krr_r, krr_z);
            derivatives!(kzz_f, kzz, kzz_r, kzz_z);
            derivatives!(krz_f, krz, krz_r, krz_z);

            // Y
            derivatives!(y_f, y, y_r, y_z);

            // Gauge
            second_derivatives!(
                lapse_f, lapse, lapse_r, lapse_z, lapse_rr, lapse_rz, lapse_zz
            );
            derivatives!(shiftr_f, shiftr, shiftr_r, shiftr_z);
            derivatives!(shiftz_f, shiftz, shiftz_r, shiftz_z);

            // Constraints
            derivatives!(theta_f, theta, theta_r, theta_z);
            derivatives!(zr_f, zr, zr_r, zr_z);
            derivatives!(zz_f, zz, zz_r, zz_z);

            for (i, mass) in output.system().scalar_fields().enumerate() {
                let phi = input.field(Field::ScalarField(ScalarField::Phi, i));
                let pi = input.field(Field::ScalarField(ScalarField::Pi, i));

                let scalar_field = &mut scalar_fields[i];

                scalar_field.phi = phi[index];
                scalar_field.phi_r = engine.derivative(phi, 0, vertex);
                scalar_field.phi_z = engine.derivative(phi, 1, vertex);
                scalar_field.phi_rr = engine.second_derivative(phi, 0, vertex);
                scalar_field.phi_rz = engine.mixed_derivative(phi, 0, 1, vertex);
                scalar_field.phi_zz = engine.second_derivative(phi, 1, vertex);

                scalar_field.pi = pi[index];
                scalar_field.pi_r = engine.derivative(pi, 0, vertex);
                scalar_field.pi_z = engine.derivative(pi, 1, vertex);

                scalar_field.mass = mass;
            }

            let system = DynamicalData {
                grr,
                grr_r,
                grr_z,
                grr_rr,
                grr_rz,
                grr_zz,
                grz,
                grz_r,
                grz_z,
                grz_rr,
                grz_rz,
                grz_zz,
                gzz,
                gzz_r,
                gzz_z,
                gzz_rr,
                gzz_rz,
                gzz_zz,
                s,
                s_r,
                s_z,
                s_rr,
                s_rz,
                s_zz,

                krr,
                krr_r,
                krr_z,
                krz,
                krz_r,
                krz_z,
                kzz,
                kzz_r,
                kzz_z,
                y,
                y_r,
                y_z,

                theta,
                theta_r,
                theta_z,
                zr,
                zr_r,
                zr_z,
                zz,
                zz_r,
                zz_z,

                lapse,
                lapse_r,
                lapse_z,
                lapse_rr,
                lapse_rz,
                lapse_zz,
                shiftr,
                shiftr_r,
                shiftr_z,
                shiftz,
                shiftz_r,
                shiftz_z,
            };

            let mut derivs = DynamicalDerivs::default();
            evolution(
                system,
                scalar_fields,
                pos,
                &mut derivs,
                scalar_field_derivs,
                self.gauge,
            );

            output.field_mut(Field::Metric(Metric::Grr))[index] = derivs.grr_t;
            output.field_mut(Field::Metric(Metric::Grz))[index] = derivs.grz_t;
            output.field_mut(Field::Metric(Metric::Gzz))[index] = derivs.gzz_t;
            output.field_mut(Field::Metric(Metric::S))[index] = derivs.s_t;

            output.field_mut(Field::Metric(Metric::Krr))[index] = derivs.krr_t;
            output.field_mut(Field::Metric(Metric::Krz))[index] = derivs.krz_t;
            output.field_mut(Field::Metric(Metric::Kzz))[index] = derivs.kzz_t;
            output.field_mut(Field::Metric(Metric::Y))[index] = derivs.y_t;

            output.field_mut(Field::Constraint(Constraint::Theta))[index] = derivs.theta_t;
            output.field_mut(Field::Constraint(Constraint::Zr))[index] = derivs.zr_t;
            output.field_mut(Field::Constraint(Constraint::Zz))[index] = derivs.zz_t;

            output.field_mut(Field::Gauge(Gauge::Lapse))[index] = derivs.lapse_t;
            output.field_mut(Field::Gauge(Gauge::Shiftr))[index] = derivs.shiftr_t;
            output.field_mut(Field::Gauge(Gauge::Shiftz))[index] = derivs.shiftz_t;

            for i in 0..num_scalar_fields {
                let derivs = &scalar_field_derivs[i];
                output.field_mut(Field::ScalarField(ScalarField::Phi, i))[index] = derivs.phi;
                output.field_mut(Field::ScalarField(ScalarField::Pi, i))[index] = derivs.pi;
            }
        }

        Ok(())
    }
}

pub fn evolve_data(
    config: &Config,
    history: &mut RunHistory,
    mut mesh: Mesh<2>,
    mut fields: SystemVec<Fields>,
) -> eyre::Result<()> {
    // Save initial time
    let start = Instant::now();

    let output = config.output_dir()?;
    // Create output folder.
    std::fs::create_dir_all(output.join("evolve"))?;
    // Cache system
    let system = fields.system().clone();

    // Integrate
    let mut integrator = Integrator::new(Method::RK4KO6(config.evolve.dissipation));
    let mut time = 0.0;
    let mut step = 0;

    let mut proper_time = 0.0;

    let mut save_step = 0;
    let mut steps_since_regrid = 0;

    let max_steps = config.evolve.max_steps;
    let max_time = config.evolve.max_time;
    let max_proper_time = config.evolve.max_proper_time;
    let cfl = config.evolve.cfl;
    let regrid_steps = config.regrid.flag_interval;
    let lower = config.regrid.coarsen_error;
    let upper = config.regrid.refine_error;
    let max_level = config.limits.max_levels;

    let mut time_since_save = 0.0;
    let save_interval = if config.visualize.save_evolve {
        config.visualize.save_evolve_interval
    } else {
        f64::MAX
    };
    let visualize_stride = config.visualize.stride;

    // let mut finder = ApparentHorizonFinder::new();

    // Does the simulation disperse?
    let mut disperse = true;

    println!("Evolving data");

    // Setup progress bars
    let m = MultiProgress::new();
    // Max nodes
    let node_pb = m.add(ProgressBar::new(config.limits.max_nodes as u64));
    node_pb.set_style(misc::node_style());
    node_pb.enable_steady_tick(Duration::from_millis(100));
    node_pb.set_prefix("[Node]  ");
    // Max levels
    let level_pb = m.add(ProgressBar::new(config.limits.max_levels as u64));
    level_pb.set_style(misc::level_style());
    level_pb.enable_steady_tick(Duration::from_millis(100));
    level_pb.set_prefix("[Level] ");
    // Memory usage
    let memory_pb = m.add(ProgressBar::new(config.limits.max_memory as u64));
    memory_pb.set_style(misc::byte_style());
    memory_pb.enable_steady_tick(Duration::from_millis(100));
    memory_pb.set_prefix("[Memory]");
    // Step spinner
    let step_pb = m.add(ProgressBar::no_length());
    step_pb.set_style(misc::spinner_style());
    step_pb.enable_steady_tick(Duration::from_millis(100));
    step_pb.set_prefix("[Step] ");

    while time < max_time && proper_time < max_proper_time {
        assert!(fields.len() == mesh.num_nodes());
        // Fill boundaries
        mesh.fill_boundary(ORDER, FieldConditions, fields.as_mut_slice());

        // Check norm for NaN
        let norm = mesh.l2_norm_system(fields.as_slice());

        if norm.is_nan() || norm >= 1e60 {
            println!("{}", style(format!("Norm diverges: {:.5e}", norm)).red());
            disperse = false;
            history.set_status(RunStatus::NormDiverged);
            break;
        }

        if step >= max_steps {
            println!(
                "{}",
                style(format!("Steps exceded maximum allocated steps: {}", step)).red()
            );
            disperse = false;
            history.set_status(RunStatus::MaxStepsReached);
            break;
        }

        if mesh.num_nodes() >= config.limits.max_nodes {
            println!(
                "{}",
                style(format!(
                    "Nodes exceded maximum allocated nodes: {}",
                    mesh.num_nodes()
                ))
                .red()
            );
            disperse = false;
            history.set_status(RunStatus::MaxNodesReached);
            break;
        }

        let memory_usage = fields.estimate_heap_size()
            + integrator.estimate_heap_size()
            + mesh.estimate_heap_size();

        if memory_usage >= config.limits.max_memory {
            println!(
                "{}",
                style(format!(
                    "RAM usage exceded maximum allocated bytes: {}",
                    HumanBytes(memory_usage as u64),
                ))
                .red()
            );
            disperse = false;
            history.set_status(RunStatus::MaxMemoryReached);
            break;
        }

        // Get step size
        let h = mesh.min_spacing() * cfl;

        // Periodically regrid mesh.
        if steps_since_regrid >= regrid_steps {
            steps_since_regrid = 0;

            mesh.flag_wavelets(4, lower, upper, fields.as_slice());
            mesh.limit_level_range_flags(1, max_level);
            mesh.balance_flags();
            mesh.regrid();

            // Copy system into tmp scratch space (provieded by dissipation).
            let scratch = integrator.scratch(fields.contigious().len());
            scratch.copy_from_slice(fields.contigious());
            fields.resize(mesh.num_nodes());
            mesh.transfer_system(
                ORDER,
                SystemSlice::from_contiguous(&scratch, &system),
                fields.as_mut_slice(),
            );

            continue;
        }

        // Periodically save output data
        if time_since_save >= save_interval {
            time_since_save -= save_interval;

            // Output current system to disk
            let mut checkpoint = Checkpoint::default();
            checkpoint.attach_mesh(&mesh);
            checkpoint.save_system(fields.as_slice());
            checkpoint.export_vtu(
                output
                    .join("evolve")
                    .join(format!("{}_{save_step}.vtu", config.name)),
                ExportVtuConfig {
                    title: config.name.clone(),
                    ghost: false,
                    stride: visualize_stride.into_int(),
                },
            )?;

            save_step += 1;
        }

        // Compute step
        integrator
            .step(
                &mut mesh,
                ORDER,
                FieldConditions,
                FieldDerivs {
                    gauge: config.evolve.gauge,
                },
                h,
                fields.as_mut_slice(),
            )
            .unwrap();

        let lapse = mesh.bottom_left_value(fields.field(Field::Gauge(Gauge::Lapse)));

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;

        proper_time += h * lapse;

        node_pb.set_position(mesh.num_nodes() as u64);
        level_pb.set_position(mesh.max_level() as u64);
        memory_pb.set_position(memory_usage as u64);

        step_pb.inc(1);
        step_pb.set_message(format!(
            "Step: {}, Proper Time {:.8}, Lapse {:.5e}",
            step, proper_time, lapse
        ));

        // There is a chance that lapse is now NaN, which should trigger collapse,
        // but might be interpreted as disspersion because `NaN > max_proper_time`
        // Check again
        if lapse.is_nan() || lapse.is_infinite() || lapse.abs() == 0.0 {
            println!(
                "{}",
                style(format!("Lapse Collapses, α = {:.5e}", lapse)).red()
            );
            disperse = false;
            history.set_status(RunStatus::NormDiverged);
            break;
        }

        // Serialize those values
        history.write_record(RunRecord {
            step,
            time,
            proper_time,
            lapse,
        })?;
    }

    m.clear()?;

    // Flush record at end of execution.
    history.flush()?;

    if disperse {
        println!("{}", style(format!("System disperses")).cyan());
        history.set_status(RunStatus::Dispersed);
    } else {
        println!("{}", style(format!("System collapses")).cyan());
    }

    println!(
        "Final evolution takes {}, {} steps",
        HumanDuration(start.elapsed()),
        HumanCount(step as u64),
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
        HumanBytes((fields.estimate_heap_size() + integrator.estimate_heap_size()) as u64)
    );

    Ok(())
}
