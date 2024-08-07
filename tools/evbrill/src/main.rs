#![allow(mixed_script_confusables)]

use aeon::fd::{Discretization, ExportVtkConfig};
use aeon::prelude::*;
use aeon::{array::Array, system::field_count};
use reborrow::{Reborrow, ReborrowMut};

// mod eqs;
pub mod explicit;
pub mod symbolicc;
pub mod types;

use types::HyperbolicSystem;

const STEPS: usize = 101;
const CFL: f64 = 0.1;
const ORDER: usize = 4;
const DISS_ORDER: usize = ORDER + 2;

/// Initial data in Rinne's hyperbolic variables.
#[derive(Clone, SystemLabel)]
pub enum Rinne {
    Conformal,
    Seed,
}

#[derive(Clone, SystemLabel)]
pub enum Dynamic {
    Grr,
    Grz,
    Gzz,
    S,
    Krr,
    Krz,
    Kzz,
    Y,
    Theta,
    Zr,
    Zz,
    Lapse,
    Shiftr,
    Shiftz,
}

#[derive(Clone)]
pub struct DynamicBC;

impl Boundary<2> for DynamicBC {
    fn kind(&self, face: Face<2>) -> BoundaryKind {
        match face.side {
            true => BoundaryKind::Radiative,
            false => BoundaryKind::Parity,
        }
    }
}

impl Conditions<2> for DynamicBC {
    type System = Dynamic;

    fn parity(&self, field: Self::System, face: Face<2>) -> bool {
        let axes = match field {
            Dynamic::Grr | Dynamic::Krr => [true, true],
            Dynamic::Grz | Dynamic::Krz => [false, false],
            Dynamic::Gzz | Dynamic::Kzz => [true, true],
            Dynamic::S | Dynamic::Y => [false, true],

            Dynamic::Theta | Dynamic::Lapse => [true, true],
            Dynamic::Zr | Dynamic::Shiftr => [false, true],
            Dynamic::Zz | Dynamic::Shiftz => [true, false],
        };
        axes[face.axis]
    }

    fn radiative(&self, field: Self::System, _position: [f64; 2]) -> f64 {
        match field {
            Dynamic::Grr | Dynamic::Gzz | Dynamic::Lapse => 1.0,
            _ => 0.0,
        }
    }
}

pub fn condition(field: Dynamic) -> SystemBC<Dynamic, DynamicBC> {
    SystemBC::new(field, DynamicBC)
}

#[derive(Clone)]
pub struct DynamicDerivs;

impl Operator<2> for DynamicDerivs {
    type System = Dynamic;
    type Context = Empty;

    fn apply(
        &self,
        engine: &impl Engine<2>,
        input: SystemFields<'_, Self::System>,
        _context: SystemFields<'_, Self::Context>,
    ) -> SystemValue<Self::System> {
        let [rho, z] = engine.position();

        macro_rules! derivatives {
            ($field:ident, $value:ident, $dr:ident, $dz:ident) => {
                let field = input.field(Dynamic::$field);

                let $value = engine.value(field);

                let grad = engine.gradient(condition(Dynamic::$field), field);
                let $dr = grad[0];
                let $dz = grad[1];
            };
        }

        macro_rules! second_derivatives {
            ($field:ident, $value:ident, $dr:ident, $dz:ident, $drr:ident, $drz:ident, $dzz:ident) => {
                let field = input.field(Dynamic::$field);

                let $value = engine.value(field);

                let grad = engine.gradient(condition(Dynamic::$field), field);
                let $dr = grad[0];
                let $dz = grad[1];

                let hess = engine.hessian(condition(Dynamic::$field), field);
                let $drr = hess[0][0];
                let $drz = hess[0][1];
                let $dzz = hess[1][1];
            };
        }

        // Metric
        second_derivatives!(Grr, grr, grr_r, grr_z, grr_rr, grr_rz, grr_zz);
        second_derivatives!(Gzz, gzz, gzz_r, gzz_z, gzz_rr, gzz_rz, gzz_zz);
        second_derivatives!(Grz, grz, grz_r, grz_z, grz_rr, grz_rz, grz_zz);

        // S
        second_derivatives!(S, s, s_r, s_z, s_rr, s_rz, s_zz);

        // K
        derivatives!(Krr, krr, krr_r, krr_z);
        derivatives!(Kzz, kzz, kzz_r, kzz_z);
        derivatives!(Krz, krz, krz_r, krz_z);

        // Y
        derivatives!(Y, y, y_r, y_z);

        // Gauge
        second_derivatives!(Lapse, lapse, lapse_r, lapse_z, lapse_rr, lapse_rz, lapse_zz);
        derivatives!(Shiftr, shiftr, shiftr_r, shiftr_z);
        derivatives!(Shiftz, shiftz, shiftz_r, shiftz_z);

        // Constraints
        derivatives!(Theta, theta, theta_r, theta_z);
        derivatives!(Zr, zr, zr_r, zr_z);
        derivatives!(Zz, zz, zz_r, zz_z);

        let system = HyperbolicSystem {
            grr: grr,
            grr_r: grr_r,
            grr_z: grr_z,
            grr_rr: grr_rr,
            grr_rz: grr_rz,
            grr_zz: grr_zz,
            grz: grz,
            grz_r: grz_r,
            grz_z: grz_z,
            grz_rr: grz_rr,
            grz_rz: grz_rz,
            grz_zz: grz_zz,
            gzz: gzz,
            gzz_r: gzz_r,
            gzz_z: gzz_z,
            gzz_rr: gzz_rr,
            gzz_rz: gzz_rz,
            gzz_zz: gzz_zz,
            s: s,
            s_r: s_r,
            s_z: s_z,
            s_rr: s_rr,
            s_rz: s_rz,
            s_zz: s_zz,

            krr: krr,
            krr_r: krr_r,
            krr_z: krr_z,
            krz: krz,
            krz_r: krz_r,
            krz_z: krz_z,
            kzz: kzz,
            kzz_r: kzz_r,
            kzz_z: kzz_z,
            y: y,
            y_r: y_r,
            y_z: y_z,

            theta: theta,
            theta_r: theta_r,
            theta_z: theta_z,
            zr: zr,
            zr_r: zr_r,
            zr_z: zr_z,
            zz: zz,
            zz_r: zz_r,
            zz_z: zz_z,

            lapse: lapse,
            lapse_r: lapse_r,
            lapse_z: lapse_z,
            lapse_rr: lapse_rr,
            lapse_rz: lapse_rz,
            lapse_zz: lapse_zz,
            shiftr: shiftr,
            shiftr_r: shiftr_r,
            shiftr_z: shiftr_z,
            shiftz: shiftz,
            shiftz_r: shiftz_r,
            shiftz_z: shiftz_z,
        };

        let derivs = explicit::hyperbolic(system, [rho, z]);

        let mut result = SystemValue::default();

        result.set_field(Dynamic::Grr, derivs.grr_t);
        result.set_field(Dynamic::Grz, derivs.grz_t);
        result.set_field(Dynamic::Gzz, derivs.gzz_t);
        result.set_field(Dynamic::S, derivs.s_t);

        result.set_field(Dynamic::Krr, derivs.krr_t);
        result.set_field(Dynamic::Krz, derivs.krz_t);
        result.set_field(Dynamic::Kzz, derivs.kzz_t);
        result.set_field(Dynamic::Y, derivs.y_t);

        result.set_field(Dynamic::Theta, derivs.theta_t);
        result.set_field(Dynamic::Zr, derivs.zr_t);
        result.set_field(Dynamic::Zz, derivs.zz_t);

        result.set_field(Dynamic::Lapse, derivs.lapse_t);
        result.set_field(Dynamic::Shiftr, derivs.shiftr_t);
        result.set_field(Dynamic::Shiftz, derivs.shiftz_t);

        result
    }
}

pub struct DynamicOde<'a> {
    discrete: &'a mut Discretization<2>,
}

impl<'a> Ode for DynamicOde<'a> {
    fn dim(&self) -> usize {
        field_count::<Dynamic>() * self.discrete.num_nodes()
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        self.discrete
            .order::<ORDER>()
            .fill_boundary(DynamicBC, SystemSliceMut::from_contiguous(system));
    }

    fn derivative(&mut self, system: &[f64], result: &mut [f64]) {
        let src = SystemSlice::from_contiguous(system);
        let mut dest = SystemSliceMut::from_contiguous(result);

        self.discrete.order::<ORDER>().apply(
            DynamicBC,
            DynamicDerivs,
            src.rb(),
            SystemSlice::empty(),
            dest.rb_mut(),
        );

        self.discrete
            .order::<ORDER>()
            .weak_boundary(DynamicBC, src.rb(), dest.rb_mut())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read model from disk
    let model = Model::<2>::import_dat("output/idbrill160.dat")?;

    // Build discretization
    let mut discrete = Discretization::new();
    discrete.set_mesh_from_model(&model);

    // Read initial data
    let initial = model.read_system::<Rinne>().unwrap();

    // Setup dynamic variables
    let mut dynamic = SystemVec::with_length(discrete.num_nodes());

    // Metric
    dynamic
        .field_mut(Dynamic::Grr)
        .copy_from_slice(initial.field(Rinne::Conformal));
    dynamic
        .field_mut(Dynamic::Gzz)
        .copy_from_slice(initial.field(Rinne::Conformal));
    dynamic.field_mut(Dynamic::Grz).fill(0.0);
    // S
    dynamic
        .field_mut(Dynamic::S)
        .copy_from_slice(initial.field(Rinne::Seed));
    // Extrinsic Curvature
    dynamic.field_mut(Dynamic::Krr).fill(0.0);
    dynamic.field_mut(Dynamic::Kzz).fill(0.0);
    dynamic.field_mut(Dynamic::Krz).fill(0.0);
    // Y
    dynamic.field_mut(Dynamic::Y).fill(0.0);
    // Constraint
    dynamic.field_mut(Dynamic::Theta).fill(0.0);
    dynamic.field_mut(Dynamic::Zr).fill(0.0);
    dynamic.field_mut(Dynamic::Zz).fill(0.0);
    // Gauge
    dynamic.field_mut(Dynamic::Lapse).fill(1.0);
    dynamic.field_mut(Dynamic::Shiftr).fill(0.0);
    dynamic.field_mut(Dynamic::Shiftz).fill(0.0);

    // Fill ghost nodes
    discrete
        .order::<ORDER>()
        .fill_boundary(DynamicBC, dynamic.as_mut_slice());

    // Begin integration
    let h = CFL * discrete.mesh().min_spacing();
    println!("Spacing {}", discrete.mesh().min_spacing());
    println!("Step Size {}", h);

    // Allocate vectors
    let mut derivs = SystemVec::with_length(discrete.num_nodes());
    let mut update = SystemVec::<Dynamic>::with_length(discrete.num_nodes());
    let mut dissipation = SystemVec::with_length(discrete.num_nodes());

    // Integrate
    let mut integrator = Rk4::new();

    for i in 0..STEPS {
        // Fill ghost nodes of system
        discrete
            .order::<ORDER>()
            .fill_boundary(DynamicBC, dynamic.as_mut_slice());

        discrete.order::<ORDER>().apply(
            DynamicBC,
            DynamicDerivs,
            dynamic.as_slice(),
            SystemSlice::empty(),
            derivs.as_mut_slice(),
        );

        // Output debugging data
        let norm = discrete.norm(dynamic.field(Dynamic::Theta).into());
        println!("Step {i}, Time {:.5} Norm {:.5e}", i as f64 * h, norm);

        if i % 10 == 0 {
            // Output current system to disk
            let mut model = Model::empty();
            model.set_mesh(discrete.mesh());
            model.write_system(dynamic.as_slice());

            for field in Dynamic::fields() {
                let name = format!("{}_dt", field.field_name());
                model.write_field(&name, derivs.field(field).to_vec());
            }

            model.export_vtk(
                format!("output/evbrill{}.vtu", i / 10),
                ExportVtkConfig {
                    title: "evbrill".to_string(),
                    ghost: false,
                },
            )?;
        }

        // Compute step
        integrator.step(
            h,
            &mut DynamicOde {
                discrete: &mut discrete,
            },
            dynamic.contigious(),
            update.contigious_mut(),
        );

        // Compute dissipation
        discrete.order::<DISS_ORDER>().dissipation(
            DynamicBC,
            dynamic.as_slice(),
            dissipation.as_mut_slice(),
        );

        // Add everything together
        for i in 0..dynamic.contigious().len() {
            dynamic.contigious_mut()[i] +=
                update.contigious()[i] + 0.5 * dissipation.contigious()[i];
        }
    }

    Ok(())
}
