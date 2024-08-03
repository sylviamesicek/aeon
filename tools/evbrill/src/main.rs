#![allow(mixed_script_confusables)]

use aeon::fd::{Discretization, ExportVtkConfig};
use aeon::prelude::*;
use aeon::{array::Array, system::field_count};
use reborrow::{Reborrow, ReborrowMut};

// mod eqs;
pub mod equations;

use equations::HyperbolicSystem;

const STEPS: usize = 5000;
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
    Debug1,
    Debug2,
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

            Dynamic::Debug1 | Dynamic::Debug2 => [true, true],
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

        let derivs = equations::hyperbolic(system, [rho, z]);

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

        result.set_field(Dynamic::Debug1, derivs.debug1);
        result.set_field(Dynamic::Debug2, derivs.debug2);

        result
    }
}

// impl SystemOperator<2> for DynamicDerivs {
//     type Label = Dynamic;

//     fn apply(
//         &self,
//         block: Block<2>,
//         pool: &MemPool,
//         src: SystemSlice<'_, Self::Label>,
//         mut dest: SystemSliceMut<'_, Self::Label>,
//     ) {
//         let node_count = block.node_count();

//         macro_rules! derivatives {
//             ($field:ident, $value:ident, $dr:ident, $dz:ident) => {
//                 let $value = src.field(Dynamic::$field);
//                 let $dr = pool.alloc_scalar(node_count);
//                 let $dz = pool.alloc_scalar(node_count);

//                 block.derivative::<4>(0, &DynamicBoundary.field(Dynamic::$field), $value, $dr);
//                 block.derivative::<4>(1, &DynamicBoundary.field(Dynamic::$field), $value, $dz);
//             };
//         }

//         macro_rules! second_derivatives {
//             ($field:ident, $value:ident, $dr:ident, $dz:ident, $drr:ident, $drz:ident, $dzz:ident) => {
//                 let $value = src.field(Dynamic::$field);
//                 let $dr = pool.alloc_scalar(node_count);
//                 let $dz = pool.alloc_scalar(node_count);
//                 let $drr = pool.alloc_scalar(node_count);
//                 let $drz = pool.alloc_scalar(node_count);
//                 let $dzz = pool.alloc_scalar(node_count);

//                 block.derivative::<4>(0, &DynamicBoundary.field(Dynamic::$field), $value, $dr);
//                 block.derivative::<4>(1, &DynamicBoundary.field(Dynamic::$field), $value, $dz);
//                 block.second_derivative::<4>(
//                     0,
//                     &DynamicBoundary.field(Dynamic::$field),
//                     $value,
//                     $drr,
//                 );
//                 block.derivative::<4>(1, &DynamicBoundary.field(Dynamic::$field), $dr, $drz);
//                 block.second_derivative::<4>(
//                     1,
//                     &DynamicBoundary.field(Dynamic::$field),
//                     $value,
//                     $dzz,
//                 );
//             };
//         }

//         // Metric
//         second_derivatives!(Grr, grr, grr_r, grr_z, grr_rr, grr_rz, grr_zz);
//         second_derivatives!(Gzz, gzz, gzz_r, gzz_z, gzz_rr, gzz_rz, gzz_zz);
//         second_derivatives!(Grz, grz, grz_r, grz_z, grz_rr, grz_rz, grz_zz);

//         // S
//         second_derivatives!(S, s, s_r, s_z, s_rr, s_rz, s_zz);

//         // K
//         derivatives!(Krr, krr, krr_r, krr_z);
//         derivatives!(Kzz, kzz, kzz_r, kzz_z);
//         derivatives!(Krz, krz, krz_r, krz_z);

//         // Y
//         derivatives!(Y, y, y_r, y_z);

//         // Gauge
//         second_derivatives!(Lapse, lapse, lapse_r, lapse_z, lapse_rr, lapse_rz, lapse_zz);
//         derivatives!(Shiftr, shiftr, shiftr_r, shiftr_z);
//         derivatives!(Shiftz, shiftz, shiftz_r, shiftz_z);

//         // Constraints
//         derivatives!(Theta, theta, theta_r, theta_z);
//         derivatives!(Zr, zr, zr_r, zr_z);
//         derivatives!(Zz, zz, zz_r, zz_z);

//         // Now invoke generated equations

//         for vertex in block.iter() {
//             let [rho, z] = block.position(vertex);
//             let index = block.index_from_vertex(vertex);

//             let system = HyperbolicSystem {
//                 grr: grr[index],
//                 grr_r: grr_r[index],
//                 grr_z: grr_z[index],
//                 grr_rr: grr_rr[index],
//                 grr_rz: grr_rz[index],
//                 grr_zz: grr_zz[index],
//                 grz: grz[index],
//                 grz_r: grz_r[index],
//                 grz_z: grz_z[index],
//                 grz_rr: grz_rr[index],
//                 grz_rz: grz_rz[index],
//                 grz_zz: grz_zz[index],
//                 gzz: gzz[index],
//                 gzz_r: gzz_r[index],
//                 gzz_z: gzz_z[index],
//                 gzz_rr: gzz_rr[index],
//                 gzz_rz: gzz_rz[index],
//                 gzz_zz: gzz_zz[index],
//                 s: s[index],
//                 s_r: s_r[index],
//                 s_z: s_z[index],
//                 s_rr: s_rr[index],
//                 s_rz: s_rz[index],
//                 s_zz: s_zz[index],

//                 krr: krr[index],
//                 krr_r: krr_r[index],
//                 krr_z: krr_z[index],
//                 krz: krz[index],
//                 krz_r: krz_r[index],
//                 krz_z: krz_z[index],
//                 kzz: kzz[index],
//                 kzz_r: kzz_r[index],
//                 kzz_z: kzz_z[index],
//                 y: y[index],
//                 y_r: y_r[index],
//                 y_z: y_z[index],

//                 theta: theta[index],
//                 theta_r: theta_r[index],
//                 theta_z: theta_z[index],
//                 zr: zr[index],
//                 zr_r: zr_r[index],
//                 zr_z: zr_z[index],
//                 zz: zz[index],
//                 zz_r: zz_r[index],
//                 zz_z: zz_z[index],

//                 lapse: lapse[index],
//                 lapse_r: lapse_r[index],
//                 lapse_z: lapse_z[index],
//                 lapse_rr: lapse_rr[index],
//                 lapse_rz: lapse_rz[index],
//                 lapse_zz: lapse_zz[index],
//                 shiftr: shiftr[index],
//                 shiftr_r: shiftr_r[index],
//                 shiftr_z: shiftr_z[index],
//                 shiftz: shiftz[index],
//                 shiftz_r: shiftz_r[index],
//                 shiftz_z: shiftz_z[index],
//             };

//             // let on_axis = vertex[0] == 0;

//             // let derivs = if on_axis {
//             //     assert!(rho.abs() <= 10e-10);
//             //     hyperbolic_regular(system, rho, z)
//             // } else {
//             //     assert!(rho.abs() >= 10e-10);
//             //     hyperbolic(system, rho, z)
//             // };

//             let derivs = equations::hyperbolic(system, [rho, z]);

//             dest.field_mut(Dynamic::Grr)[index] = derivs.grr_t;
//             dest.field_mut(Dynamic::Grz)[index] = derivs.grz_t;
//             dest.field_mut(Dynamic::Gzz)[index] = derivs.gzz_t;
//             dest.field_mut(Dynamic::S)[index] = derivs.s_t;

//             dest.field_mut(Dynamic::Krr)[index] = derivs.krr_t;
//             dest.field_mut(Dynamic::Krz)[index] = derivs.krz_t;
//             dest.field_mut(Dynamic::Kzz)[index] = derivs.kzz_t;
//             dest.field_mut(Dynamic::Y)[index] = derivs.y_t;

//             dest.field_mut(Dynamic::Theta)[index] = derivs.theta_t;
//             dest.field_mut(Dynamic::Zr)[index] = derivs.zr_t;
//             dest.field_mut(Dynamic::Zz)[index] = derivs.zz_t;

//             dest.field_mut(Dynamic::Lapse)[index] = derivs.lapse_t;
//             dest.field_mut(Dynamic::Shiftr)[index] = derivs.shiftr_t;
//             dest.field_mut(Dynamic::Shiftz)[index] = derivs.shiftz_t;

//             dest.field_mut(Dynamic::Debug1)[index] = derivs.debug1;
//             dest.field_mut(Dynamic::Debug2)[index] = derivs.debug2;
//         }

//         let vertex_size = block.vertex_size();

//         let border = block
//             .face_plane(Face::positive(0))
//             .chain(block.face_plane(Face::positive(1)));

//         for vertex in border {
//             let rho_axis = vertex[0] == vertex_size[0] - 1;
//             let z_axis = vertex[1] == vertex_size[1] - 1;

//             assert!(rho_axis || z_axis);

//             // Computer inner point for approximating higher order r dependence
//             let mut inner = vertex;

//             if rho_axis {
//                 inner[0] = vertex_size[0] - 2;
//             }

//             if z_axis {
//                 inner[1] = vertex_size[1] - 2;
//             }

//             let [inner_rho, inner_z] = block.position(inner);
//             let inner_r = (inner_rho * inner_rho + inner_z * inner_z).sqrt();

//             let inner_index = block.index_from_vertex(inner);

//             macro_rules! inner_advection {
//                 ($field:ident, $value:ident, $dr:ident, $dz:ident, $target:literal, $k:ident) => {
//                     let adv = ($value[inner_index] - $target)
//                         + inner_rho * $dr[inner_index]
//                         + inner_z * $dz[inner_index];
//                     let adv_full = -adv / inner_r;
//                     let $k = inner_r
//                         * inner_r
//                         * inner_r
//                         * (dest.field(Dynamic::$field)[inner_index] - adv_full);
//                 };
//             }

//             inner_advection!(Grr, grr, grr_r, grr_z, 1.0, grr_k);
//             inner_advection!(Gzz, gzz, gzz_r, gzz_z, 1.0, gzz_k);
//             inner_advection!(Grz, grz, grz_r, grz_z, 0.0, grz_k);
//             inner_advection!(S, s, s_r, s_z, 0.0, s_k);

//             inner_advection!(Krr, krr, krr_r, krr_z, 0.0, krr_k);
//             inner_advection!(Kzz, kzz, kzz_r, kzz_z, 0.0, kzz_k);
//             inner_advection!(Krz, krz, krz_r, krz_z, 0.0, krz_k);
//             inner_advection!(Y, y, y_r, y_z, 0.0, y_k);

//             inner_advection!(Lapse, lapse, lapse_r, lapse_z, 1.0, lapse_k);
//             inner_advection!(Shiftr, shiftr, shiftr_r, shiftr_z, 0.0, shiftr_k);
//             inner_advection!(Shiftz, shiftz, shiftz_r, shiftz_z, 0.0, shiftz_k);

//             inner_advection!(Theta, theta, theta_r, theta_z, 0.0, theta_k);
//             inner_advection!(Zr, zr, zr_r, zr_z, 0.0, zr_k);
//             inner_advection!(Zz, zz, zz_r, zz_z, 0.0, zz_k);

//             let [rho, z] = block.position(vertex);
//             let r = (rho * rho + z * z).sqrt();

//             let index = block.index_from_vertex(vertex);

//             macro_rules! advection {
//                 ($field:ident, $value:ident, $dr:ident, $dz:ident, $target:literal, $k:ident) => {
//                     let adv = ($value[index] - $target) + rho * $dr[index] + z * $dz[index];
//                     let adv_full = -adv / r;
//                     dest.field_mut(Dynamic::$field)[index] = adv_full + $k / (r * r * r);
//                 };
//             }

//             advection!(Grr, grr, grr_r, grr_z, 1.0, grr_k);
//             advection!(Gzz, gzz, gzz_r, gzz_z, 1.0, gzz_k);
//             advection!(Grz, grz, grz_r, grz_z, 0.0, grz_k);
//             advection!(S, s, s_r, s_z, 0.0, s_k);

//             advection!(Krr, krr, krr_r, krr_z, 0.0, krr_k);
//             advection!(Kzz, kzz, kzz_r, kzz_z, 0.0, kzz_k);
//             advection!(Krz, krz, krz_r, krz_z, 0.0, krz_k);
//             advection!(Y, y, y_r, y_z, 0.0, y_k);

//             advection!(Lapse, lapse, lapse_r, lapse_z, 1.0, lapse_k);
//             advection!(Shiftr, shiftr, shiftr_r, shiftr_z, 0.0, shiftr_k);
//             advection!(Shiftz, shiftz, shiftz_r, shiftz_z, 0.0, shiftz_k);

//             advection!(Theta, theta, theta_r, theta_z, 0.0, theta_k);
//             advection!(Zr, zr, zr_r, zr_z, 0.0, zr_k);
//             advection!(Zz, zz, zz_r, zz_z, 0.0, zz_k);
//         }
//     }
// }

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
    let model = Model::<2>::import_dat("output/idbrill.dat")?;

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
