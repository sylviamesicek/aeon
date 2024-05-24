#![allow(unused_assignments)]

use aeon::prelude::*;
use aeon::{array::Array, mesh::field_count};
use std::path::PathBuf;
use std::{fs::File, io::Read};

mod eqs;

use eqs::{hyperbolic, hyperbolic_regular, HyperbolicSystem};

const STEPS: usize = 1000;
const CFL: f64 = 0.1;

#[derive(Clone, SystemLabel)]
pub enum InitialData {
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

pub struct ParityBoundary(bool, bool);

impl Boundary for ParityBoundary {
    fn face(&self, face: Face) -> BoundaryCondition {
        if face.axis == 0 && face.side == false {
            BoundaryCondition::Parity(self.0)
        } else if face.axis == 1 && face.side == false {
            BoundaryCondition::Parity(self.1)
        } else {
            BoundaryCondition::Free
        }
    }
}

pub struct DynamicBoundary;

impl SystemBoundary for DynamicBoundary {
    type Boundary = ParityBoundary;
    type Label = Dynamic;

    fn field(&self, label: Self::Label) -> Self::Boundary {
        let (rho, z) = match label {
            Dynamic::Grr | Dynamic::Krr => (true, true),
            Dynamic::Grz | Dynamic::Krz => (false, false),
            Dynamic::Gzz | Dynamic::Kzz => (true, true),
            Dynamic::S | Dynamic::Y => (false, true),

            Dynamic::Theta | Dynamic::Lapse => (true, true),
            Dynamic::Zr | Dynamic::Shiftr => (false, true),
            Dynamic::Zz | Dynamic::Shiftz => (true, false),
        };
        ParityBoundary(rho, z)
    }
}

pub struct DynamicDerivs;

impl SystemOperator<2> for DynamicDerivs {
    type Label = Dynamic;

    fn apply(
        &self,
        block: Block<2>,
        pool: &MemPool,
        src: SystemSlice<'_, Self::Label>,
        mut dest: SystemSliceMut<'_, Self::Label>,
    ) {
        let node_count = block.node_count();

        macro_rules! derivatives {
            ($field:ident, $value:ident, $dr:ident, $dz:ident) => {
                let $value = src.field(Dynamic::$field);
                let $dr = pool.alloc_scalar(node_count);
                let $dz = pool.alloc_scalar(node_count);

                block.derivative::<4>(0, &DynamicBoundary.field(Dynamic::$field), $value, $dr);
                block.derivative::<4>(1, &DynamicBoundary.field(Dynamic::$field), $value, $dz);
            };
        }

        macro_rules! second_derivatives {
            ($field:ident, $value:ident, $dr:ident, $dz:ident, $drr:ident, $drz:ident, $dzz:ident) => {
                let $value = src.field(Dynamic::$field);
                let $dr = pool.alloc_scalar(node_count);
                let $dz = pool.alloc_scalar(node_count);
                let $drr = pool.alloc_scalar(node_count);
                let $drz = pool.alloc_scalar(node_count);
                let $dzz = pool.alloc_scalar(node_count);

                block.derivative::<4>(0, &DynamicBoundary.field(Dynamic::$field), $value, $dr);
                block.derivative::<4>(1, &DynamicBoundary.field(Dynamic::$field), $value, $dz);
                block.second_derivative::<4>(
                    0,
                    &DynamicBoundary.field(Dynamic::$field),
                    $value,
                    $drr,
                );
                block.derivative::<4>(1, &DynamicBoundary.field(Dynamic::$field), $dr, $drz);
                block.second_derivative::<4>(
                    1,
                    &DynamicBoundary.field(Dynamic::$field),
                    $value,
                    $dzz,
                );
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

        // Now invoke generated equations

        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            let system = HyperbolicSystem {
                grr: grr[index],
                grr_r: grr_r[index],
                grr_z: grr_z[index],
                grr_rr: grr_rr[index],
                grr_rz: grr_rz[index],
                grr_zz: grr_zz[index],
                grz: grz[index],
                grz_r: grz_r[index],
                grz_z: grz_z[index],
                grz_rr: grz_rr[index],
                grz_rz: grz_rz[index],
                grz_zz: grz_zz[index],
                gzz: gzz[index],
                gzz_r: gzz_r[index],
                gzz_z: gzz_z[index],
                gzz_rr: gzz_rr[index],
                gzz_rz: gzz_rz[index],
                gzz_zz: gzz_zz[index],
                s: s[index],
                s_r: s_r[index],
                s_z: s_z[index],
                s_rr: s_rr[index],
                s_rz: s_rz[index],
                s_zz: s_zz[index],

                krr: krr[index],
                krr_r: krr_r[index],
                krr_z: krr_z[index],
                krz: krz[index],
                krz_r: krz_r[index],
                krz_z: krz_z[index],
                kzz: kzz[index],
                kzz_r: kzz_r[index],
                kzz_z: kzz_z[index],
                y: y[index],
                y_r: y_r[index],
                y_z: y_z[index],

                lapse: lapse[index],
                lapse_r: lapse_r[index],
                lapse_z: lapse_z[index],
                lapse_rr: lapse_rr[index],
                lapse_rz: lapse_rz[index],
                lapse_zz: lapse_zz[index],
                shiftr: shiftr[index],
                shiftr_r: shiftr_r[index],
                shiftr_z: shiftr_z[index],
                shiftz: shiftz[index],
                shiftz_r: shiftz_r[index],
                shiftz_z: shiftz_z[index],

                theta: theta[index],
                theta_r: theta_r[index],
                theta_z: theta_z[index],
                zr: zr[index],
                zr_r: zr_r[index],
                zr_z: zr_z[index],
                zz: zz[index],
                zz_r: zz_r[index],
                zz_z: zz_z[index],
            };

            let on_axis = vertex[0] == 0;

            let derivs = if on_axis {
                assert!(rho.abs() <= 10e-10);
                hyperbolic_regular(system, rho, z)
            } else {
                assert!(rho.abs() >= 10e-10);
                hyperbolic(system, rho, z)
            };

            dest.field_mut(Dynamic::Grr)[index] = derivs.grr_t;
            dest.field_mut(Dynamic::Grz)[index] = derivs.grz_t;
            dest.field_mut(Dynamic::Gzz)[index] = derivs.gzz_t;
            dest.field_mut(Dynamic::S)[index] = derivs.s_t;

            dest.field_mut(Dynamic::Krr)[index] = derivs.krr_t;
            dest.field_mut(Dynamic::Krz)[index] = derivs.krz_t;
            dest.field_mut(Dynamic::Kzz)[index] = derivs.kzz_t;
            dest.field_mut(Dynamic::Y)[index] = derivs.y_t;

            dest.field_mut(Dynamic::Lapse)[index] = derivs.lapse_t;
            dest.field_mut(Dynamic::Shiftr)[index] = derivs.shiftr_t;
            dest.field_mut(Dynamic::Shiftz)[index] = derivs.shiftz_t;

            dest.field_mut(Dynamic::Theta)[index] = derivs.theta_t;
            dest.field_mut(Dynamic::Zr)[index] = derivs.zr_t;
            dest.field_mut(Dynamic::Zz)[index] = derivs.zz_t;
        }

        let vertex_size = block.vertex_size();

        for vertex in block
            .face_plane(Face::positive(0))
            .chain(block.face_plane(Face::positive(1)))
        {
            let rho_axis = vertex[0] == vertex_size[0] - 1;
            let z_axis = vertex[1] == vertex_size[1] - 1;

            assert!(rho_axis || z_axis);

            // Computer inner point for approximating higher order r dependence
            let mut inner = vertex;

            if rho_axis {
                inner[0] = vertex_size[0] - 2;
            }

            if z_axis {
                inner[1] = vertex_size[1] - 2;
            }

            let [inner_rho, inner_z] = block.position(inner);
            let inner_r = (inner_rho * inner_rho + inner_z * inner_z).sqrt();

            let inner_index = block.index_from_vertex(inner);

            macro_rules! inner_advection {
                ($field:ident, $value:ident, $dr:ident, $dz:ident, $target:literal, $k:ident) => {
                    let adv = ($value[inner_index] - $target)
                        + inner_rho * $dr[inner_index]
                        + inner_z * $dz[inner_index];
                    let adv_full = -adv / inner_r;
                    let $k = inner_r
                        * inner_r
                        * inner_r
                        * (dest.field(Dynamic::$field)[inner_index] - adv_full);
                };
            }

            inner_advection!(Grr, grr, grr_r, grr_z, 1.0, grr_k);
            inner_advection!(Gzz, gzz, gzz_r, gzz_z, 1.0, gzz_k);
            inner_advection!(Grz, grz, grz_r, grz_z, 0.0, grz_k);
            inner_advection!(S, s, s_r, s_z, 0.0, s_k);

            inner_advection!(Krr, krr, krr_r, krr_z, 0.0, krr_k);
            inner_advection!(Kzz, kzz, kzz_r, kzz_z, 0.0, kzz_k);
            inner_advection!(Krz, krz, krz_r, krz_z, 0.0, krz_k);
            inner_advection!(Y, y, y_r, y_z, 0.0, y_k);

            inner_advection!(Lapse, lapse, lapse_r, lapse_z, 1.0, lapse_k);
            inner_advection!(Shiftr, shiftr, shiftr_r, shiftr_z, 0.0, shiftr_k);
            inner_advection!(Shiftz, shiftz, shiftz_r, shiftz_z, 0.0, shiftz_k);

            inner_advection!(Theta, theta, theta_r, theta_z, 0.0, theta_k);
            inner_advection!(Zr, zr, zr_r, zr_z, 0.0, zr_k);
            inner_advection!(Zz, zz, zz_r, zz_z, 0.0, zz_k);

            let [rho, z] = block.position(vertex);
            let r = (rho * rho + z * z).sqrt();

            let index = block.index_from_vertex(vertex);

            macro_rules! advection {
                ($field:ident, $value:ident, $dr:ident, $dz:ident, $target:literal, $k:ident) => {
                    let adv = ($value[index] - $target) + rho * $dr[index] + z * $dz[index];
                    let adv_full = -adv / r;
                    dest.field_mut(Dynamic::$field)[index] = adv_full + $k / (r * r * r);
                };
            }

            advection!(Grr, grr, grr_r, grr_z, 1.0, grr_k);
            advection!(Gzz, gzz, gzz_r, gzz_z, 1.0, gzz_k);
            advection!(Grz, grz, grz_r, grz_z, 0.0, grz_k);
            advection!(S, s, s_r, s_z, 0.0, s_k);

            advection!(Krr, krr, krr_r, krr_z, 0.0, krr_k);
            advection!(Kzz, kzz, kzz_r, kzz_z, 0.0, kzz_k);
            advection!(Krz, krz, krz_r, krz_z, 0.0, krz_k);
            advection!(Y, y, y_r, y_z, 0.0, y_k);

            advection!(Lapse, lapse, lapse_r, lapse_z, 1.0, lapse_k);
            advection!(Shiftr, shiftr, shiftr_r, shiftr_z, 0.0, shiftr_k);
            advection!(Shiftz, shiftz, shiftz_r, shiftz_z, 0.0, shiftz_k);

            advection!(Theta, theta, theta_r, theta_z, 0.0, theta_k);
            advection!(Zr, zr, zr_r, zr_z, 0.0, zr_k);
            advection!(Zz, zz, zz_r, zz_z, 0.0, zz_k);
        }
    }
}

pub struct DynamicOde<'a> {
    driver: &'a mut Driver,
    mesh: &'a Mesh<2>,
}

impl<'a> Ode for DynamicOde<'a> {
    fn dim(&self) -> usize {
        field_count::<Dynamic>() * self.mesh.node_count()
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        self.driver.fill_boundary_system(
            self.mesh,
            &DynamicBoundary,
            SystemSliceMut::from_contiguous(system),
        );
    }

    fn derivative(&mut self, system: &[f64], result: &mut [f64]) {
        let src = SystemSlice::from_contiguous(system);
        let dest = SystemSliceMut::from_contiguous(result);

        self.driver
            .apply_system(self.mesh, &DynamicDerivs, src, dest);
    }
}

pub struct DynamicDissipation;

impl SystemOperator<2> for DynamicDissipation {
    type Label = Dynamic;

    fn apply(
        &self,
        block: Block<2>,
        pool: &MemPool,
        src: SystemSlice<'_, Self::Label>,
        mut dest: SystemSliceMut<'_, Self::Label>,
    ) {
        let node_count = block.node_count();

        macro_rules! compute_dissipation {
            ($field:ident, $value:ident, $dissr:ident, $dissz:ident) => {
                let $value = src.field(Dynamic::$field);
                let $dissr = pool.alloc_scalar(node_count);
                let $dissz = pool.alloc_scalar(node_count);

                block.dissipation::<4>(0, &DynamicBoundary.field(Dynamic::$field), $value, $dissr);
                block.dissipation::<4>(1, &DynamicBoundary.field(Dynamic::$field), $value, $dissz);
            };
        }

        compute_dissipation!(Grr, grr, grr_dr, grr_dz);
        compute_dissipation!(Grz, grz, grz_dr, grz_dz);
        compute_dissipation!(Gzz, gzz, gzz_dr, gzz_dz);
        compute_dissipation!(S, s, s_dr, s_dz);

        compute_dissipation!(Krr, krr, krr_dr, krr_dz);
        compute_dissipation!(Krz, krz, krz_dr, krz_dz);
        compute_dissipation!(Kzz, kzz, kzz_dr, kzz_dz);
        compute_dissipation!(Y, y, y_dr, y_dz);

        compute_dissipation!(Lapse, lapse, lapse_dr, lapse_dz);
        compute_dissipation!(Shiftr, shiftr, shiftr_dr, shiftr_dz);
        compute_dissipation!(Shiftz, shiftz, shiftz_dr, shiftz_dz);

        compute_dissipation!(Theta, theta, theta_dr, theta_dz);
        compute_dissipation!(Zr, zr, zr_dr, zr_dz);
        compute_dissipation!(Zz, zz, zz_dr, zz_dz);

        for vertex in block.iter() {
            let index = block.index_from_vertex(vertex);

            macro_rules! dissipation {
                ($field:ident, $value:ident, $dissr:ident, $dissz:ident) => {
                    dest.field_mut(Dynamic::$field)[index] = $dissr[index] + $dissz[index];
                };
            }

            dissipation!(Grr, grr, grr_dr, grr_dz);
            dissipation!(Grz, grz, grz_dr, grz_dz);
            dissipation!(Gzz, gzz, gzz_dr, gzz_dz);
            dissipation!(S, s, s_dr, s_dz);

            dissipation!(Krr, krr, krr_dr, krr_dz);
            dissipation!(Krz, krz, krz_dr, krz_dz);
            dissipation!(Kzz, kzz, kzz_dr, kzz_dz);
            dissipation!(Y, y, y_dr, y_dz);

            dissipation!(Lapse, lapse, lapse_dr, lapse_dz);
            dissipation!(Shiftr, shiftr, shiftr_dr, shiftr_dz);
            dissipation!(Shiftz, shiftz, shiftz_dr, shiftz_dz);

            dissipation!(Theta, theta, theta_dr, theta_dz);
            dissipation!(Zr, zr, zr_dr, zr_dz);
            dissipation!(Zz, zz, zz_dr, zz_dz);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model: Model<2> = {
        let mut contents = String::new();
        let mut file = File::open("output/idbrill.dat")?;
        file.read_to_string(&mut contents)?;

        ron::from_str(&contents)?
    };

    // Get driver
    let mut driver = Driver::new();

    // Grid
    let mesh = model.mesh().clone();
    // Initial Data
    let initial = model.read_system::<InitialData>().unwrap();

    // Setup dynamic variables
    let mut dynamic = SystemVec::new(mesh.node_count());

    // Metric
    dynamic
        .field_mut(Dynamic::Grr)
        .copy_from_slice(initial.field(InitialData::Conformal));
    dynamic
        .field_mut(Dynamic::Gzz)
        .copy_from_slice(initial.field(InitialData::Conformal));
    dynamic.field_mut(Dynamic::Grz).fill(0.0);
    // S
    dynamic
        .field_mut(Dynamic::S)
        .copy_from_slice(initial.field(InitialData::Seed));
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
    driver.fill_boundary_system(&mesh, &DynamicBoundary, dynamic.as_mut_slice());

    // Begin integration

    let h = CFL * mesh.minimum_spacing();
    println!("Spacing {}", mesh.minimum_spacing());
    println!("Step Size {}", h);

    let mut data = dynamic.as_slice().to_contigious().into_boxed_slice();
    let mut update = vec![0.0; data.len()].into_boxed_slice();
    let mut dissipation = vec![0.0; data.len()].into_boxed_slice();

    let mut integrator = Rk4::new();

    for i in 0..STEPS {
        let norm = driver.norm_system::<2, Dynamic>(&mesh, SystemSlice::from_contiguous(&data));
        println!("Step {i}, Time {:.5} Norm {:.5e}", i as f64 * h, norm);
        // Output current system to disk
        let mut model = Model::new(mesh.clone());
        model.attach_system::<Dynamic>(SystemSlice::from_contiguous(&data));
        model.export_vtk(
            format!("evbrill").as_str(),
            PathBuf::from(format!("output/evbrill{i}.vtu")),
        )?;

        // Fill ghost nodes of system
        driver.fill_boundary_system(
            &mesh,
            &DynamicBoundary,
            SystemSliceMut::from_contiguous(&mut data),
        );

        // Compute step
        integrator.step(
            h,
            &mut DynamicOde {
                driver: &mut driver,
                mesh: &mesh,
            },
            &data,
            &mut update,
        );

        // Compute dissipation
        driver.apply_system(
            &mesh,
            &DynamicDissipation,
            SystemSlice::from_contiguous(&data),
            SystemSliceMut::from_contiguous(&mut dissipation),
        );

        // Add everything together
        for i in 0..data.len() {
            data[i] += update[i] + 0.5 * dissipation[i];
        }
    }

    Ok(())
}
