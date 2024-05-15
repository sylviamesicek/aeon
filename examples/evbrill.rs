use std::path::PathBuf;
use std::{fs::File, io::Read};

use aeon::prelude::*;
use aeon::{array::Array, mesh::field_count};
use aeon_axisymmetry::{hyperbolic, hyperbolic_regular, HyperbolicSystem};

#[derive(Clone)]
pub enum InitialData {
    Conformal,
    Seed,
}

impl SystemLabel for InitialData {
    const NAME: &'static str = "InitialData";
    type FieldLike<T> = [T; 2];
    fn fields() -> Array<Self::FieldLike<Self>> {
        [InitialData::Conformal, InitialData::Seed].into()
    }

    fn field_index(&self) -> usize {
        match self {
            InitialData::Conformal => 0,
            InitialData::Seed => 1,
        }
    }

    fn field_name(&self) -> String {
        match self {
            InitialData::Conformal => "Conformal".to_string(),
            InitialData::Seed => "Seed".to_string(),
        }
    }
}

#[derive(Clone)]
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

impl SystemLabel for Dynamic {
    const NAME: &'static str = "Dynamic";
    type FieldLike<T> = [T; 14];

    fn fields() -> Array<Self::FieldLike<Self>> {
        [
            Dynamic::Grr,
            Dynamic::Grz,
            Dynamic::Gzz,
            Dynamic::S,
            Dynamic::Krr,
            Dynamic::Krz,
            Dynamic::Kzz,
            Dynamic::Y,
            Dynamic::Theta,
            Dynamic::Zr,
            Dynamic::Zz,
            Dynamic::Lapse,
            Dynamic::Shiftr,
            Dynamic::Shiftz,
        ]
        .into()
    }

    fn field_index(&self) -> usize {
        match self {
            Dynamic::Grr => 0,
            Dynamic::Grz => 1,
            Dynamic::Gzz => 2,
            Dynamic::S => 3,
            Dynamic::Krr => 4,
            Dynamic::Krz => 5,
            Dynamic::Kzz => 6,
            Dynamic::Y => 7,
            Dynamic::Theta => 8,
            Dynamic::Zr => 9,
            Dynamic::Zz => 10,
            Dynamic::Lapse => 11,
            Dynamic::Shiftr => 12,
            Dynamic::Shiftz => 13,
        }
    }

    fn field_name(&self) -> String {
        match self {
            Dynamic::Grr => "Grr",
            Dynamic::Grz => "Grz",
            Dynamic::Gzz => "Gzz",
            Dynamic::S => "S",
            Dynamic::Krr => "Krr",
            Dynamic::Krz => "Krz",
            Dynamic::Kzz => "Kzz",
            Dynamic::Y => "Y",
            Dynamic::Theta => "Theta",
            Dynamic::Zr => "Zr",
            Dynamic::Zz => "Zz",
            Dynamic::Lapse => "Lapse",
            Dynamic::Shiftr => "Shiftr",
            Dynamic::Shiftz => "Shiftz",
        }
        .to_string()
    }
}

pub struct ParityBoundary(bool);

impl Boundary for ParityBoundary {
    fn face(&self, face: Face) -> BoundaryCondition {
        if face.axis == 0 && face.side == false {
            BoundaryCondition::Parity(self.0)
        } else if face.axis == 1 && face.side == false {
            BoundaryCondition::Parity(true)
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
        ParityBoundary(match label {
            Dynamic::Grr => true,
            Dynamic::Grz => false,
            Dynamic::Gzz => true,
            Dynamic::S => false,
            Dynamic::Krr => true,
            Dynamic::Krz => false,
            Dynamic::Kzz => true,
            Dynamic::Y => false,
            Dynamic::Theta => true,
            Dynamic::Zr => false,
            Dynamic::Zz => true,
            Dynamic::Lapse => true,
            Dynamic::Shiftr => false,
            Dynamic::Shiftz => true,
        })
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

        // Metric
        let grr = src.field(Dynamic::Grr);
        let grr_r = pool.alloc_scalar(node_count);
        let grr_z = pool.alloc_scalar(node_count);
        let grr_rr = pool.alloc_scalar(node_count);
        let grr_rz = pool.alloc_scalar(node_count);
        let grr_zz = pool.alloc_scalar(node_count);
        let grr_boundary = DynamicBoundary.field(Dynamic::Grr);

        block.derivative::<4>(0, &grr_boundary, grr, grr_r);
        block.derivative::<4>(1, &grr_boundary, grr, grr_z);
        block.second_derivative::<4>(0, &grr_boundary, grr, grr_rr);
        block.second_derivative::<4>(1, &grr_boundary, grr, grr_zz);
        block.derivative::<4>(1, &grr_boundary, grr_r, grr_rz);

        let gzz = src.field(Dynamic::Gzz);
        let gzz_r = pool.alloc_scalar(node_count);
        let gzz_z = pool.alloc_scalar(node_count);
        let gzz_rr = pool.alloc_scalar(node_count);
        let gzz_rz = pool.alloc_scalar(node_count);
        let gzz_zz = pool.alloc_scalar(node_count);
        let gzz_boundary = DynamicBoundary.field(Dynamic::Gzz);

        block.derivative::<4>(0, &gzz_boundary, gzz, gzz_r);
        block.derivative::<4>(1, &gzz_boundary, gzz, gzz_z);
        block.second_derivative::<4>(0, &gzz_boundary, gzz, gzz_rr);
        block.second_derivative::<4>(1, &gzz_boundary, gzz, gzz_zz);
        block.derivative::<4>(1, &gzz_boundary, gzz_r, gzz_rz);

        let grz = src.field(Dynamic::Grz);
        let grz_r = pool.alloc_scalar(node_count);
        let grz_z = pool.alloc_scalar(node_count);
        let grz_rr = pool.alloc_scalar(node_count);
        let grz_rz = pool.alloc_scalar(node_count);
        let grz_zz = pool.alloc_scalar(node_count);
        let grz_boundary = DynamicBoundary.field(Dynamic::Grz);

        block.derivative::<4>(0, &grz_boundary, grz, grz_r);
        block.derivative::<4>(1, &grz_boundary, grz, grz_z);
        block.second_derivative::<4>(0, &grz_boundary, grz, grz_rr);
        block.second_derivative::<4>(1, &grz_boundary, grz, grz_zz);
        block.derivative::<4>(1, &grz_boundary, grz_r, grz_rz);

        // S
        let s = src.field(Dynamic::S);
        let s_r = pool.alloc_scalar(node_count);
        let s_z = pool.alloc_scalar(node_count);
        let s_rr = pool.alloc_scalar(node_count);
        let s_rz = pool.alloc_scalar(node_count);
        let s_zz = pool.alloc_scalar(node_count);
        let s_boundary = DynamicBoundary.field(Dynamic::S);

        block.derivative::<4>(0, &s_boundary, s, s_r);
        block.derivative::<4>(1, &s_boundary, s, s_z);
        block.second_derivative::<4>(0, &s_boundary, s, s_rr);
        block.second_derivative::<4>(1, &s_boundary, s, s_zz);
        block.derivative::<4>(1, &s_boundary, s_r, s_rz);

        // K
        let krr = src.field(Dynamic::Krr);
        let krr_r = pool.alloc_scalar(node_count);
        let krr_z = pool.alloc_scalar(node_count);
        let krr_boundary = DynamicBoundary.field(Dynamic::Krr);

        block.derivative::<4>(0, &krr_boundary, krr, krr_r);
        block.derivative::<4>(1, &krr_boundary, krr, krr_z);

        let kzz = src.field(Dynamic::Kzz);
        let kzz_r = pool.alloc_scalar(node_count);
        let kzz_z = pool.alloc_scalar(node_count);
        let kzz_boundary = DynamicBoundary.field(Dynamic::Kzz);

        block.derivative::<4>(0, &kzz_boundary, kzz, kzz_r);
        block.derivative::<4>(1, &kzz_boundary, kzz, kzz_z);

        let krz = src.field(Dynamic::Krz);
        let krz_r = pool.alloc_scalar(node_count);
        let krz_z = pool.alloc_scalar(node_count);
        let krz_boundary = DynamicBoundary.field(Dynamic::Krz);

        block.derivative::<4>(0, &krz_boundary, krz, krz_r);
        block.derivative::<4>(1, &krz_boundary, krz, krz_z);

        // Y
        let y = src.field(Dynamic::Krz);
        let y_r = pool.alloc_scalar(node_count);
        let y_z = pool.alloc_scalar(node_count);
        let y_boundary = DynamicBoundary.field(Dynamic::Y);

        block.derivative::<4>(0, &y_boundary, y, y_r);
        block.derivative::<4>(1, &y_boundary, y, y_z);

        // Lapse
        let lapse = src.field(Dynamic::Lapse);
        let lapse_r = pool.alloc_scalar(node_count);
        let lapse_z = pool.alloc_scalar(node_count);
        let lapse_rr = pool.alloc_scalar(node_count);
        let lapse_rz = pool.alloc_scalar(node_count);
        let lapse_zz = pool.alloc_scalar(node_count);
        let lapse_boundary = DynamicBoundary.field(Dynamic::Lapse);

        block.derivative::<4>(0, &lapse_boundary, lapse, lapse_r);
        block.derivative::<4>(1, &lapse_boundary, lapse, lapse_z);
        block.second_derivative::<4>(0, &lapse_boundary, lapse, lapse_rr);
        block.second_derivative::<4>(1, &lapse_boundary, lapse, lapse_zz);
        block.derivative::<4>(1, &lapse_boundary, lapse_r, lapse_rz);

        // Shift
        let shiftr = src.field(Dynamic::Shiftr);
        let shiftr_r = pool.alloc_scalar(node_count);
        let shiftr_z = pool.alloc_scalar(node_count);
        let shiftr_boundary = DynamicBoundary.field(Dynamic::Shiftr);

        block.derivative::<4>(0, &shiftr_boundary, shiftr, shiftr_r);
        block.derivative::<4>(1, &shiftr_boundary, shiftr, shiftr_z);

        let shiftz = src.field(Dynamic::Shiftz);
        let shiftz_r = pool.alloc_scalar(node_count);
        let shiftz_z = pool.alloc_scalar(node_count);
        let shiftz_boundary = DynamicBoundary.field(Dynamic::Shiftz);

        block.derivative::<4>(0, &shiftz_boundary, shiftz, shiftz_r);
        block.derivative::<4>(1, &shiftz_boundary, shiftz, shiftz_z);

        // Theta
        let theta = src.field(Dynamic::Theta);
        let theta_r = pool.alloc_scalar(node_count);
        let theta_z = pool.alloc_scalar(node_count);
        let theta_boundary = DynamicBoundary.field(Dynamic::Theta);

        block.derivative::<4>(0, &theta_boundary, theta, theta_r);
        block.derivative::<4>(1, &theta_boundary, theta, theta_z);

        // Z
        let zr = src.field(Dynamic::Zr);
        let zr_r = pool.alloc_scalar(node_count);
        let zr_z = pool.alloc_scalar(node_count);
        let zr_boundary = DynamicBoundary.field(Dynamic::Zr);

        block.derivative::<4>(0, &zr_boundary, zr, zr_r);
        block.derivative::<4>(1, &zr_boundary, zr, zr_z);

        let zz = src.field(Dynamic::Zz);
        let zz_r = pool.alloc_scalar(node_count);
        let zz_z = pool.alloc_scalar(node_count);
        let zz_boundary = DynamicBoundary.field(Dynamic::Zz);

        block.derivative::<4>(0, &zz_boundary, zz, zz_r);
        block.derivative::<4>(1, &zz_boundary, zz, zz_z);

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

        for vertex in block
            .face_plane(Face::positive(0))
            .chain(block.face_plane(Face::positive(1)))
        {
            let [rho, z] = block.position(vertex);
            let r = (rho * rho + z * z).sqrt();

            let index = block.index_from_vertex(vertex);

            macro_rules! advection {
                ($field:ident, $value:ident, $dr:ident, $dz:ident, $target:literal) => {
                    let adv = ($value[index] - $target) + rho * $dr[index] + z * $dz[index];
                    dest.field_mut(Dynamic::$field)[index] = -adv / r;
                };
            }

            advection!(Grr, grr, grr_r, grr_z, 1.0);
            advection!(Gzz, gzz, gzz_r, gzz_z, 1.0);
            advection!(Grz, grz, grz_r, grz_z, 0.0);
            advection!(S, s, s_r, s_z, 0.0);

            advection!(Krr, krr, krr_r, krr_z, 0.0);
            advection!(Kzz, kzz, kzz_r, kzz_z, 0.0);
            advection!(Krz, krz, krz_r, krz_z, 0.0);
            advection!(Y, y, y_r, y_z, 0.0);

            advection!(Lapse, lapse, lapse_r, lapse_z, 1.0);
            advection!(Shiftr, shiftr, shiftr_r, shiftr_z, 0.0);
            advection!(Shiftz, shiftz, shiftz_r, shiftz_z, 0.0);

            advection!(Theta, theta, theta_r, theta_z, 0.0);
            advection!(Zr, zr, zr_r, zr_z, 0.0);
            advection!(Zz, zz, zz_r, zz_z, 0.0);
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
    const STEPS: usize = 1000;
    const CFL: f64 = 0.1;

    let h = CFL * mesh.minimum_spacing();

    let mut data = dynamic.into_contigious().into_boxed_slice();
    let mut update = vec![0.0; data.len()].into_boxed_slice();
    let mut dissipation = vec![0.0; data.len()].into_boxed_slice();

    let mut integrator = Rk4::new();

    for i in 0..STEPS {
        println!("Step {i}");
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
            data[i] += update[i] + dissipation[i];
        }
    }

    Ok(())
}
