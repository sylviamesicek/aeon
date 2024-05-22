use std::{array::from_fn, marker::PhantomData};

use crate::{
    common::{Boundary, BoundaryCondition},
    geometry::faces,
    mesh::{
        field_count, BlockExt, Driver, Mesh, SystemBoundary, SystemLabel, SystemOperator,
        SystemSlice, SystemSliceMut,
    },
    ode::{Ode, Rk4},
};

#[derive(Debug, Clone)]
pub enum OutgoingWave {
    Free,
    Sommerfeld(f64),
}

#[derive(Debug, Clone)]
pub enum OutgoingOrder {
    Second,
    Fourth,
}

#[derive(Debug, Clone)]
pub struct HyperRelaxSolver<Label: SystemLabel> {
    pub tolerance: f64,
    pub max_steps: usize,
    pub outgoing: OutgoingWave,
    pub outgoing_order: OutgoingOrder,
    pub dampening: f64,
    pub cfl: f64,
    integrator: Rk4,
    _marker: PhantomData<Label>,
}

impl<Label: SystemLabel> HyperRelaxSolver<Label> {
    pub fn new() -> Self {
        Self {
            tolerance: 10e-5,
            max_steps: 100000,
            outgoing: OutgoingWave::Free,
            outgoing_order: OutgoingOrder::Second,
            dampening: 1.0,
            cfl: 0.1,
            integrator: Rk4::new(),
            _marker: PhantomData,
        }
    }

    pub fn solve<
        const N: usize,
        O: SystemOperator<N, Label = Label>,
        B: SystemBoundary<Label = Label>,
    >(
        &mut self,
        driver: &mut Driver,
        mesh: &Mesh<N>,
        operator: &O,
        boundary: &B,
        mut system: SystemSliceMut<'_, Label>,
    ) {
        // Total number of degrees of freedom in the whole system.
        let dimension = field_count::<Label>() * system.node_count();

        // Allocate storage
        let mut data = vec![0.0; 2 * dimension].into_boxed_slice();
        let mut update = vec![0.0; 2 * dimension].into_boxed_slice();

        // Fill initial guess
        {
            let (u, v) = data.split_at_mut(dimension);
            let mut usys = SystemSliceMut::from_contiguous(u);
            let mut vsys = SystemSliceMut::from_contiguous(v);

            for field in Label::fields() {
                usys.field_mut(field.clone())
                    .copy_from_slice(system.field(field.clone()))
            }

            // Let us assume that du/dt is initially zero
            for field in Label::fields() {
                vsys.field_mut(field.clone())
                    .copy_from_slice(system.field(field.clone()));

                vsys.field_mut(field.clone())
                    .iter_mut()
                    .for_each(|f| *f *= self.dampening);
            }
        }

        let spacing = mesh.minimum_spacing();
        let step = self.cfl * spacing;

        for index in 0..self.max_steps {
            {
                let u = &mut data[..dimension];

                driver.fill_boundary_system(mesh, boundary, SystemSliceMut::from_contiguous(u));
                driver.apply_system(
                    mesh,
                    operator,
                    SystemSlice::from_contiguous(u),
                    system.slice_mut(..),
                );
            }

            let norm = driver.norm_system(mesh, system.slice(..));

            log::trace!(
                "Time {:.5}/{:.5} Norm {:.5e}",
                index as f64 * step,
                self.max_steps as f64 * step,
                norm
            );

            let mut ode = FictitiousOde {
                driver,
                mesh,
                operator,
                boundary,
                dimension,
                outgoing: self.outgoing.clone(),
                outgoing_order: self.outgoing_order.clone(),
                dampening: self.dampening,
            };

            // Take step
            self.integrator.step(step, &mut ode, &data, &mut update);

            for i in 0..data.len() {
                data[i] += update[i];
            }
        }

        // Copy solution back to system vector
        {
            let (u, _v) = data.split_at_mut(dimension);
            let usys = SystemSlice::from_contiguous(u);

            for field in Label::fields() {
                system
                    .field_mut(field.clone())
                    .clone_from_slice(usys.field(field.clone()))
            }
        }
    }
}

impl<Label: SystemLabel> Default for HyperRelaxSolver<Label> {
    fn default() -> Self {
        Self::new()
    }
}

struct FictitiousOde<'a, const N: usize, O, B> {
    driver: &'a mut Driver,
    mesh: &'a Mesh<N>,
    operator: &'a O,
    boundary: &'a B,
    dimension: usize,
    outgoing: OutgoingWave,
    outgoing_order: OutgoingOrder,
    dampening: f64,
}

impl<'a, const N: usize, O, B> Ode for FictitiousOde<'a, N, O, B>
where
    O: SystemOperator<N>,
    B: SystemBoundary<Label = O::Label>,
{
    fn dim(&self) -> usize {
        2 * self.dimension
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        let (u, v) = system.split_at_mut(self.dimension);

        let usystem = SystemSliceMut::from_contiguous(u);
        self.driver
            .fill_boundary_system(self.mesh, self.boundary, usystem);

        // All the strong boundary conditions commute with the time operator and only set u to 0, so
        // this is valid.
        let vsystem = SystemSliceMut::from_contiguous(v);
        self.driver
            .fill_boundary_system(self.mesh, self.boundary, vsystem);
    }

    fn derivative(&mut self, system: &[f64], result: &mut [f64]) {
        let (u, v) = system.split_at(self.dimension);

        let usource = SystemSlice::from_contiguous(u);
        let vsource = SystemSlice::from_contiguous(v);

        let (u, v) = result.split_at_mut(self.dimension);

        let mut udest = SystemSliceMut::from_contiguous(u);
        let mut vdest = SystemSliceMut::from_contiguous(v);

        let block = self.mesh.base_block();
        let range = block.local_from_global();
        let node_count = block.node_count();

        // Apply operator to find dv/dt
        self.operator.apply(
            block.clone(),
            &self.driver.pool,
            usource.slice(range.clone()),
            vdest.slice_mut(range.clone()),
        );
        self.driver.pool.reset();

        // Find du/dt from the definition v = du/dt + η u
        for field in O::Label::fields() {
            let vfsource = &vsource.field(field.clone())[range.clone()];
            let ufsource = &usource.field(field.clone())[range.clone()];
            let ufdest = &mut udest.field_mut(field.clone())[range.clone()];

            for vertex in block.iter() {
                let index = block.index_from_vertex(vertex);
                ufdest[index] = vfsource[index] - self.dampening * ufsource[index];
            }
        }

        // Apply outgoing wave conditions
        if let OutgoingWave::Sommerfeld(value) = self.outgoing {
            for field in O::Label::fields() {
                let boundary = self.boundary.field(field.clone());

                let vfsource = &vsource.field(field.clone())[range.clone()];
                let ufsource = &usource.field(field.clone())[range.clone()];
                let vfdest = &mut vdest.field_mut(field.clone())[range.clone()];
                let ufdest = &mut udest.field_mut(field.clone())[range.clone()];

                let vf_grad: [&mut [f64]; N] =
                    from_fn(|_| self.driver.pool.alloc_scalar(node_count));

                let uf_grad: [&mut [f64]; N] =
                    from_fn(|_| self.driver.pool.alloc_scalar(node_count));

                for axis in 0..N {
                    match self.outgoing_order {
                        OutgoingOrder::Second => {
                            block.derivative::<2>(axis, &boundary, vfsource, vf_grad[axis]);
                            block.derivative::<2>(axis, &boundary, ufsource, uf_grad[axis]);
                        }
                        OutgoingOrder::Fourth => {
                            block.derivative::<4>(axis, &boundary, vfsource, vf_grad[axis]);
                            block.derivative::<4>(axis, &boundary, ufsource, uf_grad[axis]);
                        }
                    }
                }

                for face in faces::<N>() {
                    if boundary.face(face) == BoundaryCondition::Free {
                        for vertex in block.face_plane(face) {
                            let index = block.index_from_vertex(vertex);
                            let position = block.position(vertex);
                            let r = position.iter().map(|&f| f * f).sum::<f64>().sqrt();

                            let mut vadv = vfsource[index] - self.dampening * value;
                            let mut uadv = ufsource[index] - value;

                            for axis in 0..N {
                                vadv += position[axis] * vf_grad[axis][index];
                                uadv += position[axis] * uf_grad[axis][index];
                            }

                            vfdest[index] = -vadv / r;
                            ufdest[index] = -uadv / r;
                        }
                    }
                }

                self.driver.pool.reset();
            }
        }
    }
}
