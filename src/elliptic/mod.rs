use std::marker::PhantomData;

use crate::{
    fd::{Boundary, Condition, Conditions, Driver, Engine, Mesh, Operator},
    ode::{Ode, Rk4},
    system::{field_count, SystemLabel, SystemSlice, SystemSliceMut, SystemVal},
};

pub struct HyperRelaxSolver<Label: SystemLabel> {
    pub tolerance: f64,
    pub max_steps: usize,
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
            dampening: 1.0,
            cfl: 0.1,
            integrator: Rk4::new(),
            _marker: PhantomData,
        }
    }

    pub fn solve<const N: usize, const ORDER: usize, O: Operator<N, System = Label>>(
        &mut self,
        driver: &mut Driver<N>,
        mesh: &Mesh<N>,
        boundary: &impl Boundary,
        conditions: &impl Conditions<N, System = Label>,
        operator: &O,
        context: SystemSlice<'_, O::Context>,
        mut system: SystemSliceMut<'_, Label>,
    ) {
        // Total number of degreees of freedom in the whole system
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

        // let spacing = 0.1;
        // let step = 0.1;

        for index in 0..self.max_steps {
            {
                let u = &mut data[..dimension];

                driver.fill_boundary::<ORDER, _, _>(
                    mesh,
                    boundary,
                    conditions,
                    SystemSliceMut::from_contiguous(u),
                );
                driver.apply::<ORDER, _>(
                    mesh,
                    boundary,
                    operator,
                    SystemSlice::from_contiguous(u),
                    context.slice(..),
                    system.slice_mut(..),
                );
            }

            let norm = driver.norm(mesh, system.slice(..));

            // log::trace!(
            //     "Time {:.5}/{:.5} Norm {:.5e}",
            //     index as f64 * step,
            //     self.max_steps as f64 * step,
            //     norm
            // );

            if norm <= self.tolerance {
                log::trace!(
                    "Hyperbolic Relaxation converged with error {:.5e}",
                    self.tolerance
                );
                break;
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

/// Wraps the du/dt = v - u * η operation.
struct UOperator<const N: usize, System: SystemLabel> {
    dampening: f64,
    _marker: PhantomData<System>,
}

impl<'a, const N: usize, System: SystemLabel> Operator<N> for UOperator<N, System> {
    type System = System;
    type Context = System;

    fn evaluate(
        &self,
        engine: &impl Engine<N>,
        input: SystemSlice<'_, System>,
        context: SystemSlice<'_, System>,
    ) -> SystemVal<System> {
        SystemVal::from_fn(|field: System| {
            let u = engine.value(input.field(field.clone()));
            let v = engine.value(context.field(field.clone()));

            v - u * self.dampening
        })
    }
}

/// Provides boundary conditions for v. All strong boundary condition commute with the time derivative
/// and radiative boundary conditions simply need to be multipled by the dampening.
struct VConditions<'a, const N: usize, C: Conditions<N>> {
    dampening: f64,
    conditions: &'a C,
}

impl<'a, const N: usize, C: Conditions<N>> Conditions<N> for VConditions<'a, N, C> {
    type System = C::System;
    type Condition = VCondition<'a, N, C>;

    fn field(&self, label: Self::System) -> Self::Condition {
        VCondition {
            dampening: self.dampening,
            conditions: self.conditions,
            label: label,
        }
    }
}

struct VCondition<'a, const N: usize, C: Conditions<N>> {
    dampening: f64,
    conditions: &'a C,
    label: C::System,
}

impl<'a, const N: usize, C: Conditions<N>> Condition<N> for VCondition<'a, N, C> {
    fn parity(&self, face: crate::prelude::Face) -> bool {
        self.conditions.field(self.label.clone()).parity(face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.conditions
            .field(self.label.clone())
            .radiative(position)
            * self.dampening
    }
}

struct FictitiousOde<'a, const N: usize, const ORDER: usize, B, C: Conditions<N>, O: Operator<N>> {
    driver: &'a mut Driver<N>,
    mesh: &'a Mesh<N>,
    dimension: usize,
    dampening: f64,

    boundary: &'a B,
    conditions: &'a C,
    operator: &'a O,

    context: SystemSlice<'a, O::Context>,
}

impl<'a, const N: usize, const ORDER: usize, B, C, O> Ode for FictitiousOde<'a, N, ORDER, B, C, O>
where
    B: Boundary,
    C: Conditions<N>,
    O: Operator<N, System = C::System>,
{
    fn dim(&self) -> usize {
        2 * self.dimension
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        let (u, v) = system.split_at_mut(self.dimension);

        let usystem = SystemSliceMut::from_contiguous(u);
        self.driver.fill_boundary::<ORDER, _, _>(
            self.mesh,
            self.boundary,
            self.conditions,
            usystem,
        );

        let vsystem = SystemSliceMut::from_contiguous(v);
        self.driver.fill_boundary::<ORDER, _, _>(
            self.mesh,
            self.boundary,
            self.conditions,
            vsystem,
        );
    }

    fn derivative(&mut self, system: &[f64], result: &mut [f64]) {
        let (u, v) = system.split_at(self.dimension);

        let usource = SystemSlice::from_contiguous(u);
        let vsource = SystemSlice::from_contiguous(v);

        let (u, v) = result.split_at_mut(self.dimension);

        let udest = SystemSliceMut::from_contiguous(u);
        let vdest = SystemSliceMut::from_contiguous(v);

        // Compute derivatives

        // dv/dt = Lu
        self.driver.apply::<ORDER, _>(
            self.mesh,
            self.boundary,
            self.operator,
            usource.clone(),
            self.context.clone(),
            vdest,
        );

        // Find du/dt from the definition v = du/dt + η u
        self.driver.apply::<ORDER, _>(
            self.mesh,
            self.boundary,
            &UOperator {
                dampening: self.dampening,
                _marker: PhantomData,
            },
            usource.clone(),
            vsource,
            udest,
        );

        // TODO apply weak boundary conditions

        todo!("Weak boundary conditions")

        // // Apply outgoing wave conditions
        // if let OutgoingWave::Sommerfeld(value) = self.outgoing {
        //     for field in O::Label::fields() {
        //         let boundary = self.boundary.field(field.clone());

        //         let vfsource = &vsource.field(field.clone())[range.clone()];
        //         let ufsource = &usource.field(field.clone())[range.clone()];
        //         let vfdest = &mut vdest.field_mut(field.clone())[range.clone()];
        //         let ufdest = &mut udest.field_mut(field.clone())[range.clone()];

        //         let vf_grad: [&mut [f64]; N] =
        //             from_fn(|_| self.driver.pool.alloc_scalar(node_count));

        //         let uf_grad: [&mut [f64]; N] =
        //             from_fn(|_| self.driver.pool.alloc_scalar(node_count));

        //         for axis in 0..N {
        //             match self.outgoing_order {
        //                 OutgoingOrder::Second => {
        //                     block.derivative::<2>(axis, &boundary, vfsource, vf_grad[axis]);
        //                     block.derivative::<2>(axis, &boundary, ufsource, uf_grad[axis]);
        //                 }
        //                 OutgoingOrder::Fourth => {
        //                     block.derivative::<4>(axis, &boundary, vfsource, vf_grad[axis]);
        //                     block.derivative::<4>(axis, &boundary, ufsource, uf_grad[axis]);
        //                 }
        //             }
        //         }

        //         for face in faces::<N>() {
        //             if boundary.face(face) == BoundaryCondition::Free {
        //                 for vertex in block.face_plane(face) {
        //                     let index = block.index_from_vertex(vertex);
        //                     let position = block.position(vertex);
        //                     let r = position.iter().map(|&f| f * f).sum::<f64>().sqrt();

        //                     let mut vadv = vfsource[index] - self.dampening * value;
        //                     let mut uadv = ufsource[index] - value;

        //                     for axis in 0..N {
        //                         vadv += position[axis] * vf_grad[axis][index];
        //                         uadv += position[axis] * uf_grad[axis][index];
        //                     }

        //                     vfdest[index] = -vadv / r;
        //                     ufdest[index] = -uadv / r;
        //                 }
        //             }
        //         }

        //         self.driver.pool.reset();
        //     }
        // }
    }
}
