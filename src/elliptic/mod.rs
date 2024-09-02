use std::marker::PhantomData;

use reborrow::{Reborrow, ReborrowMut};

use crate::{
    fd::{Boundary, Conditions, Engine, Kernels, Mesh, Operator, PairConditions},
    ode::{Ode, Rk4},
    prelude::Face,
    system::{field_count, SystemFields, SystemLabel, SystemSlice, SystemSliceMut, SystemValue},
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

    pub fn solve<const N: usize, K: Kernels + Sync, O: Operator<N, System = Label> + Sync>(
        &mut self,
        mesh: &mut Mesh<N>,
        order: K,
        operator: O,
        context: SystemSlice<'_, O::Context>,
        mut system: SystemSliceMut<'_, Label>,
    ) where
        O::Boundary: Sync,
        O::SystemConditions: Sync,
        O::ContextConditions: Sync,
    {
        // Total number of degreees of freedom in the whole system
        let dimension = field_count::<Label>() * system.len();

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

        let spacing: f64 = mesh.min_spacing();
        let step = self.cfl * spacing;

        for index in 0..self.max_steps {
            {
                mesh.fill_boundary(
                    order,
                    operator.boundary(),
                    operator.system_conditions(),
                    SystemSliceMut::from_contiguous(&mut data[..dimension]),
                );
                mesh.apply(
                    order,
                    operator.clone(),
                    SystemSlice::from_contiguous(&mut data[..dimension]),
                    context.rb(),
                    system.rb_mut(),
                );

                operator.callback(
                    mesh,
                    SystemSlice::from_contiguous(&mut data[..dimension]),
                    context.rb(),
                    index,
                );
            }

            let norm = mesh.norm(system.rb());

            log::trace!(
                "Time {:.5}/{:.5} Norm {:.5e}",
                index as f64 * step,
                self.max_steps as f64 * step,
                norm
            );

            if norm <= self.tolerance {
                log::trace!(
                    "Hyperbolic Relaxation converged with error {:.5e}",
                    self.tolerance
                );
                break;
            }

            let mut ode = FictitiousOde {
                mesh,
                dimension,
                dampening: self.dampening,
                order: order.clone(),
                operator: operator.clone(),
                context: context.rb(),
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

/// Wraps the du/dt = v - u * η operation.
#[derive(Clone)]
struct UOperator<const N: usize, O> {
    dampening: f64,
    operator: O,
}

impl<const N: usize, O: Operator<N>> Operator<N> for UOperator<N, O> {
    type System = O::System;
    type Context = O::System;

    type Boundary = O::Boundary;
    fn boundary(&self) -> Self::Boundary {
        self.operator.boundary()
    }

    type SystemConditions = O::SystemConditions;
    fn system_conditions(&self) -> Self::SystemConditions {
        self.operator.system_conditions()
    }

    type ContextConditions = O::SystemConditions;
    fn context_conditions(&self) -> Self::ContextConditions {
        self.operator.system_conditions()
    }

    fn apply(
        &self,
        engine: &impl Engine<N>,
        input: SystemFields<'_, S>,
        context: SystemFields<'_, S>,
    ) -> SystemValue<S> {
        SystemValue::from_fn(|field: S| {
            let u = engine.value(input.field(field.clone()));
            let v = engine.value(context.field(field.clone()));

            v - u * self.dampening
        })
    }
}

#[derive(Clone)]
struct VOperator<const N: usize, O> {
    dampening: f64,
    inner: O,
}

impl<const N: usize, O: Operator<N>> Operator<N> for VOperator<N, O> {
    type System = O::System;
    type Context = O::Context;

    type Boundary = VBoundary<O::Boundary>;

    fn boundary(&self) -> Self::Boundary {
        VBoundary {
            dampening: self.dampening,
            inner: self.inner.boundary(),
        }
    }

    fn apply(
        &self,
        engine: &impl Engine<N>,
        input: SystemFields<'_, Self::System>,
        context: SystemFields<'_, Self::Context>,
    ) -> SystemValue<Self::System> {
        self.inner.apply(engine, input, context)
    }
}

/// Provides boundary conditions for v. All strong boundary condition commute with the time derivative
/// and radiative boundary conditions simply need to be multipled by the dampening.
#[derive(Clone)]
struct VBoundary<I> {
    dampening: f64,
    inner: I,
}

impl<const N: usize, I: Boundary<N>> Boundary<N> for VBoundary<I> {
    fn kind(&self, face: Face<N>) -> crate::fd::BoundaryKind {
        self.inner.kind(face)
    }
}

impl<const N: usize, I: Conditions<N>> Conditions<N> for VBoundary<I> {
    type System = I::System;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        self.inner.parity(field, face)
    }

    fn radiative(&self, field: Self::System, position: [f64; N]) -> f64 {
        self.dampening * self.inner.radiative(field, position)
    }
}

struct FictitiousOde<'a, const N: usize, Or: Kernels, O: Operator<N>> {
    mesh: &'a mut Mesh<N>,
    dimension: usize,
    dampening: f64,

    order: Or,
    operator: O,

    context: SystemSlice<'a, O::Context>,
}

impl<'a, const N: usize, Or: Kernels, O> Ode for FictitiousOde<'a, N, Or, O>
where
    O: Operator<N> + Sync,
    O::Boundary: Conditions<N, System = O::System> + Kernels + Sync,
{
    fn dim(&self) -> usize {
        2 * self.dimension
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        let (u, v) = system.split_at_mut(self.dimension);

        self.mesh.fill_boundary(
            self.order.clone(),
            self.operator.boundary(),
            SystemSliceMut::from_contiguous(u),
        );

        self.mesh.fill_boundary(
            self.order.clone(),
            VBoundary {
                inner: self.operator.boundary(),
                dampening: self.dampening,
            },
            SystemSliceMut::from_contiguous(v),
        );
    }

    fn derivative(&mut self, system: &[f64], result: &mut [f64]) {
        let (udata, vdata) = system.split_at(self.dimension);

        let u = SystemSlice::from_contiguous(udata);
        let v = SystemSlice::from_contiguous(vdata);

        let (udata, vdata) = result.split_at_mut(self.dimension);

        let mut dudt = SystemSliceMut::from_contiguous(udata);
        let mut dvdt = SystemSliceMut::from_contiguous(vdata);

        // Compute derivatives

        // Find du/dt from the definition v = du/dt + η u
        self.mesh.apply(
            UOperator {
                dampening: self.dampening,
                boundary: self.operator.boundary(),
                _marker: PhantomData,
            },
            u.rb(),
            v.rb(),
            dudt.rb_mut(),
        );

        // dv/dt = Lu
        self.mesh.apply(
            self.operator.clone(),
            u.rb(),
            self.context.rb(),
            dvdt.rb_mut(),
        );

        // Apply Outer boundary conditions

        self.mesh
            .weak_boundary(self.order.clone(), self.operator.boundary(), u, dudt);
        self.mesh.weak_boundary(
            self.order.clone(),
            VBoundary {
                dampening: self.dampening,
                inner: self.operator.boundary(),
            },
            v,
            dvdt,
        );
    }
}
