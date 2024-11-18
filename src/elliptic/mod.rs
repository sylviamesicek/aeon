use aeon_basis::{Boundary, Kernels, RadiativeParams};
use reborrow::{Reborrow, ReborrowMut};
use std::marker::PhantomData;

use crate::{
    fd::{Conditions, Engine, Mesh, Operator},
    ode::{Ode, Rk4},
    prelude::Face,
    system::{field_count, Pair, SystemLabel, SystemSlice, SystemSliceMut, SystemValue},
};
pub struct HyperRelaxSolver<Label: SystemLabel> {
    /// Error tolerance (relaxation stops once error goes below this value).
    pub tolerance: f64,
    /// Maximum number of relaxation steps to perform
    pub max_steps: usize,
    pub dampening: f64,
    pub cfl: f64,
    integrator: Rk4,
    _marker: PhantomData<Label>,
}

impl<Label: SystemLabel> Default for HyperRelaxSolver<Label> {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn solve<
        const N: usize,
        K: Kernels + Sync,
        B: Boundary<N> + Sync,
        C: Conditions<N, System = Label> + Sync,
        O: Operator<N, System = Label> + Sync,
    >(
        &mut self,
        mesh: &mut Mesh<N>,
        order: K,
        boundary: B,
        conditions: C,
        operator: O,
        context: SystemSlice<'_, O::Context>,
        mut system: SystemSliceMut<'_, Label>,
    ) {
        // Total number of degreees of freedom in the whole system
        let dimension = field_count::<Label>() * system.len();

        // Allocate storage
        let mut data = vec![0.0; 2 * dimension].into_boxed_slice();
        let mut update = vec![0.0; 2 * dimension].into_boxed_slice();

        // Fill initial guess
        {
            let (u, v) = data.split_at_mut(dimension);
            let mut usys = SystemSliceMut::<O::System>::from_contiguous(u);
            let mut vsys = SystemSliceMut::<O::System>::from_contiguous(v);

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

        let min_spacing: f64 = mesh.min_spacing();
        let time_step = self.cfl * min_spacing;

        for index in 0..self.max_steps {
            {
                mesh.fill_boundary(
                    order,
                    boundary.clone(),
                    conditions.clone(),
                    SystemSliceMut::from_contiguous(&mut data[..dimension]),
                );
                mesh.apply(
                    order,
                    boundary.clone(),
                    operator.clone(),
                    SystemSlice::from_contiguous(&data[..dimension]),
                    context.rb(),
                    system.rb_mut(),
                );

                operator.callback(
                    mesh,
                    SystemSlice::from_contiguous(&data[..dimension]),
                    context.rb(),
                    index,
                );
            }

            let norm = mesh.l2_norm(system.rb());

            if index % 1000 == 0 {
                log::info!("Relaxed {}k steps, norm: {:.5e}", index / 1000, norm);
            }

            if norm <= self.tolerance {
                log::info!(
                    "Hyperbolic Relaxation converged with error {:.5e} in {} steps.",
                    self.tolerance,
                    index
                );
                break;
            }

            // let max_spacing = (0..mesh.num_blocks())
            //     .map(|block| mesh.block_spacing(block))
            //     .max_by(|a, b| a.total_cmp(b))
            //     .unwrap_or(1.0);

            let mut ode = FictitiousOde {
                mesh,
                dimension,
                dampening: self.dampening,
                order,
                boundary: boundary.clone(),
                conditions: conditions.clone(),
                operator: operator.clone(),
                context: context.rb(),
            };

            // Take step
            self.integrator
                .step(time_step, &mut ode, &data, &mut update);

            for i in 0..data.len() {
                data[i] += update[i];
            }

            if index == self.max_steps - 1 {
                log::error!(
                    "Hyperbolic relaxation failed to converge in {} steps.",
                    self.max_steps
                );
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

#[derive(Clone)]
struct UConditions<C> {
    conditions: C,
}

impl<const N: usize, I: Conditions<N>> Conditions<N> for UConditions<I> {
    type System = I::System;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        self.conditions.parity(field, face)
    }

    fn radiative(&self, field: Self::System, position: [f64; N], spacing: f64) -> RadiativeParams {
        self.conditions.radiative(field, position, spacing)
        // if ADAPTIVE_WAVESPEED {
        //     params.speed *= spacing / self.min_spacing;
        // }
    }
}

/// Wraps the du/dt = v - u * η operation.
#[derive(Clone)]
struct UOperator<const N: usize, Label> {
    dampening: f64,
    _marker: PhantomData<Label>,
}

impl<const N: usize, Label: SystemLabel> Operator<N> for UOperator<N, Label> {
    type System = Label;
    type Context = Label;

    fn apply(
        &self,
        engine: &impl Engine<N, Pair<Self::System, Self::Context>>,
    ) -> SystemValue<Self::System> {
        SystemValue::from_fn(|field: Self::System| {
            let u = engine.value(Pair::Left(field.clone()));
            let v = engine.value(Pair::Right(field.clone()));

            v - u * self.dampening
        })
    }
}

/// Provides boundary conditions for v. All strong boundary condition commute with the time derivative
/// and radiative boundary conditions simply need to be multipled by the dampening.
#[derive(Clone)]
struct VConditions<C> {
    dampening: f64,
    conditions: C,
}

impl<const N: usize, I: Conditions<N>> Conditions<N> for VConditions<I> {
    type System = I::System;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        self.conditions.parity(field, face)
    }

    fn radiative(&self, field: Self::System, position: [f64; N], spacing: f64) -> RadiativeParams {
        let mut params = self.conditions.radiative(field, position, spacing);

        // if ADAPTIVE_WAVESPEED {
        //     params.speed *= spacing / self.min_spacing;
        // }

        params.target *= self.dampening;
        params
    }
}

/// Wraps the c^2 * L{u(x)} operation.
#[derive(Clone)]
struct VOperator<const N: usize, O> {
    operator: O,
}

impl<const N: usize, O: Operator<N>> Operator<N> for VOperator<N, O> {
    type System = O::System;
    type Context = O::Context;

    fn apply(
        &self,
        engine: &impl Engine<N, Pair<Self::System, Self::Context>>,
    ) -> SystemValue<Self::System> {
        // let relax = self.operator.apply(engine);
        // if ADAPTIVE_WAVESPEED {
        //     let speed = engine.spacing() / self.min_spacing;
        //     SystemValue::from_fn(|field| speed * speed * relax.field(field))
        // } else {
        //     relax
        // }

        self.operator.apply(engine)
    }
}

struct FictitiousOde<
    'a,
    const N: usize,
    K: Kernels,
    B: Boundary<N>,
    C: Conditions<N>,
    O: Operator<N>,
> {
    mesh: &'a mut Mesh<N>,
    dimension: usize,
    dampening: f64,

    order: K,
    boundary: B,
    conditions: C,
    operator: O,

    context: SystemSlice<'a, O::Context>,
}

impl<
        'a,
        const N: usize,
        K: Kernels + Sync,
        B: Boundary<N> + Sync,
        C: Conditions<N> + Sync,
        O: Operator<N, System = C::System> + Sync,
    > Ode for FictitiousOde<'a, N, K, B, C, O>
{
    fn dim(&self) -> usize {
        2 * self.dimension
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        let (u, v) = system.split_at_mut(self.dimension);

        self.mesh.fill_boundary(
            self.order,
            self.boundary.clone(),
            UConditions {
                conditions: self.conditions.clone(),
            },
            SystemSliceMut::from_contiguous(u),
        );

        self.mesh.fill_boundary(
            self.order,
            self.boundary.clone(),
            VConditions {
                conditions: self.conditions.clone(),
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
            self.order,
            self.boundary.clone(),
            UOperator {
                dampening: self.dampening,
                _marker: PhantomData,
            },
            u.rb(),
            v.rb(),
            dudt.rb_mut(),
        );

        // dv/dt = c^2 Lu

        self.mesh.apply(
            self.order,
            self.boundary.clone(),
            VOperator {
                operator: self.operator.clone(),
            },
            u.rb(),
            self.context.rb(),
            dvdt.rb_mut(),
        );

        // Apply Outer boundary conditions
        self.mesh.weak_boundary(
            self.order,
            self.boundary.clone(),
            UConditions {
                conditions: self.conditions.clone(),
            },
            u.rb(),
            dudt,
        );
        self.mesh.weak_boundary(
            self.order,
            self.boundary.clone(),
            VConditions {
                dampening: self.dampening,
                conditions: self.conditions.clone(),
            },
            v,
            dvdt,
        );
    }
}
