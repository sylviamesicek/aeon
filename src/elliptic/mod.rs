use std::{marker::PhantomData, path::PathBuf};

use reborrow::{Reborrow, ReborrowMut};

use crate::{
    fd::{Boundary, Conditions, Engine, Mesh, Model, Operator},
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

    pub fn solve<
        const N: usize,
        const ORDER: usize,
        O: Operator<N, System = Label>,
        BC: Boundary<N> + Conditions<N, System = Label>,
    >(
        &mut self,
        mesh: &mut Mesh<N>,
        bc: BC,
        operator: &O,
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
                let u = &mut data[..dimension];

                mesh.order::<ORDER>()
                    .fill_boundary(bc.clone(), SystemSliceMut::from_contiguous(u));
                mesh.order::<ORDER>().apply(
                    bc.clone(),
                    operator.clone(),
                    SystemSlice::from_contiguous(u),
                    context.slice(..),
                    system.slice_mut(..),
                );
            }

            if index % 1 == 0 {
                let mut model = Model::from_mesh(mesh);
                model.attach_field("u", data[..dimension].to_vec());
                model.attach_field("v", data[dimension..].to_vec());

                model
                    .export_vtk(
                        format!("idbrill").as_str(),
                        PathBuf::from(format!("output/idbrill{}.vtu", { index / 1 })),
                    )
                    .unwrap();
            }

            let norm = mesh.norm(system.slice(..));

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
                bc: &bc,
                operator,
                context: context.slice(..),
                _marker: PhantomData::<[usize; ORDER]>,
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
        input: SystemFields<'_, System>,
        context: SystemFields<'_, System>,
    ) -> SystemValue<System> {
        SystemValue::from_fn(|field: System| {
            let u = engine.value(input.field(field.clone()));
            let v = engine.value(context.field(field.clone()));

            v - u * self.dampening
        })
    }
}

/// Provides boundary conditions for v. All strong boundary condition commute with the time derivative
/// and radiative boundary conditions simply need to be multipled by the dampening.
#[derive(Clone)]
struct VBC<I> {
    dampening: f64,
    inner: I,
}

impl<const N: usize, I: Boundary<N>> Boundary<N> for VBC<I> {
    fn kind(&self, face: Face<N>) -> crate::fd::BoundaryKind {
        self.inner.kind(face)
    }
}

impl<const N: usize, I: Conditions<N>> Conditions<N> for VBC<I> {
    type System = I::System;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        self.inner.parity(field, face)
    }

    fn radiative(&self, field: Self::System, position: [f64; N]) -> f64 {
        self.dampening * self.inner.radiative(field, position)
    }
}

struct FictitiousOde<
    'a,
    const N: usize,
    const ORDER: usize,
    BC: Boundary<N> + Conditions<N>,
    O: Operator<N>,
> {
    mesh: &'a mut Mesh<N>,
    dimension: usize,
    dampening: f64,

    bc: &'a BC,
    operator: &'a O,

    context: SystemSlice<'a, O::Context>,

    _marker: PhantomData<[usize; ORDER]>,
}

impl<'a, const N: usize, const ORDER: usize, BC, O> Ode for FictitiousOde<'a, N, ORDER, BC, O>
where
    BC: Boundary<N> + Conditions<N>,

    O: Operator<N, System = BC::System>,
{
    fn dim(&self) -> usize {
        2 * self.dimension
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        let (u, v) = system.split_at_mut(self.dimension);

        self.mesh
            .order::<ORDER>()
            .fill_boundary(self.bc.clone(), SystemSliceMut::from_contiguous(u));

        self.mesh.order::<ORDER>().fill_boundary(
            VBC {
                inner: self.bc.clone(),
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
        self.mesh.order::<ORDER>().apply(
            self.bc.clone(),
            UOperator {
                dampening: self.dampening,
                _marker: PhantomData,
            },
            u.rb(),
            v.rb(),
            dudt.rb_mut(),
        );

        // dv/dt = Lu
        self.mesh.order::<ORDER>().apply(
            self.bc.clone(),
            self.operator.clone(),
            u.rb(),
            self.context.rb(),
            dvdt.rb_mut(),
        );

        // Apply Outer boundary conditions

        self.mesh
            .order::<ORDER>()
            .weak_boundary(self.bc.clone(), u, dudt);

        self.mesh.order::<ORDER>().weak_boundary(
            VBC {
                dampening: self.dampening,
                inner: self.bc.clone(),
            },
            v,
            dvdt,
        );
    }
}
