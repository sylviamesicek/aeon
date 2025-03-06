use crate::{
    kernel::{Kernels, Order},
    mesh::{Function, Mesh},
    system::{System, SystemBoundaryConds, SystemSlice, SystemSliceMut},
};
use rayon::iter::{ParallelBridge, ParallelIterator};
use reborrow::{Reborrow, ReborrowMut};

/// Method to be used for numerical intergration of ODE.
#[derive(Clone, Copy, Debug, Default)]
pub enum Method {
    // First order accurate Euler integration
    #[default]
    ForwardEuler,
    RK4,
    RK4KO6(f64),
}

#[derive(Clone, Debug)]
pub struct Integrator {
    /// Numerical Method
    pub method: Method,
    /// Intermediate data storage.
    tmp: Vec<f64>,
}

impl Integrator {
    /// Constructs a new integrator which is set to use the given method.
    pub fn new(method: Method) -> Self {
        Self {
            method,
            tmp: Vec::new(),
        }
    }

    /// Step the integrator forwards in time.
    pub fn step<
        const N: usize,
        K: Kernels + Sync,
        C: SystemBoundaryConds<N> + Sync,
        F: Function<N, Input = C::System, Output = C::System> + Clone + Sync,
    >(
        &mut self,
        mesh: &mut Mesh<N>,
        order: K,
        conditions: C,
        deriv: F,
        h: f64,
        mut result: SystemSliceMut<C::System>,
    ) where
        C::System: Clone + Sync,
    {
        assert!(mesh.num_nodes() == result.len());

        let system = result.system().clone();

        // Number of degrees of freedom required to store one system.
        let dimension = system.count() * mesh.num_nodes();
        self.tmp.clear();

        match self.method {
            Method::ForwardEuler => {
                // Resize temporary vector to appropriate size
                self.tmp.resize(dimension, 0.0);
                // Retrieve reference to tmp `SystemVec`.
                let mut tmp = SystemSliceMut::from_contiguous(&mut self.tmp, &system);

                // First step
                Self::copy_from(tmp.rb_mut(), result.rb());
                mesh.apply(order, conditions.clone(), deriv, tmp.rb_mut());
                Self::fused_multiply_add_assign(result, h, tmp.rb());
            }
            Method::RK4 | Method::RK4KO6(..) => {
                self.tmp.resize(2 * dimension, 0.0);

                let (tmp1, tmp2) = self.tmp.split_at_mut(dimension);
                let mut tmp = SystemSliceMut::from_contiguous(tmp1, &system);
                let mut update = SystemSliceMut::from_contiguous(tmp2, &system);

                mesh.fill_boundary(order, conditions.clone(), result.rb_mut());

                // K1
                Self::copy_from(tmp.rb_mut(), result.rb());
                deriv.preprocess(mesh, tmp.rb_mut());
                mesh.apply(order, conditions.clone(), deriv.clone(), tmp.rb_mut());
                Self::fused_multiply_add_assign(update.rb_mut(), 1. / 6., tmp.rb());

                // K2
                Self::fused_multiply_add_dest(tmp.rb_mut(), result.rb(), h / 2.0);
                mesh.fill_boundary(order, conditions.clone(), tmp.rb_mut());
                deriv.preprocess(mesh, tmp.rb_mut());
                mesh.apply(order, conditions.clone(), deriv.clone(), tmp.rb_mut());
                Self::fused_multiply_add_assign(update.rb_mut(), 1. / 3., tmp.rb());

                // K3
                Self::fused_multiply_add_dest(tmp.rb_mut(), result.rb(), h / 2.0);
                mesh.fill_boundary(order, conditions.clone(), tmp.rb_mut());
                deriv.preprocess(mesh, tmp.rb_mut());
                mesh.apply(order, conditions.clone(), deriv.clone(), tmp.rb_mut());
                Self::fused_multiply_add_assign(update.rb_mut(), 1. / 3., tmp.rb());

                // K4
                Self::fused_multiply_add_dest(tmp.rb_mut(), result.rb(), h);
                mesh.fill_boundary(order, conditions.clone(), tmp.rb_mut());
                deriv.preprocess(mesh, tmp.rb_mut());
                mesh.apply(order, conditions.clone(), deriv.clone(), tmp.rb_mut());
                Self::fused_multiply_add_assign(update.rb_mut(), 1. / 6., tmp.rb());

                // Sum everything
                Self::fused_multiply_add_assign(result.rb_mut(), h, update.rb());

                if let Method::RK4KO6(diss) = self.method {
                    mesh.fill_boundary_to_extent(order, 3, conditions.clone(), result.rb_mut());
                    deriv.preprocess(mesh, result.rb_mut());
                    mesh.dissipation(Order::<6>, diss, result.rb_mut());
                }
            }
        }
    }

    fn copy_from<S: System + Clone + Sync>(dest: SystemSliceMut<S>, source: SystemSlice<S>) {
        let shared = dest.into_shared();
        source.system().enumerate().par_bridge().for_each(|field| {
            unsafe { shared.field_mut(field) }.copy_from_slice(source.field(field))
        });
    }

    /// Performs operation `dest = dest + h * b`
    fn fused_multiply_add_assign<S: System + Clone + Sync>(
        dest: SystemSliceMut<S>,
        h: f64,
        b: SystemSlice<S>,
    ) {
        let shared = dest.into_shared();
        b.system().enumerate().par_bridge().for_each(|field| {
            let dest = unsafe { shared.field_mut(field) };
            let src = b.field(field);

            dest.iter_mut().zip(src).for_each(|(a, b)| *a += h * b);
        });
    }

    // fn fused_multiply_add<S: System + Clone + Sync>(
    //     dest: SystemSliceMut<S>,
    //     a: SystemSlice<S>,
    //     h: f64,
    //     b: SystemSlice<S>,
    // ) {
    //     let shared = dest.into_shared();
    //     a.system().enumerate().par_bridge().for_each(|field| {
    //         let dest = unsafe { shared.field_mut(field) };
    //         let a = a.field(field);
    //         let b = b.field(field);

    //         dest.iter_mut()
    //             .zip(a.iter().zip(b))
    //             .for_each(|(d, (a, b))| {
    //                 *d = a + h * b;
    //             });
    //     });
    // }

    /// Performs operation `dest = a + h * dest`
    fn fused_multiply_add_dest<S: System + Clone + Sync>(
        dest: SystemSliceMut<S>,
        a: SystemSlice<S>,
        h: f64,
    ) {
        let shared = dest.into_shared();
        a.system().enumerate().par_bridge().for_each(|field| {
            let dest = unsafe { shared.field_mut(field) };
            let a = a.field(field);
            dest.iter_mut().zip(a.iter()).for_each(|(d, a)| {
                *d = a + h * *d;
            });
        });
    }

    // Allocates `len` elements using the intergrator's scratch data.
    pub fn scratch(&mut self, len: usize) -> &mut [f64] {
        self.tmp.clear();
        self.tmp.resize(len, 0.0);
        &mut self.tmp
    }
}
