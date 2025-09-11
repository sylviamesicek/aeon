use crate::{
    image::{ImageMut, ImageRef, ImageShared},
    kernel::SystemBoundaryConds,
    mesh::{Function, FunctionBorrowMut, Mesh},
};
use datasize::DataSize;
use rayon::iter::{ParallelBridge, ParallelIterator};
use reborrow::{Reborrow, ReborrowMut};

/// Method to be used for numerical intergration of ODE.
#[derive(Clone, Copy, Debug, Default, DataSize)]
pub enum Method {
    // First order accurate Euler integration
    #[default]
    ForwardEuler,
    RK4,
    RK4KO6(f64),
}

#[derive(Clone, Debug, DataSize)]
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
    pub fn step<const N: usize, C: SystemBoundaryConds<N> + Sync, F: Function<N> + Sync>(
        &mut self,
        mesh: &mut Mesh<N>,
        order: usize,
        conditions: C,
        mut deriv: F,
        h: f64,
        mut result: ImageMut,
    ) -> Result<(), F::Error>
    where
        F::Error: Send,
    {
        assert!(mesh.num_nodes() == result.num_nodes());
        let num_channels = result.num_channels();

        // Number of degrees of freedom required to store one system.
        let dimension = num_channels * result.num_nodes();
        self.tmp.clear();

        match self.method {
            Method::ForwardEuler => {
                // Resize temporary vector to appropriate size
                self.tmp.resize(dimension, 0.0);
                // Retrieve reference to tmp `SystemVec`.
                let mut tmp = ImageMut::from_storage(&mut self.tmp, num_channels);

                // First step
                Self::copy_from(tmp.rb_mut(), result.rb());
                mesh.apply(order, conditions.clone(), deriv, tmp.rb_mut())?;
                Self::fused_multiply_add_assign(result, h, tmp.rb());

                Ok(())
            }
            Method::RK4 | Method::RK4KO6(..) => {
                self.tmp.resize(2 * dimension, 0.0);

                let (tmp1, tmp2) = self.tmp.split_at_mut(dimension);
                let mut tmp = ImageMut::from_storage(tmp1, num_channels);
                let mut update = ImageMut::from_storage(tmp2, num_channels);

                mesh.fill_boundary(order, conditions.clone(), result.rb_mut());

                // K1
                Self::copy_from(tmp.rb_mut(), result.rb());
                deriv.preprocess(mesh, tmp.rb_mut())?;
                mesh.apply(
                    order,
                    conditions.clone(),
                    FunctionBorrowMut(&mut deriv),
                    tmp.rb_mut(),
                )?;
                Self::fused_multiply_add_assign(update.rb_mut(), 1. / 6., tmp.rb());

                // K2
                Self::fused_multiply_add_dest(tmp.rb_mut(), result.rb(), h / 2.0);
                mesh.fill_boundary(order, conditions.clone(), tmp.rb_mut());
                deriv.preprocess(mesh, tmp.rb_mut())?;
                mesh.apply(
                    order,
                    conditions.clone(),
                    FunctionBorrowMut(&mut deriv),
                    tmp.rb_mut(),
                )?;
                Self::fused_multiply_add_assign(update.rb_mut(), 1. / 3., tmp.rb());

                // K3
                Self::fused_multiply_add_dest(tmp.rb_mut(), result.rb(), h / 2.0);
                mesh.fill_boundary(order, conditions.clone(), tmp.rb_mut());
                deriv.preprocess(mesh, tmp.rb_mut())?;
                mesh.apply(
                    order,
                    conditions.clone(),
                    FunctionBorrowMut(&mut deriv),
                    tmp.rb_mut(),
                )?;
                Self::fused_multiply_add_assign(update.rb_mut(), 1. / 3., tmp.rb());

                // K4
                Self::fused_multiply_add_dest(tmp.rb_mut(), result.rb(), h);
                mesh.fill_boundary(order, conditions.clone(), tmp.rb_mut());
                deriv.preprocess(mesh, tmp.rb_mut())?;
                mesh.apply(
                    order,
                    conditions.clone(),
                    FunctionBorrowMut(&mut deriv),
                    tmp.rb_mut(),
                )?;
                Self::fused_multiply_add_assign(update.rb_mut(), 1. / 6., tmp.rb());

                // Sum everything
                Self::fused_multiply_add_assign(result.rb_mut(), h, update.rb());

                if let Method::RK4KO6(diss) = self.method {
                    mesh.fill_boundary_to_extent(order, 3, conditions.clone(), result.rb_mut());
                    deriv.preprocess(mesh, result.rb_mut())?;
                    mesh.dissipation::<6>(diss, result.rb_mut());
                }

                Ok(())
            }
        }
    }

    fn copy_from(dest: ImageMut, source: ImageRef) {
        let shared: ImageShared = dest.into();
        source.channels().par_bridge().for_each(|field| {
            unsafe { shared.channel_mut(field) }.copy_from_slice(source.channel(field))
        });
    }

    /// Performs operation `dest = dest + h * b`
    fn fused_multiply_add_assign(dest: ImageMut, h: f64, b: ImageRef) {
        let shared: ImageShared = dest.into();
        b.channels().par_bridge().for_each(|field| {
            let dest = unsafe { shared.channel_mut(field) };
            let src = b.channel(field);

            dest.iter_mut().zip(src).for_each(|(a, b)| *a += h * b);
        });
    }

    /// Performs operation `dest = a + h * dest`
    fn fused_multiply_add_dest(dest: ImageMut, a: ImageRef, h: f64) {
        let shared: ImageShared = dest.into();
        a.channels().par_bridge().for_each(|field| {
            let dest = unsafe { shared.channel_mut(field) };
            let a = a.channel(field);
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
