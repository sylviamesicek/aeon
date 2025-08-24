use std::convert::Infallible;
use std::{array, ops::Range};

use crate::geometry::{BlockId, Face, FaceMask, IndexSpace};
use crate::image::ImageShared;
use crate::kernel::{is_boundary_compatible, Derivative, Dissipation, Kernel, SecondDerivative, SystemBoundaryConds};
use crate::{
    kernel::{
        BoundaryConds as _, BoundaryKind, Hessian, NodeSpace, node_from_vertex, vertex_from_node,
    },
};
use reborrow::{Reborrow, ReborrowMut as _};

use crate::{
    mesh::{Engine, Function, Projection},
    shared::SharedSlice,
    image::{ImageRef, ImageMut},
};

use super::{Mesh, MeshStore};

/// A finite difference engine of a given order, but potentially bordering a free boundary.
struct FdEngine<'store, const N: usize, const ORDER: usize> {
    space: NodeSpace<N>,
    store: &'store MeshStore,
    range: Range<usize>,
}

impl<'store, const N: usize, const ORDER: usize> FdEngine<'store, N, ORDER> {
    fn evaluate_axis(
        &self,
        field: &[f64],
        axis: usize,
        kernel: impl Kernel,
        vertex: [usize; N],
    ) -> f64 {
        self.space
            .evaluate_axis(kernel, node_from_vertex(vertex), field, axis)
    }
}

impl<'store, const N: usize, const ORDER: usize> Engine<N> for FdEngine<'store, N, ORDER> {
    fn space(&self) -> &NodeSpace<N> {
        &self.space
    }

    fn node_range(&self) -> Range<usize> {
        self.range.clone()
    }

    fn alloc<T: Default>(&self, len: usize) -> &mut [T] {
        self.store.scratch(len)
    }

    fn value(&self, field: &[f64], vertex: [usize; N]) -> f64 {
        let index = self.space.index_from_vertex(vertex);
        field[index]
    }

    fn derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate_axis(field, axis, Derivative::<ORDER>, vertex)
    }

    fn second_derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate_axis(field, axis, SecondDerivative::<ORDER>, vertex)
    }

    fn mixed_derivative(&self, field: &[f64], i: usize, j: usize, vertex: [usize; N]) -> f64 {
        self.space
            .evaluate(Hessian::<ORDER>::new(i, j), node_from_vertex(vertex), field)
    }

    fn dissipation(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate_axis(field, axis, Dissipation::<ORDER>, vertex)
    }
}

/// A finite difference engine that only every relies on interior support (and can thus use better optimized stencils).
struct FdIntEngine<'store, const N: usize, const ORDER: usize> {
    space: NodeSpace<N>,
    store: &'store MeshStore,
    range: Range<usize>,
}

impl<'store, const N: usize, const ORDER: usize> FdIntEngine<'store, N, ORDER> {
    fn evaluate(&self, field: &[f64], axis: usize, kernel: impl Kernel, vertex: [usize; N]) -> f64 {
        self.space
            .evaluate_axis_interior(kernel, node_from_vertex(vertex), field, axis)
    }
}

impl<'store, const N: usize, const ORDER: usize> Engine<N> for FdIntEngine<'store, N, ORDER> {
    fn space(&self) -> &NodeSpace<N> {
        &self.space
    }

    fn node_range(&self) -> Range<usize> {
        self.range.clone()
    }

    fn alloc<T: Default>(&self, len: usize) -> &mut [T] {
        self.store.scratch(len)
    }

    fn value(&self, field: &[f64], vertex: [usize; N]) -> f64 {
        let index = self.space.index_from_vertex(vertex);
        field[index]
    }

    fn derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate(field, axis, Derivative::<ORDER>, vertex)
    }

    fn second_derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate(field, axis, SecondDerivative::<ORDER>, vertex)
    }

    fn mixed_derivative(&self, field: &[f64], i: usize, j: usize, vertex: [usize; N]) -> f64 {
        self.space
            .evaluate_interior(Hessian::<ORDER>::new(i, j), node_from_vertex(vertex), field)
    }

    fn dissipation(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate(field, axis, Dissipation::<ORDER>, vertex)
    }
}

/// Transforms a projection into a function.
#[derive(Clone)]
struct ProjectionAsFunction<P>(P);

impl<const N: usize, P: Projection<N>> Function<N> for ProjectionAsFunction<P> {
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<N>,
        _input: ImageRef,
        mut output: ImageMut,
    ) -> Result<(), Infallible> {
        let dest = output.channel_mut(0);

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            dest[index] = self.0.project(engine.position(vertex))
        }

        Ok(())
    }
}

impl<const N: usize> Mesh<N> {
    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn evaluate<P: Function<N> + Sync>(
        &mut self,
        order: usize,
        function: P,
        source: ImageRef,
        dest: ImageMut,
    ) -> Result<(), P::Error>
    where
        P::Error: Send,
    {
        assert!(dest.num_nodes() == source.num_nodes() || source.num_channels() == 0);
        assert_eq!(dest.num_nodes(), self.num_nodes());

        // Make sure order is valid.
        assert!(matches!(order, 2 | 4 | 6));

        let dest = ImageShared::from(dest);

        self.try_block_compute(|mesh, store, block| {
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            let block_source = source.slice(nodes.clone());
            let block_dest = unsafe { dest.slice_mut(nodes.clone()) };

            if mesh.is_block_in_interior(block) {
                macro_rules! evaluate_int {
                    ($order:literal) => {
                        function.evaluate(
                            FdIntEngine::<N, $order> {
                                space: space.clone(),
                                store,
                                range: nodes.clone(),
                            },
                            block_source,
                            block_dest,
                        )
                    };
                }

                match order {
                    2 => evaluate_int!(2),
                    4 => evaluate_int!(4),
                    6 => evaluate_int!(6),
                    _ => unreachable!(),
                }
            } else {
                macro_rules! evaluate {
                    ($order:literal) => {
                        function.evaluate(
                            FdEngine::<N, $order> {
                                space: space.clone(),
                                store,
                                range: nodes.clone(),
                            },
                            block_source,
                            block_dest,
                        )
                    };
                }

                match order {
                    2 => evaluate!(2),
                    4 => evaluate!(4),
                    6 => evaluate!(6),
                    _ => unreachable!(),
                }
            }
        })
    }

    pub fn is_block_in_interior(&self, block: BlockId) -> bool {
        let boundary = self.block_boundary_classes(block);

        let mut result = true;

        for axis in 0..N {
            result &= boundary[Face::negative(axis)].has_ghost();
            result &= boundary[Face::positive(axis)].has_ghost();
        }

        result
    }

    /// Evaluates the given function on a system in place.
    fn evaluate_mut<
        const ORDER: usize,
        P: Function<N> + Sync,
    >(
        &mut self,
        function: P,
        dest: ImageMut,
    ) -> Result<(), P::Error>
    where
        P::Error: Send,
    {
        let dest = ImageShared::from(dest);

        self.try_block_compute(|mesh, store, block| {
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            let block_dest = unsafe { dest.slice_mut(nodes.clone()) };
            let mut block_source =
                ImageMut::from_storage(store.scratch(block_dest.num_nodes() * block_dest.num_channels()), block_dest.num_channels());

            for field in dest.channels() {
                block_source
                    .channel_mut(field)
                    .copy_from_slice(block_dest.channel(field));
            }

            if mesh.is_block_in_interior(block) {
                let engine = FdIntEngine::<N, ORDER> {
                    space: space.clone(),
                    store,
                    range: nodes.clone(),
                };

                function.evaluate(engine, block_source.rb(), block_dest)
            } else {
                let engine = FdEngine::<N, ORDER> {
                    space: space.clone(),
                    store,
                    range: nodes.clone(),
                };

                function.evaluate(engine, block_source.rb(), block_dest)
            }
        })
    }

    /// Applies an operator to a system in place, enforcing both strong and weak boundary conditions
    /// and running necessary preprocessing.
    pub fn apply<
        C: SystemBoundaryConds<N> + Sync,
        P: Function<N> + Sync,
    >(
        &mut self,
        order: usize,
        bcs: C,
        mut op: P,
        mut f: ImageMut<'_>,
    ) -> Result<(), P::Error>
    where
        P::Error: Send,
    {
        assert_eq!(f.num_nodes(), self.num_nodes());

        for field in f.channels() {
            assert!(
                is_boundary_compatible(&self.boundary, &bcs.field(field)),
                "Boundary Conditions incompatible with set boundary classes"
            )
        }

        // Strong boundary condition
        self.fill_boundary(order, bcs.clone(), f.rb_mut());
        // Preprocess data
        op.preprocess(self, f.rb_mut())?;

        let f: ImageShared = f.into();

        self.try_block_compute(|mesh, store, block| {
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);
            let bcs = mesh.block_bcs(block, bcs.clone());

            let mut block_dest = unsafe { f.slice_mut(nodes.clone()) };

            let mut block_source =
                ImageMut::from_storage(store.scratch(block_dest.num_nodes() * block_dest.num_channels()), block_dest.num_channels());

            for field in f.channels() {
                block_source
                    .channel_mut(field)
                    .copy_from_slice(block_dest.channel(field));
            }

            if mesh.is_block_in_interior(block) {
                macro_rules! evaluate_int {
                    ($order:literal) => {
                        op.evaluate(
                            FdIntEngine::<N, $order> {
                                space: space.clone(),
                                store,
                                range: nodes.clone(),
                            },
                            block_source.rb(),
                            block_dest.rb_mut(),
                        )
                    };
                }

                match order {
                    2 => evaluate_int!(2),
                    4 => evaluate_int!(4),
                    6 => evaluate_int!(6),
                    _ => unreachable!(),
                }
            } else {
                macro_rules! evaluate {
                    ($order:literal) => {
                        op.evaluate(
                            FdEngine::<N, $order> {
                                space: space.clone(),
                                store,
                                range: nodes.clone(),
                            },
                            block_source.rb(),
                            block_dest.rb_mut(),
                        )?
                    };
                }

                match order {
                    2 => evaluate!(2),
                    4 => evaluate!(4),
                    6 => evaluate!(6),
                    _ => unreachable!(),
                }

                // Weak boundary conditions.
                for face in Face::<N>::iterate() {
                    for field in f.channels() {
                        let boundary = bcs.field(field);
                        let source = block_source.channel(field);
                        let dest = block_dest.channel_mut(field);

                        // Apply weak dirichlet boundary conditions
                        if boundary.kind(face) == BoundaryKind::WeakDirichlet {
                            for node in space.face_window_disjoint(face) {
                                let index = space.index_from_node(node);
                                let position = space.position(node);
                                let dirichlet = boundary.dirichlet(position);
                                dest[index] =
                                    dirichlet.strength * (dirichlet.target - source[index])
                            }
                        }

                        // Apply radiative condition
                        if boundary.kind(face) != BoundaryKind::Radiative {
                            continue;
                        }

                        // Sommerfeld radiative boundary conditions.
                        for node in space.face_window(face) {
                            let vertex = vertex_from_node(node);
                            // *************************
                            // At vertex

                            let position: [f64; N] = space.position(node);
                            let r = position.iter().map(|&v| v * v).sum::<f64>().sqrt();
                            let index = space.index_from_node(node);

                            macro_rules! inner {
                                ($order:literal) => {
                                    // *************************
                                    // Inner

                                    let engine = FdEngine::<N, $order> {
                                        space: space.clone(),
                                        store,
                                        range: nodes.clone(),
                                    };

                                    let mut inner = vertex;

                                    // Find innter vertex for approximating higher order r dependence
                                    for axis in 0..N {
                                        if boundary.kind(Face::negative(axis)) == BoundaryKind::Radiative
                                            && vertex[axis] == 0
                                        {
                                            inner[axis] += 1;
                                        }
                                    
                                        if boundary.kind(Face::positive(axis)) == BoundaryKind::Radiative
                                            && vertex[axis] == engine.vertex_size()[axis] - 1
                                        {
                                            inner[axis] -= 1;
                                        }
                                    }
                                
                                    let inner_position = engine.position(inner);
                                    let inner_r = inner_position.iter().map(|&v| v * v).sum::<f64>().sqrt();
                                    let inner_index = engine.index_from_vertex(inner);
                                
                                    // Get condition parameters.
                                    let params = boundary.radiative(position);
                                    // Inner R dependence.
                                    let mut inner_advection = source[inner_index] - params.target;
                                
                                    for axis in 0..N {
                                        let derivative = engine.derivative(source, axis, inner);
                                        inner_advection += inner_position[axis] * derivative;
                                    }
                                
                                    inner_advection *= params.speed;
                                
                                    let k = inner_r
                                        * inner_r
                                        * inner_r
                                        * (dest[inner_index] + inner_advection / inner_r);
                                
                                    // Vertex
                                    let mut advection = source[index] - params.target;
                                
                                    for axis in 0..N {
                                        let derivative = engine.derivative(source, axis, vertex);
                                        advection += position[axis] * derivative;
                                    }
                                
                                    advection *= params.speed;
                                    dest[index] = -advection / r + k / (r * r * r);
                                };
                            }

                            match order {
                                2 => { inner!(2); },
                                4 => { inner!(4); },
                                6 => { inner!(6); },
                                _ => unimplemented!("Order unimplemented")
                            }
                        }
                    }
                }

                Ok(())
            }
        })
    }

    /// Copies an immutable src slice into a mutable dest slice.
    pub fn copy_from_slice(&mut self, mut dest: ImageMut, src: ImageRef) {
        assert_eq!(dest.num_nodes(), src.num_nodes());

        for label in dest.channels() {
            dest.channel_mut(label).copy_from_slice(src.channel(label));
        }
    }

    pub fn project<P: Projection<N> + Sync>(
        &mut self,
        order: usize,
        projection: P,
        dest: &mut [f64],
    ) {
        assert_eq!(dest.len(), self.num_nodes());
        self.evaluate(
            order,
            ProjectionAsFunction(projection),
            ImageRef::empty(),
            ImageMut::from(dest),
        )
        .unwrap();
    }

    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn dissipation<const ORDER: usize>(
        &mut self,
        amplitude: f64,
        mut dest: ImageMut,
    ) {
        assert_eq!(dest.num_nodes(), self.num_nodes());

        #[derive(Clone)]
        struct Dissipation(f64);

        impl<const N: usize> Function<N> for Dissipation {
            type Error = Infallible;

            fn evaluate(
                &self,
                engine: impl Engine<N>,
                input: ImageRef,
                mut output: ImageMut,
            ) -> Result<(), Infallible> {
                let input = input.channel(0);
                let output = output.channel_mut(0);

                for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                    let index = engine.index_from_vertex(vertex);

                    for axis in 0..N {
                        output[index] += self.0 * engine.dissipation(input, axis, vertex);
                    }
                }

                Ok(())
            }
        }

        for field in dest.channels() {
            self.evaluate_mut::<ORDER, _>(
                Dissipation(amplitude),
                ImageMut::from(dest.channel_mut(field)),
            )
            .unwrap();
        }
    }

    /// This function computes the distance between each vertex and its nearest
    /// neighbor.
    pub fn spacing_per_vertex(&mut self, dest: &mut [f64]) {
        assert!(dest.len() == self.num_nodes());

        let dest = SharedSlice::new(dest);

        self.block_compute(|mesh, _, block| {
            let nodes = mesh.block_nodes(block);
            let space = mesh.block_space(block);

            let spacing = space.spacing();
            let min_spacing = spacing
                .iter()
                .min_by(|a, b| a.total_cmp(&b))
                .cloned()
                .unwrap_or(1.0);

            let vertex_size = space.vertex_size();

            let block_dest = unsafe { dest.slice_mut(nodes) };

            for &cell in mesh.blocks.active_cells(block) {
                let node_size = mesh.cell_node_size(cell);
                let node_origin = mesh.active_node_origin(cell);

                let mut flags = FaceMask::empty();

                for face in Face::iterate() {
                    let Some(neighbor) = mesh
                        .tree
                        .neighbor(mesh.tree.cell_from_active_index(cell), face)
                    else {
                        continue;
                    };
                    // If neighbors have larger refinement than us
                    if !mesh.tree.is_active(neighbor) {
                        flags.set(face);
                    }
                }

                for offset in IndexSpace::new(node_size).iter() {
                    let vertex: [_; N] = array::from_fn(|axis| node_origin[axis] + offset[axis]);
                    let index = space.index_from_vertex(vertex);

                    let mut refined = false;

                    for axis in 0..N {
                        refined |= vertex[axis] == 0 && flags.is_set(Face::negative(axis));
                        refined |= vertex[axis] == vertex_size[axis] - 1
                            && flags.is_set(Face::positive(axis));
                    }

                    if refined {
                        block_dest[index] = min_spacing / 2.0;
                    } else {
                        block_dest[index] = min_spacing;
                    }
                }
            }
        });
    }

    pub fn adaptive_cfl(
        &mut self,
        spacing_per_vertex: &[f64],
        dest: ImageMut,
    ) {
        let dest = ImageShared::from(dest);
        let min_spacing = self.min_spacing();

        self.block_compute(|mesh, _, block| {
            let block_space = mesh.block_space(block);
            let block_nodes = mesh.block_nodes(block);

            let block_spacings = &spacing_per_vertex[block_nodes.clone()];
            let mut block_dest = unsafe { dest.slice_mut(block_nodes.clone()) };

            for field in block_dest.channels() {
                let block_dest = block_dest.channel_mut(field);

                for vertex in IndexSpace::new(block_space.vertex_size()).iter() {
                    let index = block_space.index_from_vertex(vertex);
                    block_dest[index] *= block_spacings[index] / min_spacing;
                }
            }
        });
    }
}
