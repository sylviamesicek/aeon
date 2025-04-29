use std::{array, ops::Range};

use crate::geometry::{BlockId, Face, FaceMask, IndexSpace, faces};
use crate::kernel::is_boundary_compatible;
use crate::{
    kernel::{
        BoundaryConds as _, BoundaryKind, Hessian, Kernels, NodeSpace, Order, VertexKernel,
        node_from_vertex, vertex_from_node,
    },
    system::Empty,
};
use reborrow::{Reborrow, ReborrowMut as _};

use crate::{
    mesh::{Engine, Function, Projection},
    shared::SharedSlice,
    system::{Scalar, System, SystemBoundaryConds, SystemSlice, SystemSliceMut},
};

use super::{Mesh, MeshStore};

/// A finite difference engine of a given order, but potentially bordering a free boundary.
struct FdEngine<'store, const N: usize, K: Kernels> {
    space: NodeSpace<N>,
    _order: K,
    store: &'store MeshStore,
    range: Range<usize>,
}

impl<'store, const N: usize, K: Kernels> FdEngine<'store, N, K> {
    fn evaluate_axis(
        &self,
        field: &[f64],
        axis: usize,
        kernel: &impl VertexKernel,
        vertex: [usize; N],
    ) -> f64 {
        self.space
            .evaluate_axis(kernel, node_from_vertex(vertex), field, axis)
    }
}

impl<'store, const N: usize, K: Kernels> Engine<N> for FdEngine<'store, N, K> {
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
        self.evaluate_axis(field, axis, K::derivative(), vertex)
    }

    fn second_derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate_axis(field, axis, K::second_derivative(), vertex)
    }

    fn mixed_derivative(&self, field: &[f64], i: usize, j: usize, vertex: [usize; N]) -> f64 {
        self.space
            .evaluate(Hessian::<K>::new(i, j), node_from_vertex(vertex), field)
    }

    fn dissipation(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate_axis(field, axis, K::dissipation(), vertex)
    }
}

/// A finite difference engine that only every relies on interior support (and can thus use better optimized stencils).
struct FdIntEngine<'store, const N: usize, K: Kernels> {
    space: NodeSpace<N>,
    _order: K,
    store: &'store MeshStore,
    range: Range<usize>,
}

impl<'store, const N: usize, K: Kernels> FdIntEngine<'store, N, K> {
    fn evaluate(
        &self,
        field: &[f64],
        axis: usize,
        kernel: &impl VertexKernel,
        vertex: [usize; N],
    ) -> f64 {
        self.space
            .evaluate_axis_interior(kernel, node_from_vertex(vertex), field, axis)
    }
}

impl<'store, const N: usize, K: Kernels> Engine<N> for FdIntEngine<'store, N, K> {
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
        self.evaluate(field, axis, K::derivative(), vertex)
    }

    fn second_derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate(field, axis, K::second_derivative(), vertex)
    }

    fn mixed_derivative(&self, field: &[f64], i: usize, j: usize, vertex: [usize; N]) -> f64 {
        self.space
            .evaluate_interior(Hessian::<K>::new(i, j), node_from_vertex(vertex), field)
    }

    fn dissipation(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate(field, axis, K::dissipation(), vertex)
    }
}

/// Transforms a projection into a function.
#[derive(Clone)]
struct ProjectionAsFunction<P>(P);

impl<const N: usize, P: Projection<N>> Function<N> for ProjectionAsFunction<P> {
    type Input = Empty;
    type Output = Scalar;

    fn evaluate(
        &self,
        engine: impl Engine<N>,
        _input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
        let dest = output.field_mut(());

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            dest[index] = self.0.project(engine.position(vertex))
        }
    }
}

impl<const N: usize> Mesh<N> {
    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn evaluate<P: Function<N> + Sync>(
        &mut self,
        order: usize,
        function: P,
        source: SystemSlice<'_, P::Input>,
        dest: SystemSliceMut<'_, P::Output>,
    ) where
        P::Input: Sync,
        P::Output: Sync,
    {
        // Make sure order is valid.
        assert!(matches!(order, 2 | 4 | 6));

        let dest = dest.into_shared();

        self.block_compute(|mesh, store, block| {
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            let block_source = source.slice(nodes.clone());
            let block_dest = unsafe { dest.slice_mut(nodes.clone()) };

            if mesh.is_block_in_interior(block) {
                macro_rules! evaluate_int {
                    ($order:literal) => {
                        function.evaluate(
                            FdIntEngine {
                                space: space.clone(),
                                _order: Order::<$order>,
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
                            FdEngine {
                                space: space.clone(),
                                _order: Order::<$order>,
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
        });
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
        S: System + Sync,
        K: Kernels + Sync,
        P: Function<N, Input = S, Output = S> + Sync,
    >(
        &mut self,
        order: K,
        function: P,
        dest: SystemSliceMut<'_, S>,
    ) {
        let dest = dest.into_shared();

        self.block_compute(|mesh, store, block| {
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            let block_dest = unsafe { dest.slice_mut(nodes.clone()) };

            let num_nodes = block_dest.len() * dest.system().count();
            let mut block_source =
                SystemSliceMut::from_contiguous(store.scratch(num_nodes), dest.system());

            for field in dest.system().enumerate() {
                block_source
                    .field_mut(field)
                    .copy_from_slice(block_dest.field(field));
            }

            if mesh.is_block_in_interior(block) {
                let engine = FdIntEngine {
                    space: space.clone(),

                    _order: order,
                    store,
                    range: nodes.clone(),
                };

                function.evaluate(engine, block_source.rb(), block_dest);
            } else {
                let engine = FdEngine {
                    space: space.clone(),

                    _order: order,
                    store,
                    range: nodes.clone(),
                };

                function.evaluate(engine, block_source.rb(), block_dest);
            }
        });
    }

    /// Applies an operator to a system in place, enforcing both strong and weak boundary conditions
    /// and running necessary preprocessing.
    pub fn apply<
        O: Kernels + Sync,
        C: SystemBoundaryConds<N> + Sync,
        P: Function<N, Input = C::System, Output = C::System> + Sync,
    >(
        &mut self,
        order: O,
        bcs: C,
        op: P,
        mut f: SystemSliceMut<'_, C::System>,
    ) where
        C::System: Sync,
    {
        for field in f.system().enumerate() {
            assert!(
                is_boundary_compatible(&self.boundary, &bcs.field(field)),
                "Boundary Conditions incompatible with set boundary classes"
            )
        }

        // Strong boundary condition
        self.fill_boundary(order, bcs.clone(), f.rb_mut());
        // Preprocess data
        op.preprocess(self, f.rb_mut());

        let f = f.into_shared();

        self.block_compute(|mesh, store, block| {
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);
            let bcs = mesh.block_bcs(block, bcs.clone());

            let mut block_dest = unsafe { f.slice_mut(nodes.clone()) };

            let num_nodes = block_dest.len() * f.system().count();
            let mut block_source =
                SystemSliceMut::from_contiguous(store.scratch(num_nodes), f.system());

            for field in f.system().enumerate() {
                block_source
                    .field_mut(field)
                    .copy_from_slice(block_dest.field(field));
            }

            if mesh.is_block_in_interior(block) {
                let engine = FdIntEngine {
                    space: space.clone(),
                    _order: order,
                    store,
                    range: nodes.clone(),
                };

                op.evaluate(engine, block_source.rb(), block_dest.rb_mut());
            } else {
                let engine = FdEngine {
                    space: space.clone(),

                    _order: order,
                    store,
                    range: nodes.clone(),
                };

                op.evaluate(&engine, block_source.rb(), block_dest.rb_mut());

                // Weak boundary conditions.
                for face in faces::<N>() {
                    for field in f.system().enumerate() {
                        let boundary = bcs.field(field);
                        let source = block_source.field(field);
                        let dest = block_dest.field_mut(field);

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

                            // *************************
                            // Inner

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
                        }
                    }
                }
            }
        });
    }

    /// Copies an immutable src slice into a mutable dest slice.
    pub fn copy_from_slice<S: System>(&mut self, mut dest: SystemSliceMut<S>, src: SystemSlice<S>) {
        for label in dest.system().enumerate() {
            dest.field_mut(label).copy_from_slice(src.field(label));
        }
    }

    pub fn project<P: Projection<N> + Sync>(
        &mut self,
        order: usize,
        projection: P,
        dest: &mut [f64],
    ) {
        self.evaluate(
            order,
            ProjectionAsFunction(projection),
            SystemSlice::empty(),
            SystemSliceMut::from_scalar(dest),
        );
    }

    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn dissipation<K: Kernels + Sync, S: System>(
        &mut self,
        order: K,
        amplitude: f64,
        mut dest: SystemSliceMut<S>,
    ) {
        #[derive(Clone)]
        struct Dissipation(f64);

        impl<const N: usize> Function<N> for Dissipation {
            type Input = Scalar;
            type Output = Scalar;

            fn evaluate(
                &self,
                engine: impl Engine<N>,
                input: SystemSlice<Self::Input>,
                mut output: SystemSliceMut<Self::Output>,
            ) {
                let input = input.field(());
                let output = output.field_mut(());

                for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                    let index = engine.index_from_vertex(vertex);

                    for axis in 0..N {
                        output[index] += self.0 * engine.dissipation(input, axis, vertex);
                    }
                }
            }
        }

        for field in dest.system().enumerate() {
            self.evaluate_mut(
                order,
                Dissipation(amplitude),
                SystemSliceMut::from_scalar(dest.field_mut(field)),
            );
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
                let node_origin = mesh.cell_node_origin(cell);

                let mut flags = FaceMask::empty();

                for face in faces() {
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

    pub fn adaptive_cfl<S: System + Sync>(
        &mut self,
        spacing_per_vertex: &[f64],
        dest: SystemSliceMut<S>,
    ) {
        let dest = dest.into_shared();
        let min_spacing = self.min_spacing();

        self.block_compute(|mesh, _, block| {
            let block_space = mesh.block_space(block);
            let block_nodes = mesh.block_nodes(block);

            let block_spacings = &spacing_per_vertex[block_nodes.clone()];
            let mut block_dest = unsafe { dest.slice_mut(block_nodes.clone()) };

            for field in block_dest.system().enumerate() {
                let block_dest = block_dest.field_mut(field);

                for vertex in IndexSpace::new(block_space.vertex_size()).iter() {
                    let index = block_space.index_from_vertex(vertex);
                    block_dest[index] *= block_spacings[index] / min_spacing;
                }
            }
        });
    }
}
