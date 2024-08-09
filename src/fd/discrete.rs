use rayon::prelude::*;
use std::array::from_fn;

use crate::fd::{
    node_from_vertex, Boundary, BoundaryKind, Condition, Conditions, Engine, FdEngine, FdIntEngine,
    Function, Mesh, Model, Operator, Projection, SystemBC,
};
use crate::geometry::{faces, Face, IndexSpace, Rectangle};
use crate::system::{SystemLabel, SystemSlice, SystemSliceMut};

pub struct Discretization<const N: usize> {
    mesh: Mesh<N>,
    // pool: ThreadPool,
}

impl<const N: usize> Discretization<N> {
    pub fn new() -> Self {
        Self {
            mesh: Mesh::new(Rectangle::UNIT, [2; N], 0),
            // pool: ThreadPoolBuilder::new().build().unwrap(),
        }
    }

    pub fn mesh(&self) -> &Mesh<N> {
        &self.mesh
    }

    pub fn mesh_mut(&mut self) -> &mut Mesh<N> {
        &mut self.mesh
    }

    pub fn set_mesh(&mut self, mesh: &Mesh<N>) {
        self.mesh.clone_from(mesh);
    }

    pub fn set_mesh_from_model(&mut self, model: &Model<N>) {
        model.load_mesh(&mut self.mesh);
    }

    pub fn num_nodes(&self) -> usize {
        self.mesh.num_nodes()
    }

    pub fn order<const ORDER: usize>(&mut self) -> DiscretizationOrder<N, ORDER> {
        const {
            assert!(ORDER % 2 == 0);
        }
        debug_assert!(
            self.mesh.ghost() >= ORDER / 2,
            "Mesh has insufficient ghost nodes for the requested order."
        );
        DiscretizationOrder(self)
    }
}

pub struct DiscretizationOrder<'a, const N: usize, const ORDER: usize>(&'a mut Discretization<N>);

impl<'a, const N: usize, const ORDER: usize> DiscretizationOrder<'a, N, ORDER> {
    /// Enforces strong boundary conditions. This includes strong physical boundary conditions, as well
    /// as handling interior boundaries (same level, coarse-fine, or fine-coarse).
    pub fn fill_boundary<BC: Boundary<N> + Conditions<N> + Sync>(
        mut self,
        bc: BC,
        mut system: SystemSliceMut<'_, BC::System>,
    ) {
        self.fill_physical(&bc, &mut system);
        self.fill_direct(&mut system);
        self.fill_fine(&mut system);
        self.fill_prolong(&bc, &mut system);
    }

    fn fill_physical<BC: Boundary<N> + Conditions<N> + Sync>(
        &mut self,
        bc: &BC,
        system: &mut SystemSliceMut<'_, BC::System>,
    ) {
        let mesh = &self.0.mesh;

        let system = system.as_range();

        (0..mesh.num_blocks()).for_each(|block| {
            // Fill Physical Boundary conditions
            let nodes = mesh.block_nodes(block);
            let space = mesh.block_space(block);
            let boundary = mesh.block_boundary(block, bc.clone());

            let mut block_system = unsafe { system.slice_mut(nodes) };

            for field in BC::System::fields() {
                space
                    .set_context(SystemBC::new(field.clone(), boundary.clone()))
                    .fill_boundary(block_system.field_mut(field));
            }
        });
    }

    fn fill_direct<System: SystemLabel>(&mut self, system: &mut SystemSliceMut<'_, System>) {
        let mesh = self.0.mesh();
        let system = system.as_range();

        // Fill direct interfaces
        mesh.direct_interfaces().par_bridge().for_each(|interface| {
            let block_space = mesh.block_space(interface.block);
            let block_nodes = mesh.block_nodes(interface.block);
            let neighbor_space = mesh.block_space(interface.neighbor);
            let neighbor_nodes = mesh.block_nodes(interface.neighbor);

            let mut block_system = unsafe { system.slice_mut(block_nodes).fields_mut() };
            let neighbor_system = unsafe { system.slice(neighbor_nodes).fields() };

            for node in IndexSpace::new(interface.size).iter() {
                let block_node = from_fn(|axis| node[axis] as isize + interface.dest[axis]);
                let neighbor_node = from_fn(|axis| node[axis] as isize + interface.source[axis]);

                let block_index = block_space.index_from_node(block_node);
                let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                for field in System::fields() {
                    let value = neighbor_system.field(field.clone())[neighbor_index];
                    block_system.field_mut(field.clone())[block_index] = value;
                }
            }
        });
    }

    fn fill_fine<System: SystemLabel>(&mut self, system: &mut SystemSliceMut<'_, System>) {
        let mesh = self.0.mesh();
        let system = system.as_range();

        mesh.fine_interfaces().par_bridge().for_each(|interface| {
            let block_space = mesh.block_space(interface.block);
            let block_nodes = mesh.block_nodes(interface.block);
            let neighbor_space = mesh.block_space(interface.neighbor);
            let neighbor_nodes = mesh.block_nodes(interface.neighbor);

            let mut block_system = unsafe { system.slice_mut(block_nodes).fields_mut() };
            let neighbor_system = unsafe { system.slice(neighbor_nodes).fields() };

            for node in IndexSpace::new(interface.size).iter() {
                let block_node = from_fn(|axis| node[axis] as isize + interface.dest[axis]);
                let neighbor_node =
                    from_fn(|axis| 2 * (node[axis] as isize + interface.source[axis]));

                let block_index = block_space.index_from_node(block_node);
                let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                for field in System::fields() {
                    let value = neighbor_system.field(field.clone())[neighbor_index];
                    block_system.field_mut(field.clone())[block_index] = value;
                }
            }
        });
    }

    fn fill_prolong<BC: Boundary<N> + Conditions<N> + Sync>(
        &mut self,
        bc: &BC,
        system: &mut SystemSliceMut<'_, BC::System>,
    ) {
        let mesh = self.0.mesh();
        let system = system.as_range();

        mesh.coarse_interfaces().par_bridge().for_each(|interface| {
            let block_nodes = mesh.block_nodes(interface.block);
            let block_space = mesh.block_space(interface.block);
            let neighbor_nodes = mesh.block_nodes(interface.neighbor);
            let neighbor_boundary = mesh.block_boundary(interface.neighbor, bc.clone());
            let neighbor_space = mesh.block_space(interface.neighbor);

            let mut block_system = unsafe { system.slice_mut(block_nodes).fields_mut() };
            let neighbor_system = unsafe { system.slice(neighbor_nodes).fields() };

            for node in IndexSpace::new(interface.size).iter() {
                let block_node = from_fn(|axis| node[axis] as isize + interface.dest[axis]);

                let neighbor_vertex =
                    from_fn(|axis| (node[axis] as isize + interface.source[axis]) as usize);

                let block_index = block_space.index_from_node(block_node);

                for field in BC::System::fields() {
                    let bc = SystemBC::new(field.clone(), neighbor_boundary.clone());

                    let value = neighbor_space
                        .set_context(bc)
                        .prolong::<ORDER>(neighbor_vertex, neighbor_system.field(field.clone()));
                    block_system.field_mut(field.clone())[block_index] = value;
                }
            }
        });
    }

    pub fn weak_boundary<BC: Boundary<N> + Conditions<N>>(
        self,
        bc: BC,
        system: SystemSlice<'_, BC::System>,
        deriv: SystemSliceMut<'_, BC::System>,
    ) {
        let mesh = self.0.mesh();
        let system = system.as_range();
        let deriv = deriv.as_range();

        (0..mesh.num_blocks()).for_each(|block| {
            let boundary = mesh.block_boundary(block, bc.clone());
            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block).set_context(bounds);
            let nodes = mesh.block_nodes(block);
            let vertex_size = space.vertex_size();

            let block_system = unsafe { system.slice(nodes.clone()).fields() };
            let mut block_deriv = unsafe { deriv.slice_mut(nodes.clone()).fields_mut() };

            for face in faces::<N>() {
                if boundary.kind(face) != BoundaryKind::Radiative {
                    continue;
                }

                // Sommerfeld radiative boundary conditions.
                for vertex in IndexSpace::new(vertex_size).face(face).iter() {
                    // *************************
                    // At vertex

                    let engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        vertex,
                        boundary: boundary.clone(),
                    };
                    let position: [f64; N] = engine.position();
                    let r = position.iter().map(|&v| v * v).sum::<f64>().sqrt();
                    let index = space.index_from_vertex(vertex);

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
                            && vertex[axis] == vertex_size[axis] - 1
                        {
                            inner[axis] -= 1;
                        }
                    }

                    let inner_engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        vertex: inner,
                        boundary: boundary.clone(),
                    };

                    let inner_position = inner_engine.position();
                    let inner_r = inner_position.iter().map(|&v| v * v).sum::<f64>().sqrt();
                    let inner_index = space.index_from_vertex(inner);

                    for field in BC::System::fields() {
                        let field_system = block_system.field(field.clone());
                        let field_derivs = block_deriv.field_mut(field.clone());
                        let field_boundary = SystemBC::new(field.clone(), boundary.clone());

                        let target = Condition::radiative(&field_boundary, position);

                        // Inner R dependence
                        let inner_gradient =
                            inner_engine.gradient(field_boundary.clone(), field_system);
                        let mut inner_advection = inner_engine.value(field_system) - target;

                        for axis in 0..N {
                            inner_advection += inner_position[axis] * inner_gradient[axis];
                        }

                        let k = inner_r
                            * inner_r
                            * inner_r
                            * (field_derivs[inner_index] + inner_advection / inner_r);

                        // Vertex
                        let gradient = engine.gradient(field_boundary.clone(), field_system);
                        let mut advection = engine.value(field_system) - target;

                        for axis in 0..N {
                            advection += position[axis] * gradient[axis];
                        }

                        field_derivs[index] = -advection / r + k / (r * r * r);
                        // field_derivs[index] = -advection / r;
                    }
                }
            }
        });
    }

    /// Fills the system by applying the given function at each node on the mesh.
    pub fn evaluate<F: Function<N> + Sync>(self, f: F, system: SystemSliceMut<'_, F::Output>) {
        let mesh = self.0.mesh();
        let system = system.as_range();

        (0..mesh.num_blocks()).par_bridge().for_each(|block| {
            let nodes = mesh.block_nodes(block);

            let mut block_system = unsafe { system.slice_mut(nodes.clone()).fields_mut() };

            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block).set_context(bounds);
            let vertex_size = space.vertex_size();

            for vertex in IndexSpace::new(vertex_size).iter() {
                let position = space.position(node_from_vertex(vertex));
                let result = f.evaluate(position);

                let index = space.index_from_vertex(vertex);
                for field in F::Output::fields() {
                    block_system.field_mut(field.clone())[index] = result.field(field.clone());
                }
            }
        });
    }

    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn project<BC: Boundary<N> + Sync, P: Projection<N> + Sync>(
        self,
        boundary: BC,
        projection: P,
        source: SystemSlice<'_, P::Input>,
        dest: SystemSliceMut<'_, P::Output>,
    ) {
        let mesh = self.0.mesh();
        let source = source.as_range();
        let dest = dest.as_range();

        (0..mesh.num_blocks()).par_bridge().for_each(|block| {
            let nodes = mesh.block_nodes(block);

            let block_source = unsafe { source.slice(nodes.clone()).fields() };
            let mut block_dest = unsafe { dest.slice_mut(nodes.clone()).fields_mut() };

            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block).set_context(bounds);
            let vertex_size = space.vertex_size();

            let boundary = mesh.block_boundary(block, boundary.clone());

            for vertex in IndexSpace::new(vertex_size).iter() {
                let is_interior = Self::is_interior(&boundary, vertex_size, vertex);

                let result = if is_interior {
                    let engine = FdIntEngine::<N, ORDER> {
                        space: space.clone(),
                        vertex,
                    };

                    projection.project(&engine, block_source.as_fields())
                } else {
                    let engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        vertex,
                        boundary: boundary.clone(),
                    };

                    projection.project(&engine, block_source.as_fields())
                };

                let index = space.index_from_vertex(vertex);

                for field in P::Output::fields() {
                    block_dest.field_mut(field.clone())[index] = result.field(field.clone());
                }
            }
        });
    }

    /// Applies the given operator to `source`, storing the result in `dest`, and utilizing `context` to store
    /// extra fields.
    pub fn apply<BC: Boundary<N> + Sync, O: Operator<N> + Sync>(
        self,
        boundary: BC,
        operator: O,
        source: SystemSlice<'_, O::System>,
        context: SystemSlice<'_, O::Context>,
        dest: SystemSliceMut<'_, O::System>,
    ) {
        let mesh = self.0.mesh();
        let source = source.as_range();
        let context = context.as_range();
        let dest = dest.as_range();

        (0..mesh.num_blocks()).par_bridge().for_each(|block| {
            let nodes = mesh.block_nodes(block);

            let block_source = unsafe { source.slice(nodes.clone()).fields() };
            let block_context = unsafe { context.slice(nodes.clone()).fields() };
            let mut block_dest = unsafe { dest.slice_mut(nodes.clone()).fields_mut() };

            let boundary = mesh.block_boundary(block, boundary.clone());
            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block).set_context(bounds);
            let vertex_size = space.vertex_size();

            for vertex in IndexSpace::new(vertex_size).iter() {
                let is_interior = Self::is_interior(&boundary, vertex_size, vertex);

                let result = if is_interior {
                    let engine = FdIntEngine::<N, ORDER> {
                        space: space.clone(),
                        vertex,
                    };

                    operator.apply(&engine, block_source.as_fields(), block_context.as_fields())
                } else {
                    let engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        vertex,
                        boundary: boundary.clone(),
                    };

                    operator.apply(&engine, block_source.as_fields(), block_context.as_fields())
                };

                let index = space.index_from_node(node_from_vertex(vertex));

                for field in O::System::fields() {
                    block_dest.field_mut(field.clone())[index] = result.field(field.clone());
                }
            }
        });
    }

    pub fn dissipation<BC: Boundary<N> + Conditions<N> + Sync>(
        self,
        bc: BC,
        source: SystemSlice<'_, BC::System>,
        dest: SystemSliceMut<'_, BC::System>,
    ) {
        let mesh = self.0.mesh();
        let source = source.as_range();
        let dest = dest.as_range();

        (0..mesh.num_blocks()).par_bridge().for_each(|block| {
            let nodes = mesh.block_nodes(block);

            let block_source = unsafe { source.slice(nodes.clone()).fields() };
            let mut block_dest = unsafe { dest.slice_mut(nodes.clone()).fields_mut() };

            let boundary = mesh.block_boundary(block, bc.clone());
            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block).set_context(bounds);

            let vertex_size = space.vertex_size();

            for field in BC::System::fields() {
                let src = block_source.field(field.clone());
                let dst = block_dest.field_mut(field.clone());

                let space = space.set_bc(SystemBC::new(field.clone(), boundary.clone()));

                for vertex in IndexSpace::new(vertex_size).iter() {
                    let value = space.dissipation::<ORDER>(vertex, src);
                    space.set_value(node_from_vertex(vertex), value, dst);
                }
            }
        });
    }

    /// Determines if a vertex is not within `ORDER` of any weakly enforced boundary.
    fn is_interior(
        boundary: &impl Boundary<N>,
        vertex_size: [usize; N],
        vertex: [usize; N],
    ) -> bool {
        let mut result = true;

        for axis in 0..N {
            result &= boundary.kind(Face::negative(axis)).has_ghost() || vertex[axis] >= ORDER / 2;
            result &= boundary.kind(Face::positive(axis)).has_ghost()
                || vertex[axis] < vertex_size[axis] - ORDER / 2;
        }

        result

        // false
    }
}

impl<const N: usize> Discretization<N> {
    /// Computes the maximum l2 norm of all fields in the system.
    pub fn norm<S: SystemLabel>(&mut self, source: SystemSlice<'_, S>) -> f64 {
        S::fields()
            .into_iter()
            .map(|label| self.norm_scalar(source.field(label)))
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    fn norm_scalar(&mut self, src: &[f64]) -> f64 {
        let mut result = 0.0;

        for block in 0..self.mesh.num_blocks() {
            let bounds = self.mesh.block_bounds(block);
            let space = self.mesh.block_space(block).set_context(bounds);
            let vertex_size = space.vertex_size();

            let data = &src[self.mesh.block_nodes(block)];

            let mut block_result = 0.0;

            for vertex in IndexSpace::new(vertex_size).iter() {
                let index = space.index_from_vertex(vertex);

                let mut value = data[index] * data[index];

                for axis in 0..N {
                    if vertex[axis] == 0 || vertex[axis] == vertex_size[axis] - 1 {
                        value *= 0.5;
                    }
                }

                block_result += value;
            }

            for spacing in space.spacing() {
                block_result *= spacing;
            }

            result += block_result;
        }

        result.sqrt()
    }
}
