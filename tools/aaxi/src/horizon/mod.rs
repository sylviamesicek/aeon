use crate::rinne::{Field, Fields, Metric};
use aeon::{
    element::UniformInterpolate,
    kernel::{Hessian, Kernels, NodeWindow, node_from_vertex},
    prelude::*,
};
use aeon_tensor::Tensor;
use core::f64;
use faer::linalg::svd::SvdError;
use reborrow::Reborrow;
use std::convert::Infallible;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HorizonError {
    #[error("surface point not contained in mesh: ${0:?}")]
    SurfaceNotContained([f64; 2]),
    #[error("interpolation didn't converge")]
    InterpolateFailed,
}

#[derive(Clone)]
pub struct ApparentHorizonFinder {
    surface: Mesh<1>,
    surface_radius: SystemVec<Scalar>,

    surface_to_cell: Vec<CellId>,
    surface_position: Vec<[f64; 2]>,

    prolong: UniformInterpolate<2>,
}

impl ApparentHorizonFinder {
    pub fn new() -> Self {
        let mut surface = Mesh::new(
            Rectangle::from_aabb([0.0], [f64::consts::PI / 2.0]),
            4,
            2,
            FaceArray::from_fn(|face| BoundaryClass::Periodic),
        );

        for _ in 0..3 {
            surface.refine_global();
        }

        let mut surface_radius = SystemVec::new(Scalar);
        surface_radius.resize(surface.num_nodes());
        surface_radius.field_mut(()).fill(1.0);

        let surface_to_cell = vec![CellId(0); surface.num_nodes()];
        let surface_position = vec![[0.0; 2]; surface.num_nodes()];

        Self {
            surface,
            surface_radius,

            surface_to_cell,
            surface_position,
            prolong: UniformInterpolate::default(),
        }
    }

    pub fn search(&mut self, mesh: &Mesh<2>, fields: SystemSlice<Fields>) -> eyre::Result<()> {
        Ok(())
    }
}

struct HorizonRadialDerivs<'a, K> {
    surface_to_cell: &'a mut Vec<CellId>,
    surface_position: &'a mut Vec<[f64; 2]>,
    mesh: &'a Mesh<2>,
    fields: SystemSlice<'a, Fields>,
    _phantom: std::marker::PhantomData<K>,
}

impl<'a, K: Kernels> Function<1> for HorizonRadialDerivs<'a, K> {
    type Input = Scalar;
    type Output = Scalar;
    type Error = HorizonError;

    fn preprocess(
        &mut self,
        surface: &mut Mesh<1>,
        mut input: SystemSliceMut<Self::Input>,
    ) -> Result<(), Self::Error> {
        let radius = input.field_mut(());

        for block in 0..surface.num_blocks() {
            let nodes = surface.block_nodes(BlockId(block));
            let space = surface.block_space(BlockId(block));

            let surface_radius = &radius[nodes.clone()];
            let surface_position = &mut self.surface_position[nodes.clone()];
            let surface_to_cell = &mut self.surface_to_cell[nodes.clone()];

            for node in space.inner_window() {
                let index = space.index_from_node(node);
                let [theta] = space.position(node);
                let radius = surface_radius[index];

                let point = position(radius, theta);

                if self.mesh.tree().domain().contains(point) {
                    return Err(HorizonError::SurfaceNotContained(point));
                }

                let cell = self
                    .mesh
                    .tree()
                    .cell_from_point_cached(point, surface_to_cell[index]);

                surface_position[index] = point;
                surface_to_cell[index] = cell;
            }
        }

        Ok(())
    }

    fn evaluate(
        &self,
        engine: impl Engine<1>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Self::Error> {
        let mesh = self.mesh;
        let fields = self.fields.rb();

        let input = input.field(());
        let deriv = output.field_mut(());

        let nodes = engine.node_range();

        let surface_positions = &self.surface_position[nodes.clone()];
        let surface_to_cell = &self.surface_to_cell[nodes.clone()];

        // Size of every cell along each axis
        let cell_size = [mesh.width() + 1; 2];
        // Number of support points per cell axis
        let cell_support = mesh.width() + 1;

        let num_nodes_per_cell = cell_size.iter().product();

        let scratch: &mut [f64] = engine.alloc(num_nodes_per_cell);

        let mut interpolate = UniformInterpolate::<2>::default();

        for [vertex] in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex([vertex]);

            let surface_radius = input[index];
            let surface_position = surface_positions[index];

            let mesh_cell = surface_to_cell[index];
            let mesh_active_cell = mesh.tree().active_index_from_cell(mesh_cell).unwrap();
            let mesh_block = mesh.blocks().active_cell_block(mesh_active_cell);

            let block_space = mesh.block_space(mesh_block);
            let cell_space = NodeWindow {
                origin: node_from_vertex(mesh.cell_node_origin(mesh_active_cell)),
                size: cell_size,
            };

            interpolate
                .build(cell_support, cell_support, surface_position)
                .map_err(|_err| HorizonError::InterpolateFailed)?;

            let block_fields = fields.slice(nodes.clone());

            let grr_f = block_fields.field(Field::Metric(Metric::Grr));
            let grz_f = block_fields.field(Field::Metric(Metric::Grz));
            let gzz_f = block_fields.field(Field::Metric(Metric::Gzz));
            let s_f = block_fields.field(Field::Metric(Metric::S));

            let krr_f = block_fields.field(Field::Metric(Metric::Krr));
            let krz_f = block_fields.field(Field::Metric(Metric::Krz));
            let kzz_f = block_fields.field(Field::Metric(Metric::Kzz));
            let y_f = block_fields.field(Field::Metric(Metric::Y));

            // Handle metric values
            macro_rules! interpolate_value {
                ($output:ident, $field:ident) => {
                    for (i, node) in cell_space.iter().enumerate() {
                        scratch[i] = $field[block_space.index_from_node(node)];
                    }
                    let $output = interpolate.apply(&scratch);
                };
            }

            macro_rules! interpolate_derivative {
                ($output:ident, $field:ident, $axis:expr) => {
                    for (i, node) in cell_space.iter().enumerate() {
                        scratch[i] =
                            block_space.evaluate_axis(K::derivative(), node, $field, $axis);
                    }
                    let $output = interpolate.apply(&scratch);
                };
            }

            interpolate_value!(grr, grr_f);
            interpolate_value!(grz, grz_f);
            interpolate_value!(gzz, gzz_f);
            interpolate_derivative!(grr_r, grr_f, 0);
            interpolate_derivative!(grr_z, grr_f, 1);
            interpolate_derivative!(grz_r, grz_f, 0);
            interpolate_derivative!(grz_z, grz_f, 1);
            interpolate_derivative!(gzz_r, gzz_f, 0);
            interpolate_derivative!(gzz_z, gzz_f, 1);
            interpolate_value!(s, s_f);
            interpolate_derivative!(s_r, s_f, 0);
            interpolate_derivative!(s_z, s_f, 1);

            interpolate_value!(krr, krr_f);
            interpolate_value!(krz, krz_f);
            interpolate_value!(kzz, kzz_f);
            interpolate_value!(y, y_f);

            let g = [[grr, grz], [grz, gzz]].into();
            let g_partials = {
                let grr_par = [grr_r, grr_z];
                let grz_par = [grz_r, grz_z];
                let gzz_par = [gzz_r, gzz_z];

                [[grr_par, grz_par], [grz_par, gzz_par]].into()
            };
            let g_second_partials = Tensor::zeros();

            let metric = aeon_tensor::Metric::new(g, g_partials, g_second_partials);

            let g = [[grr, grz], [grz, gzz]];
        }

        Ok(())
    }
}

fn position(radius: f64, theta: f64) -> [f64; 2] {
    let x = radius * theta.cos();
    let y = radius * theta.sin();

    [x.max(0.0), y.max(0.0)]
}
