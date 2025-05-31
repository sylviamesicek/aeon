use core::f64;

use crate::rinne::Fields;
use aeon::{element::ProlongEngine, prelude::*};

#[derive(Clone)]
pub struct ApparentHorizonFinder {
    surface: Mesh<1>,
    surface_radius: SystemVec<Scalar>,

    surface_to_cell: Vec<CellId>,
    surface_position: Vec<[f64; 2]>,

    prolong: ProlongEngine<2>,
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
            prolong: ProlongEngine::default(),
        }
    }

    pub fn search(&mut self, mesh: &Mesh<2>, fields: SystemSlice<Fields>) -> eyre::Result<()> {
        Ok(())
    }

    pub fn build(&mut self, mesh: &Mesh<2>) {
        for block in 0..self.surface.num_blocks() {
            let nodes = self.surface.block_nodes(BlockId(block));
            let space = self.surface.block_space(BlockId(block));

            let surface_radius = &self.surface_radius.field(())[nodes.clone()];
            let surface_position = &mut self.surface_position[nodes.clone()];
            let surface_to_cell = &mut self.surface_to_cell[nodes.clone()];

            for node in space.inner_window() {
                let index = space.index_from_node(node);
                let [theta] = space.position(node);
                let radius = surface_radius[index];

                let point = position(radius, theta);

                if mesh.tree().domain().contains(point) {
                    // TODO Fail gracefully
                    unimplemented!()
                }

                let cell = mesh
                    .tree()
                    .cell_from_point_cached(point, surface_to_cell[index]);

                surface_position[index] = point;
                surface_to_cell[index] = cell;
            }
        }
    }
}

fn position(radius: f64, theta: f64) -> [f64; 2] {
    let x = radius * theta.cos();
    let y = radius * theta.sin();

    [x.max(0.0), y.max(0.0)]
}
