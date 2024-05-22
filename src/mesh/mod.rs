//! Meshes and adaptive mesh refinement.

use serde::{Deserialize, Serialize};
use std::array::from_fn;

use crate::array::Array;
use crate::common::{Boundary, NodeSpace};
use crate::geometry::Rectangle;

mod block;
mod driver;
mod model;
mod system;

pub use block::{Block, BlockExt};
pub use driver::{Driver, MemPool};
pub use model::Model;
pub use system::{field_count, Scalar, SystemLabel, SystemSlice, SystemSliceMut, SystemVec};

pub trait Projection<const N: usize> {
    fn evaluate(&self, block: Block<N>, pool: &MemPool, dest: &mut [f64]);
}

pub struct Gaussian(f64);

impl<const N: usize> Projection<N> for Gaussian {
    fn evaluate(&self, block: Block<N>, _pool: &MemPool, dest: &mut [f64]) {
        for vertex in block.iter() {
            let pos = block.position(vertex);
            let mag = pos.iter().map(|f| f * f).sum::<f64>();
            let index = block.index_from_vertex(vertex);

            dest[index] = self.0 * (-mag).exp();
        }
    }
}

pub trait Operator<const N: usize> {
    fn apply(&self, block: Block<N>, pool: &MemPool, src: &[f64], dest: &mut [f64]);
}

pub trait SystemProjection<const N: usize> {
    type Label: SystemLabel;

    fn evaluate(&self, block: Block<N>, pool: &MemPool, dest: SystemSliceMut<'_, Self::Label>);
}

pub trait SystemOperator<const N: usize> {
    type Label: SystemLabel;

    fn apply(
        &self,
        block: Block<N>,
        pool: &MemPool,
        src: SystemSlice<'_, Self::Label>,
        dest: SystemSliceMut<'_, Self::Label>,
    );
}

pub trait SystemBoundary {
    type Label: SystemLabel;
    type Boundary: Boundary;

    fn field(&self, label: Self::Label) -> Self::Boundary;
}

#[derive(Debug, Clone)]
pub struct Mesh<const N: usize> {
    /// Uniform bounds for mesh
    bounds: Rectangle<N>,
    /// Number of cells on base block
    size: [usize; N],
    /// Number of ghost cells.
    ghost: usize,
}

impl<const N: usize> Mesh<N> {
    /// Constructs a new mesh.
    pub fn new(bounds: Rectangle<N>, size: [usize; N], ghost: usize) -> Self {
        Self {
            bounds,
            size,
            ghost,
        }
    }

    /// Returns the number of nodes in the mesh.
    pub fn node_count(&self) -> usize {
        self.base_block().node_count()
    }

    /// Physical bounds of the mesh.
    pub fn bounds(&self) -> Rectangle<N> {
        self.bounds.clone()
    }

    pub fn minimum_spacing(&self) -> f64 {
        from_fn::<_, N, _>(|axis| self.base_block().space.spacing_axis(axis))
            .into_iter()
            .min_by(f64::total_cmp)
            .unwrap_or(0.0)
    }

    pub fn base_block(&self) -> Block<N> {
        let space = NodeSpace {
            bounds: self.bounds.clone(),
            size: self.size,
            ghost: self.ghost,
        };

        let node_count = space.node_count();

        Block::new(space, 0..node_count)
    }
}

impl<const N: usize> Serialize for Mesh<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        MeshSerde {
            bounds: self.bounds.clone(),
            size: self.size.into(),
            ghost: self.ghost,
        }
        .serialize(serializer)
    }
}

impl<'de, const N: usize> Deserialize<'de> for Mesh<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mesh = MeshSerde::deserialize(deserializer)?;

        Ok(Self {
            bounds: mesh.bounds,
            size: mesh.size.inner(),
            ghost: mesh.ghost,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct MeshSerde<const N: usize> {
    /// Uniform bounds for mesh
    bounds: Rectangle<N>,
    /// Number of cells on base block
    size: Array<[usize; N]>,
    /// Number of ghost cells.
    ghost: usize,
}
