use std::array;

use aeon_geometry::{Rectangle, Tree, TreeBlocks, TreeDofs, TreeInterfaces, TreeNeighbors};

#[derive(Debug, Clone)]
pub struct Mesh<const N: usize> {
    tree: Tree<N>,
    width: [usize; N],
    ghost: usize,

    blocks: TreeBlocks<N>,
    neighbors: TreeNeighbors<N>,

    dofs: TreeDofs<N>,
    interfaces: TreeInterfaces<N>,

    coarse_dofs: TreeDofs<N>,
    coarse_interfaces: TreeInterfaces<N>,
}

impl<const N: usize> Mesh<N> {
    pub fn new(bounds: Rectangle<N>, width: [usize; N], ghost: usize) -> Self {
        let tree = Tree::new(bounds);

        let mut result = Self {
            tree,
            width,
            ghost,

            blocks: TreeBlocks::default(),
            neighbors: TreeNeighbors::default(),
        
            dofs: TreeDofs::default(),
            interfaces: TreeInterfaces::default(),
        
            coarse_dofs: TreeDofs::default(),
            coarse_interfaces: TreeInterfaces::default(),
        };

        result.build();

        result

    }

    pub fn build(&mut self) {
        self.blocks.build(&self.tree);
        self.neighbors.build(&self.tree, &self.blocks);

        self.dofs.width = self.width;
        self.dofs.ghost = self.ghost;
        self.dofs.build(&self.blocks);
        self.interfaces.build(&self.tree, &self.blocks, &self.neighbors, &self.dofs);

        self.coarse_dofs.width = array::from_fn(|axis| self.width[axis] / 2);
        self.coarse_dofs.ghost = self.ghost;
        self.coarse_dofs.build(&self.blocks);
        self.coarse_interfaces.build(&self.tree, &self.blocks, &self.neighbors, &self.coarse_dofs);
    }

    pub fn save_to_disk(&self, disk: &mut MeshDisk<N>) {
        disk.tree.clone_from(&self.tree);
        disk.width = self.width;
        disk.ghost = self.ghost;
    }

    pub fn load_from_disk(&mut self, disk: &MeshDisk<N>) {
        self.tree.clone_from(&disk.tree);
        self.width = disk.width;
        self.ghost = disk.ghost;
    }
}

impl<const N: usize> Default for Mesh<N> {
    fn default() -> Self {
        let mut result = Self {
            tree: Tree::new(Rectangle::UNIT),
            width: [4; N],
            ghost: 1,

            blocks: TreeBlocks::default(),
            neighbors: TreeNeighbors::default(),
        
            dofs: TreeDofs::default(),
            interfaces: TreeInterfaces::default(),
        
            coarse_dofs: TreeDofs::default(),
            coarse_interfaces: TreeInterfaces::default(),
        };

        result.build();

        result
    }
}

/// Represents all information nessessary to store and load meshes from
/// disk.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct MeshDisk<const N: usize> {
    tree: Tree<N>,
    #[serde(with="aeon_array")]
    width: [usize; N],
    ghost: usize,
}

impl<const N: usize> Default for MeshDisk<N> {
    fn default() -> Self {
        Self {
            tree: Tree::new(Rectangle::UNIT),
            width: [4; N],
            ghost: 1,
        }
    }
} 