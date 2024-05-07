use crate::geometry::Face;

/// Used to strongly enforce boundary conditions along faces.
pub trait Boundary<const N: usize> {
    fn kind(&self, face: Face) -> BoundaryKind;
}

/// Represents a strongly enforced boundary condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryKind {
    /// The default boundary procedure: use increasingly one-sided stencils in boundary region.
    Free,
    /// A (anti)symmetric boundary condition. True indicates an even function
    /// (so function values are reflected across the axis with the same sign)
    /// and false indicates an odd function.
    Parity(bool),
    /// This boundary condition indicates that the ghost nodes have been filled manually via some external
    /// process. This can be used to implement custom boundary conditions, and is primarily used to
    /// fill inter-grid boundaries in the adaptive mesh refinement driver.
    Custom,
}

impl Default for BoundaryKind {
    fn default() -> Self {
        Self::Free
    }
}

impl BoundaryKind {
    pub fn is_custom(self) -> bool {
        match self {
            Self::Custom => true,
            _ => false,
        }
    }

    pub fn is_free(self) -> bool {
        match self {
            Self::Free => true,
            _ => false,
        }
    }
}
