use crate::{geometry::Face, prelude::SystemLabel};

/// Indicates what type of boundary condition is used along a particualr
/// face of the domain. More specific boundary conditions are provided
/// by the `Condition` API, but for many funtions, `Boundary` provides
/// enough information to compute supports and apply stencils.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryKind {
    /// A (anti)symmetric boundary condition. True indicates an even function
    /// (so function values are reflected across the axis with the same sign)
    /// and false indicates an odd function.
    Parity,
    /// This boundary condition indicates that the ghost nodes have been filled manually via some external
    /// process. This can be used to implement custom boundary conditions, and is primarily used to
    /// fill inter-grid boundaries in the adaptive mesh refinement driver.
    Custom,
    Radiative,
    Free,
}

impl BoundaryKind {
    /// Returns true if this boundary condition is weakly enforced
    pub fn is_weak(self) -> bool {
        matches!(self, Self::Radiative | Self::Free)
    }

    /// Returns true if this boundary condition is strongly enforced
    pub fn is_strong(self) -> bool {
        matches!(self, Self::Parity | Self::Custom)
    }
}

/// Provides information about what kind of boundary is used on each face.
pub trait Boundary: Clone {
    fn kind(&self, face: Face) -> BoundaryKind;
}

/// Provides specifics for enforcing boundary conditions, whether
/// strongly or weakly.
pub trait Condition<const N: usize> {
    fn parity(&self, _face: Face) -> bool {
        false
    }

    fn radiative(&self, _position: [f64; N]) -> f64 {
        0.0
    }
}

/// A generalization of `Condition<N>` for a coupled system.
pub trait Conditions<const N: usize> {
    type System: SystemLabel;
    type Condition: Condition<N>;

    fn field(&self, label: Self::System) -> Self::Condition;
}
