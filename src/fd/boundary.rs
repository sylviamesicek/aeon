use crate::{
    geometry::{Face, FaceMask},
    prelude::{Rectangle, SystemLabel},
};

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
    pub fn has_ghost(self) -> bool {
        matches!(self, Self::Custom)
    }
}

/// Provides information about what kind of boundary is used on each face.
pub trait Boundary<const N: usize>: Clone {
    fn kind(&self, face: Face<N>) -> BoundaryKind;
}

pub trait Domain<const N: usize>: Clone {
    fn bounds(&self) -> Rectangle<N>;
}

impl<const N: usize> Domain<N> for Rectangle<N> {
    fn bounds(&self) -> Rectangle<N> {
        self.clone()
    }
}

/// Provides specifics for enforcing boundary conditions, whether
/// strongly or weakly.
pub trait Condition<const N: usize>: Clone {
    fn parity(&self, _face: Face<N>) -> bool {
        false
    }

    fn radiative(&self, _position: [f64; N]) -> f64 {
        0.0
    }

    // fn attach_boundary<B: Boundary<N>>(self, boundary: B) -> AttachBoundary<Self, B> {
    //     AttachBoundary {
    //         inner: self,
    //         boundary,
    //     }
    // }
}

/// A generalization of `Condition<N>` for a coupled system.
pub trait Conditions<const N: usize>: Clone {
    type System: SystemLabel;

    fn parity(&self, _field: Self::System, _face: Face<N>) -> bool {
        false
    }

    fn radiative(&self, _field: Self::System, _position: [f64; N]) -> f64 {
        0.0
    }
}

#[derive(Clone)]
pub struct SystemBC<S: SystemLabel, I> {
    field: S,
    inner: I,
}

impl<S: SystemLabel, C> SystemBC<S, C> {
    pub fn new(field: S, inner: C) -> Self {
        Self { field, inner }
    }
}

impl<const N: usize, S: SystemLabel, C: Conditions<N, System = S>> Condition<N> for SystemBC<S, C> {
    fn parity(&self, face: Face<N>) -> bool {
        self.inner.parity(self.field.clone(), face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.inner.radiative(self.field.clone(), position)
    }
}

impl<const N: usize, S: SystemLabel, B: Boundary<N>> Boundary<N> for SystemBC<S, B> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        self.inner.kind(face)
    }
}

impl<const N: usize, S: SystemLabel, B: Domain<N>> Domain<N> for SystemBC<S, B> {
    fn bounds(&self) -> Rectangle<N> {
        self.inner.bounds()
    }
}

#[derive(Debug, Clone)]
pub struct BlockBC<const N: usize, I> {
    pub inner: I,
    pub flags: FaceMask<N>,
}

impl<const N: usize, I> BlockBC<N, I> {
    pub fn new(flags: FaceMask<N>, inner: I) -> Self {
        Self { inner, flags }
    }
}

impl<const N: usize, I: Boundary<N>> Boundary<N> for BlockBC<N, I> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        if self.flags.is_set(face) {
            self.inner.kind(face)
        } else {
            BoundaryKind::Custom
        }
    }
}

impl<const N: usize, I: Domain<N>> Domain<N> for BlockBC<N, I> {
    fn bounds(&self) -> Rectangle<N> {
        self.inner.bounds()
    }
}

impl<const N: usize, I: Condition<N>> Condition<N> for BlockBC<N, I> {
    fn parity(&self, face: Face<N>) -> bool {
        self.inner.parity(face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.inner.radiative(position)
    }
}

impl<const N: usize, C: Conditions<N>> Conditions<N> for BlockBC<N, C> {
    type System = C::System;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        self.inner.parity(field, face)
    }

    fn radiative(&self, field: Self::System, position: [f64; N]) -> f64 {
        self.inner.radiative(field, position)
    }
}

// /// Transformer which attaches a boundary to a condition or domain.
// #[derive(Clone)]
// pub struct AttachBoundary<I, B> {
//     pub inner: I,
//     pub boundary: B,
// }

// impl<const N: usize, I: Condition<N>, B: Clone> Condition<N> for AttachBoundary<I, B> {
//     fn parity(&self, face: Face<N>) -> bool {
//         self.inner.parity(face)
//     }

//     fn radiative(&self, position: [f64; N]) -> f64 {
//         self.inner.radiative(position)
//     }
// }

// impl<const N: usize, I: Domain<N> + Clone, B: Clone> Domain<N> for AttachBoundary<I, B> {
//     fn bounds(&self) -> Rectangle<N> {
//         self.inner.bounds()
//     }
// }

// impl<const N: usize, I: Clone, B: Boundary<N>> Boundary<N> for AttachBoundary<I, B> {
//     const GHOST: usize = B::GHOST;

//     fn kind(&self, face: Face<N>) -> BoundaryKind {
//         self.boundary.kind(face)
//     }
// }

// /// Transformer which attaches a boundary to a condition or domain.
// #[derive(Clone)]
// pub struct AttachCondition<I, C> {
//     pub inner: I,
//     pub condition: C,
// }

// impl<const N: usize, I: Clone, C: Condition<N>> Condition<N> for AttachCondition<I, C> {
//     fn parity(&self, face: Face<N>) -> bool {
//         self.condition.parity(face)
//     }

//     fn radiative(&self, position: [f64; N]) -> f64 {
//         self.condition.radiative(position)
//     }
// }

// impl<const N: usize, I: Domain<N>, C: Clone> Domain<N> for AttachCondition<I, C> {
//     fn bounds(&self) -> Rectangle<N> {
//         self.inner.bounds()
//     }
// }

// impl<const N: usize, I: Boundary<N>, C> Boundary<N> for AttachCondition<I, C> {
//     const GHOST: usize = I::GHOST;

//     fn kind(&self, face: Face<N>) -> BoundaryKind {
//         self.inner.kind(face)
//     }
// }

#[derive(Clone)]
pub struct BC<B, C> {
    pub boundary: B,
    pub condition: C,
}

impl<B, C> BC<B, C> {
    pub fn new(boundary: B, condition: C) -> Self {
        Self {
            boundary,
            condition,
        }
    }
}

impl<const N: usize, B: Boundary<N>, C: Clone> Boundary<N> for BC<B, C> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        self.boundary.kind(face)
    }
}

impl<const N: usize, B: Clone, C: Condition<N>> Condition<N> for BC<B, C> {
    fn parity(&self, face: Face<N>) -> bool {
        self.condition.parity(face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.condition.radiative(position)
    }
}

impl<const N: usize, B: Clone, C: Conditions<N>> Conditions<N> for BC<B, C> {
    type System = C::System;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        self.condition.parity(field, face)
    }

    fn radiative(&self, field: Self::System, position: [f64; N]) -> f64 {
        self.condition.radiative(field, position)
    }
}

#[derive(Clone)]
pub struct DomainBC<D, I> {
    pub domain: D,
    pub inner: I,
}

impl<D, I> DomainBC<D, I> {
    pub fn new(domain: D, inner: I) -> Self {
        Self { domain, inner }
    }
}

impl<const N: usize, D: Domain<N>, I: Clone> Domain<N> for DomainBC<D, I> {
    fn bounds(&self) -> Rectangle<N> {
        self.domain.bounds()
    }
}

impl<const N: usize, I: Clone, BC: Boundary<N>> Boundary<N> for DomainBC<I, BC> {
    fn kind(&self, face: Face<N>) -> BoundaryKind {
        self.inner.kind(face)
    }
}

impl<const N: usize, I: Clone, BC: Condition<N>> Condition<N> for DomainBC<I, BC> {
    fn parity(&self, face: Face<N>) -> bool {
        self.inner.parity(face)
    }

    fn radiative(&self, position: [f64; N]) -> f64 {
        self.inner.radiative(position)
    }
}

impl<const N: usize, B: Clone, C: Conditions<N>> Conditions<N> for DomainBC<B, C> {
    type System = C::System;

    fn parity(&self, field: Self::System, face: Face<N>) -> bool {
        self.inner.parity(field, face)
    }

    fn radiative(&self, field: Self::System, position: [f64; N]) -> f64 {
        self.inner.radiative(field, position)
    }
}

// /// Transformer that combines a boundary and a condition.
// pub struct BoundaryCondition<B, C> {
//     pub boundary: B,
//     pub condition: C,
// }

// impl<B, C> BoundaryCondition<B, C> {
//     pub fn new(boundary: B, condition: C) -> Self {
//         Self {
//             boundary,
//             condition,
//         }
//     }
// }

// impl<const N: usize, B: Boundary<N>, C> Boundary<N> for BoundaryCondition<B, C> {
//     const GHOST: usize = B::GHOST;

//     fn kind(&self, face: Face<N>) -> BoundaryKind {
//         self.boundary.kind(face)
//     }
// }

// impl<const N: usize, B, C: Condition<N>> Condition<N> for BoundaryCondition<B, C> {
//     fn parity(&self, face: Face<N>) -> bool {
//         self.condition.parity(face)
//     }

//     fn radiative(&self, position: [f64; N]) -> f64 {
//         self.condition.radiative(position)
//     }
// }
