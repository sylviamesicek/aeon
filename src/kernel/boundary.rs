//! Provides generalized interfaces for expressing various kinds of boundary condition.
//!
//! This module uses a combination of trait trickery and type transformers to
//! create an ergonomic API for working with boundaries.

use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use crate::{array::ArrayWrap, geometry::Face};

/// Indicates what type of boundary condition is used along a particualr
/// face of the domain. More specific boundary conditions are provided
/// by the `Condition` API, but for many funtions, `Boundary` provides
/// enough information to compute supports and apply stencils.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BoundaryKind {
    /// Symmetric Boundary condition. Function is even along axis
    /// (so function values are reflected across the axis with the same sign)
    Symmetric,
    /// Antisymmetric Boundary condition. Function is even along axis
    /// (so function values are reflected across the axis with the opposite sign,
    /// on axis set to zero)
    AntiSymmetric,
    /// This boundary condition indicates that the ghost nodes have been filled manually via some external
    /// process. This can be used to implement custom boundary conditions, and is primarily used to
    /// fill inter-grid boundaries in the adaptive mesh refinement driver.
    Custom,
    /// The boundary condition is implemented via sommerfeld radiative boundary conditions.
    Radiative,
    #[default]
    Free,
    /// Boundary is set to a given value, lopsided stencils are used near the boundary. This
    /// condition can either be strongly enforced (values of systems are set at boundary before
    /// application) or weakly enforced (condition is applied to time derivative of system).
    StrongDirichlet,
    WeakDirichlet,
}

impl BoundaryKind {
    /// Are ghost nodes are used when enforcing this kind of boundary condition?
    pub fn needs_ghost(self) -> bool {
        match self {
            BoundaryKind::AntiSymmetric | BoundaryKind::Symmetric | BoundaryKind::Custom => true,
            BoundaryKind::Radiative
            | BoundaryKind::Free
            | BoundaryKind::StrongDirichlet
            | BoundaryKind::WeakDirichlet => false,
        }
    }

    /// Boundary class corresponding to this kind of boundary condition.
    pub fn class(self) -> BoundaryClass {
        match self {
            BoundaryKind::AntiSymmetric | BoundaryKind::Symmetric | BoundaryKind::Custom => {
                BoundaryClass::Ghost
            }
            BoundaryKind::Radiative
            | BoundaryKind::StrongDirichlet
            | BoundaryKind::WeakDirichlet
            | BoundaryKind::Free => BoundaryClass::OneSided,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BoundaryClass {
    /// Boundary condition implemented by invoking one-sided stencils near the boundary.
    #[default]
    OneSided = 0,
    /// Boundary condition implemented by setting ghost nodes to fixed values.
    Ghost = 1,
    /// Boundary where data is read from the opposite side of the domain, allowing data
    /// to tile infinitely along this axis.
    Periodic = 2,
}

impl BoundaryClass {
    /// Does this boundary class depend on setting ghost node values.
    pub fn has_ghost(self) -> bool {
        matches!(self, BoundaryClass::Ghost | BoundaryClass::Periodic)
    }
}

impl Distribution<BoundaryClass> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BoundaryClass {
        match rng.random_range(0..3) {
            0 => BoundaryClass::OneSided,
            1 => BoundaryClass::Ghost,
            2 => BoundaryClass::Periodic,
            _ => unreachable!(),
        }
    }
}

/// An array storing a value for each `Face<N>` in a N-dimensional space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryClasses<const N: usize>([[BoundaryClass; 2]; N]);

impl<const N: usize> BoundaryClasses<N> {
    pub const GHOST: Self = Self::splat(BoundaryClass::Ghost);
    pub const ONE_SIDED: Self = Self::splat(BoundaryClass::OneSided);
    pub const PERIODIC: Self = Self::splat(BoundaryClass::Periodic);

    /// Constructs a `FaceArray<N>` by calling `f` for each `Face<N>`.
    pub fn from_fn<F: FnMut(Face<N>) -> BoundaryClass>(mut f: F) -> Self {
        Self(core::array::from_fn(|axis| {
            [f(Face::negative(axis)), f(Face::positive(axis))]
        }))
    }

    /// Retrieves the inner representation of `FaceArray`, i.e. an array of type
    /// `[[T; 2]; N]` where the first index is axis and the second index is size.
    pub const fn into_inner(self) -> [[BoundaryClass; 2]; N] {
        self.0
    }

    /// Constructs a `FaceArray` by filling the whole array with `value`.
    pub const fn splat(value: BoundaryClass) -> Self {
        Self([[value; 2]; N])
    }

    pub fn from_sides(negative: [BoundaryClass; N], positive: [BoundaryClass; N]) -> Self {
        Self::from_fn(|face| match face.side {
            true => positive[face.axis].clone(),
            false => negative[face.axis].clone(),
        })
    }

    pub fn is_compatible<B: Boundary<N>>(&self, boundary: &B, num_channels: usize) -> bool {
        (0..num_channels).all(|channel| {
            Face::iterate().all(|face| boundary.kind(channel, face).class() == self[face])
        })
    }
}

impl<const N: usize> From<[[BoundaryClass; 2]; N]> for BoundaryClasses<N> {
    fn from(value: [[BoundaryClass; 2]; N]) -> Self {
        Self(value)
    }
}

impl<const N: usize> From<[(BoundaryClass, BoundaryClass); N]> for BoundaryClasses<N> {
    fn from(value: [(BoundaryClass, BoundaryClass); N]) -> Self {
        Self(value.map(|(l, r)| [l, r]))
    }
}

impl<const N: usize> serde::Serialize for BoundaryClasses<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        ArrayWrap(self.0.clone()).serialize(serializer)
    }
}

impl<'de, const N: usize> serde::de::Deserialize<'de> for BoundaryClasses<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(BoundaryClasses(ArrayWrap::deserialize(deserializer)?.0))
    }
}

impl<const N: usize> Default for BoundaryClasses<N> {
    fn default() -> Self {
        Self::from_fn(|_| BoundaryClass::default())
    }
}

impl<const N: usize> std::ops::Index<Face<N>> for BoundaryClasses<N> {
    type Output = BoundaryClass;
    fn index(&self, index: Face<N>) -> &Self::Output {
        &self.0[index.axis][index.side as usize]
    }
}

impl<const N: usize> std::ops::IndexMut<Face<N>> for BoundaryClasses<N> {
    fn index_mut(&mut self, index: Face<N>) -> &mut Self::Output {
        &mut self.0[index.axis][index.side as usize]
    }
}

/// Describes a radiative boundary condition at a point on the boundary.
#[derive(Clone, Copy, Debug)]
pub struct RadiativeParams {
    /// Target value for field.
    pub target: f64,
    /// Wavespeed of field at boundary.
    pub speed: f64,
}

impl RadiativeParams {
    /// Constructs a boundary condition for a wave asymptotically approaching a given value, travelling
    /// at the speed of light (c = 1).
    pub fn lightlike(target: f64) -> Self {
        Self { target, speed: 1.0 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DirichletParams {
    /// Target value for the field.
    pub target: f64,
    /// Scale factor for "spring" drawing function to this value.
    pub strength: f64,
}

// /// Provides specifics for enforcing boundary conditions for
// /// a particular field.
// pub trait BoundaryConds<const N: usize>: Clone {
//     fn kind(&self, _face: Face<N>) -> BoundaryKind;

//     fn radiative(&self, _position: [f64; N]) -> RadiativeParams {
//         RadiativeParams {
//             target: 0.0,
//             speed: 1.0,
//         }
//     }

//     fn dirichlet(&self, _position: [f64; N]) -> DirichletParams {
//         DirichletParams {
//             target: 0.0,
//             strength: 1.0,
//         }
//     }
// }

// /// Checks whether a set of boundary conditions are compatible with the given ghost flags.
// pub fn is_boundary_compatible<const N: usize, B: BoundaryConds<N>>(
//     boundary: &FaceArray<N, BoundaryClass>,
//     conditions: &B,
// ) -> bool {
//     Face::iterate().all(|face| conditions.kind(face).class() == boundary[face])
// }

// /// A generalization of `Condition<N>` for a coupled systems of scalar fields.
// pub trait ImageBoundaryConds<const N: usize>: Clone {
//     fn kind(&self, channel: usize, _face: Face<N>) -> BoundaryKind;

//     fn radiative(&self, _channel: usize, _position: [f64; N]) -> RadiativeParams {
//         RadiativeParams {
//             target: 0.0,
//             speed: 1.0,
//         }
//     }

//     fn dirichlet(&self, _channel: usize, _position: [f64; N]) -> DirichletParams {
//         DirichletParams {
//             target: 0.0,
//             strength: 1.0,
//         }
//     }

//     // fn channel(&self, channel: usize) -> ChannelBoundaryConds<N, Self> {
//     //     ChannelBoundaryConds::new(self.clone(), channel)
//     // }
// }

pub trait Boundary<const N: usize>: Clone {
    fn kind(&self, channel: usize, face: Face<N>) -> BoundaryKind;

    fn radiative(&self, _channel: usize, _position: [f64; N]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }

    fn dirichlet(&self, _channel: usize, _position: [f64; N]) -> DirichletParams {
        DirichletParams {
            target: 0.0,
            strength: 1.0,
        }
    }
}

// /// Transfers a set of `Conditions<N>` into a single `Condition<N>` by only applying the set of conditions
// /// to a single field.
// pub struct ChannelBoundaryConds<const N: usize, C> {
//     conditions: C,
//     channel: usize,
// }

// impl<const N: usize, C: ImageBoundaryConds<N>> ChannelBoundaryConds<N, C> {
//     pub const fn new(conditions: C, channel: usize) -> Self {
//         Self {
//             channel,
//             conditions,
//         }
//     }
// }

// impl<const N: usize, C: ImageBoundaryConds<N>> Clone for ChannelBoundaryConds<N, C> {
//     fn clone(&self) -> Self {
//         Self {
//             conditions: self.conditions.clone(),
//             channel: self.channel,
//         }
//     }
// }

// impl<const N: usize, C: ImageBoundaryConds<N>> BoundaryConds<N> for ChannelBoundaryConds<N, C> {
//     fn kind(&self, face: Face<N>) -> BoundaryKind {
//         self.conditions.kind(self.channel, face)
//     }

//     fn radiative(&self, position: [f64; N]) -> RadiativeParams {
//         self.conditions.radiative(self.channel, position)
//     }

//     fn dirichlet(&self, position: [f64; N]) -> DirichletParams {
//         self.conditions.dirichlet(self.channel, position)
//     }
// }

// ****************************
// Specializations ************
// ****************************

// /// Transforms a single condition into a set of `Conditions<N>` where `Self::System = Scalar`.
// #[derive(Clone)]
// pub struct ScalarConditions<I>(pub I);

// impl<I> ScalarConditions<I> {
//     pub const fn new(inner: I) -> Self {
//         Self(inner)
//     }
// }

// impl<const N: usize, I: BoundaryConds<N>> ImageBoundaryConds<N> for ScalarConditions<I> {
//     fn kind(&self, channel: usize, face: Face<N>) -> BoundaryKind {
//         debug_assert!(channel == 0);
//         self.0.kind(face)
//     }

//     fn radiative(&self, channel: usize, position: [f64; N]) -> RadiativeParams {
//         debug_assert!(channel == 0);
//         self.0.radiative(position)
//     }

//     fn dirichlet(&self, channel: usize, position: [f64; N]) -> DirichletParams {
//         debug_assert!(channel == 0);
//         self.0.dirichlet(position)
//     }
// }

#[derive(Clone)]
pub struct EmptyBoundary;

impl<const N: usize> Boundary<N> for EmptyBoundary {
    fn kind(&self, _channel: usize, _face: Face<N>) -> BoundaryKind {
        unreachable!()
    }

    fn radiative(&self, _channel: usize, _position: [f64; N]) -> RadiativeParams {
        unreachable!()
    }

    fn dirichlet(&self, _channel: usize, _position: [f64; N]) -> DirichletParams {
        unreachable!()
    }
}
