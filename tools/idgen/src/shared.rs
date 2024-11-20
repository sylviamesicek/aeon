use aeon::prelude::*;

/// Quadrant upon which all simulations run.
#[derive(Clone)]
pub struct Quadrant;

impl Boundary<2> for Quadrant {
    fn kind(&self, face: Face<2>) -> BoundaryKind {
        if !face.side {
            BoundaryKind::Parity
        } else {
            BoundaryKind::Radiative
        }
    }
}
