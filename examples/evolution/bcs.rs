use aeon::common::{
    AntiSymmetricBoundary, AsymptoticFlatness, Boundary, BoundarySet, FreeBoundary, Mixed,
    RobinBoundary, Simple, SymmetricBoundary,
};

use crate::config::*;

pub type AsymptoticOdd =
    Mixed<2, Simple<AntiSymmetricBoundary<{ ORDER + 2 }>>, AsymptoticFlatness<ORDER>>;
pub type AsymptoticEven =
    Mixed<2, Simple<SymmetricBoundary<{ ORDER + 2 }>>, AsymptoticFlatness<ORDER>>;
pub type FreeOdd = Mixed<2, Simple<AntiSymmetricBoundary<{ ORDER + 2 }>>, Simple<FreeBoundary>>;
pub type FreeEven = Mixed<2, Simple<SymmetricBoundary<{ ORDER + 2 }>>, Simple<FreeBoundary>>;

pub const ASYMPTOTIC_ODD_RHO: AsymptoticOdd = Mixed::new(
    Simple::new(AntiSymmetricBoundary),
    AsymptoticFlatness::new(0),
);

pub const ASYMPTOTIC_EVEN_RHO: AsymptoticEven =
    Mixed::new(Simple::new(SymmetricBoundary), AsymptoticFlatness::new(0));

pub const ASYMPTOTIC_ODD_Z: AsymptoticOdd = Mixed::new(
    Simple::new(AntiSymmetricBoundary),
    AsymptoticFlatness::new(1),
);

pub const ASYMPTOTIC_EVEN_Z: AsymptoticEven =
    Mixed::new(Simple::new(SymmetricBoundary), AsymptoticFlatness::new(1));

pub const FREE_ODD: FreeOdd = Mixed::new(
    Simple::new(AntiSymmetricBoundary),
    Simple::new(FreeBoundary),
);

pub const FREE_EVEN: FreeEven =
    Mixed::new(Simple::new(SymmetricBoundary), Simple::new(FreeBoundary));

// ******************************
// Boundary Aliases

// Initial
pub const PSI_INITIAL_RHO: AsymptoticEven = ASYMPTOTIC_EVEN_RHO;
pub const PSI_INITIAL_Z: AsymptoticEven = ASYMPTOTIC_EVEN_Z;
// Gauge
pub const LAPSE_RHO: AsymptoticEven = ASYMPTOTIC_EVEN_RHO;
pub const LAPSE_Z: AsymptoticEven = ASYMPTOTIC_EVEN_Z;
pub const SHIFTR_RHO: AsymptoticOdd = ASYMPTOTIC_ODD_RHO;
pub const SHIFTR_Z: AsymptoticEven = ASYMPTOTIC_EVEN_Z;
pub const SHIFTZ_RHO: AsymptoticEven = ASYMPTOTIC_EVEN_RHO;
pub const SHIFTZ_Z: AsymptoticOdd = ASYMPTOTIC_ODD_Z;
// Dynamic
pub const PSI_RHO: FreeEven = FREE_EVEN;
pub const PSI_Z: FreeEven = FREE_EVEN;
pub const PSI_RHOZ: FreeEven = FREE_EVEN;
pub const SEED_RHO: FreeOdd = FREE_ODD;
pub const SEED_Z: FreeEven = FREE_EVEN;
pub const W_RHO: FreeOdd = FREE_ODD;
pub const W_Z: FreeEven = FREE_EVEN;
pub const U_RHO: FreeEven = FREE_EVEN;
pub const U_Z: FreeEven = FREE_EVEN;
pub const X_RHO: FreeOdd = FREE_ODD;
pub const X_Z: FreeOdd = FREE_ODD;

// Mixed asymptotic boundaries

pub struct AsymptoticFlatnessMixedDerivatve<const ORDER: usize>;

impl<const ORDER: usize> BoundarySet<2> for AsymptoticFlatnessMixedDerivatve<ORDER>
where
    RobinBoundary<ORDER>: Boundary,
{
    type NegativeBoundary = RobinBoundary<ORDER>;
    type PositiveBoundary = RobinBoundary<ORDER>;

    fn negative(&self, position: [f64; 2]) -> Self::NegativeBoundary {
        let mut r2 = 0.0;

        for pos in position.iter() {
            r2 += pos * pos;
        }

        RobinBoundary {
            coefficient: 3.0 * position[0] * position[1] / (r2 * r2),
        }
    }

    fn positive(&self, position: [f64; 2]) -> Self::PositiveBoundary {
        let mut r2 = 0.0;

        for pos in position.iter() {
            r2 += pos * pos;
        }

        RobinBoundary {
            coefficient: 3.0 * position[0] * position[1] / (r2 * r2),
        }
    }
}
