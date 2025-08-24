use aeon::prelude::*;

pub const GRR_CH: usize = 0;
pub const GRZ_CH: usize = 1;
pub const GZZ_CH: usize = 2;
pub const S_CH: usize = 3;
pub const KRR_CH: usize = 4;
pub const KRZ_CH: usize = 5;
pub const KZZ_CH: usize = 6;
pub const Y_CH: usize = 7;
pub const THETA_CH: usize = 8;
pub const ZR_CH: usize = 9;
pub const ZZ_CH: usize = 10;
pub const LAPSE_CH: usize = 11;
pub const SHIFTR_CH: usize = 12;
pub const SHIFTZ_CH: usize = 13;

pub const fn phi_ch(i: usize) -> usize {
    14 + 2 * i
}

pub const fn pi_ch(i: usize) -> usize {
    14 + 2 * i + 1
}

pub const fn num_channels(scalar_fields: usize) -> usize {
    14 + 2 * scalar_fields
}

pub fn save_image(checkpoint: &mut Checkpoint<2>, image: ImageRef) {
    checkpoint.save_field("Grr", image.channel(GRR_CH));
    checkpoint.save_field("Grz", image.channel(GRZ_CH));
    checkpoint.save_field("Gzz", image.channel(GZZ_CH));
    checkpoint.save_field("S", image.channel(S_CH));
    checkpoint.save_field("Krr", image.channel(KRR_CH));
    checkpoint.save_field("Krz", image.channel(KRZ_CH));
    checkpoint.save_field("Kzz", image.channel(KZZ_CH));
    checkpoint.save_field("Y", image.channel(Y_CH));
    checkpoint.save_field("Theta", image.channel(THETA_CH));
    checkpoint.save_field("Zr", image.channel(ZR_CH));
    checkpoint.save_field("Zz", image.channel(ZZ_CH));
    checkpoint.save_field("Lapse", image.channel(LAPSE_CH));
    checkpoint.save_field("Shiftr", image.channel(SHIFTR_CH));
    checkpoint.save_field("Shiftz", image.channel(SHIFTZ_CH));

    let num_scalar_fields = (image.num_channels() - 14) / 2;

    for i in 0..num_scalar_fields {
        checkpoint.save_field(&format!("Phi{i}"), image.channel(phi_ch(i)));
        checkpoint.save_field(&format!("Pi{i}"), image.channel(phi_ch(i)));
    }
}

/// Boundary conditions for various fields.
#[derive(Clone)]
pub struct FieldConditions;

impl SystemBoundaryConds<2> for FieldConditions {
    fn kind(&self, channel: usize, face: Face<2>) -> BoundaryKind {
        if face.side {
            return BoundaryKind::Radiative;
        }

        let s00 = [BoundaryKind::AntiSymmetric, BoundaryKind::AntiSymmetric];
        let s10 = [BoundaryKind::Symmetric, BoundaryKind::AntiSymmetric];
        let s01 = [BoundaryKind::AntiSymmetric, BoundaryKind::Symmetric];
        let s11 = [BoundaryKind::Symmetric, BoundaryKind::Symmetric];

        let axes = match channel {
            GRR_CH | KRR_CH => s11,
            GRZ_CH | KRZ_CH => s00,
            GZZ_CH | KZZ_CH => s11,
            S_CH | Y_CH => s01,

            THETA_CH | LAPSE_CH => s11,
            ZR_CH | SHIFTR_CH => s01,
            ZZ_CH | SHIFTZ_CH => s10,

            _ => s11,
        };
        axes[face.axis]
    }

    fn radiative(&self, channel: usize, _position: [f64; 2]) -> RadiativeParams {
        match channel {
            GRR_CH | GZZ_CH | LAPSE_CH => RadiativeParams::lightlike(1.0),
            _ => RadiativeParams::lightlike(0.0),
        }
    }
}
