// use std::path::PathBuf;

// Global imports
use aeon::{common::BoundarySet, prelude::*};

// Submodules
mod config;

use config::*;

// **********************************
// Settings

pub struct Dissipation<'a, Rho: BoundarySet<2>, Z: BoundarySet<2>> {
    src: &'a [f64],
    rho: &'a Rho,
    z: &'a Z,
}

impl<'a, Rho: BoundarySet<2>, Z: BoundarySet<2>> Projection<2> for Dissipation<'a, Rho, Z> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        // Bizarre rust analyzer warning.
        let mut _f = dest;

        let diss_r = arena.alloc(block.len());
        let diss_z = arena.alloc(block.len());
        let src = block.auxillary(self.src);

        block.dissipation::<ORDER>(0, self.rho, src, diss_r);
        block.dissipation::<ORDER>(1, self.z, src, diss_z);

        for (i, _) in block.iter().enumerate() {
            _f[i] = DISS * (diss_r[i] + diss_z[i]);
        }
    }
}

/// Initial data (solved for before evolution).
#[derive(Clone, Debug)]
pub struct Initial;

impl SystemLabel for Initial {
    const NAME: &'static str = "Initial";
    const FIELDS: usize = 1;

    fn from_index(_: usize) -> Self {
        Initial
    }

    fn field_index(&self) -> usize {
        0
    }

    fn field_name(&self) -> String {
        "psi".into()
    }
}

#[derive(Clone, Debug)]
pub enum Evolution {
    Grr,
    Grz,
    Gzz,
    S,

    Krr,
    Krz,
    Kzz,
    Y,

    Theta,
    Zr,
    Zz,

    Lapse,
    Shiftr,
    Shiftz,
}

impl SystemLabel for Evolution {
    const NAME: &'static str = "Evolution";
    const FIELDS: usize = 14;

    fn from_index(idx: usize) -> Self {
        match idx {
            0 => Evolution::Grr,
            1 => Evolution::Grz,
            2 => Evolution::Gzz,
            3 => Evolution::S,
            4 => Evolution::Krr,
            5 => Evolution::Krz,
            6 => Evolution::Kzz,
            7 => Evolution::Y,
            8 => Evolution::Theta,
            9 => Evolution::Zr,
            10 => Evolution::Zz,
            11 => Evolution::Lapse,
            12 => Evolution::Shiftr,
            13 => Evolution::Shiftz,
            _ => panic!("Invalid Index"),
        }
    }

    fn field_index(&self) -> usize {
        match self {
            Evolution::Grr => 0,
            Evolution::Grz => 1,
            Evolution::Gzz => 2,
            Evolution::S => 3,
            Evolution::Krr => 4,
            Evolution::Krz => 5,
            Evolution::Kzz => 6,
            Evolution::Y => 7,
            Evolution::Theta => 8,
            Evolution::Zr => 9,
            Evolution::Zz => 10,
            Evolution::Lapse => 11,
            Evolution::Shiftr => 12,
            Evolution::Shiftz => 13,
        }
    }

    fn field_name(&self) -> String {
        match self {
            Evolution::Grr => "grr",
            Evolution::Grz => "grz",
            Evolution::Gzz => "gzz",
            Evolution::S => "s",
            Evolution::Krr => "krr",
            Evolution::Krz => "krz",
            Evolution::Kzz => "kzz",
            Evolution::Y => "y",
            Evolution::Theta => "theta",
            Evolution::Zr => "zr",
            Evolution::Zz => "zz",
            Evolution::Lapse => "lapse",
            Evolution::Shiftr => "shiftr",
            Evolution::Shiftz => "shiftz",
        }
        .into()
    }
}

pub fn main() {
    env_logger::builder()
        .format_timestamp(None)
        .filter_level(log::LevelFilter::max())
        .init();

    // Scratch allocator
    // let mut arena = Arena::new();

    let mesh = UniformMesh::new(
        Rectangle {
            size: [RADIUS, RADIUS],
            origin: [0.0, 0.0],
        },
        [8, 8],
        LEVELS,
    );

    log::info!("Node Count: {}", mesh.node_count());
}
