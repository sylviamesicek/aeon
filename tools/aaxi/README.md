# `aaxi`: Axisymmetric Numerical Relavitity

[![Latest version](https://img.shields.io/crates/v/aaxi.svg)](https://crates.io/crates/aaxi)
[![Documentation](https://docs.rs/aaxi/badge.svg)](https://docs.rs/aaxi)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/emilk/egui/blob/main/LICENSE-MIT)
[![Apache](https://img.shields.io/badge/license-Apache-blue.svg)](https://github.com/emilk/egui/blob/main/LICENSE-APACHE)


`aaxi` implements an formulation derived from the axisymmetric evolution scheme of Rinne 2006 to second-order in space, first-order in time using `aeon-tk`. This scheme is purely hyperbolic during evolution, and solves for initial data using a hyperbolic relaxation solver (modelled after NRPyElliptic's solver). This is significantly more complex and numerically expensive than spherical symmetry.