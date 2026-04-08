# `asphere`: Spherically Symmetric Numerical Relavitity

[![Latest version](https://img.shields.io/crates/v/asphere.svg)](https://crates.io/crates/asphere)
[![Documentation](https://docs.rs/asphere/badge.svg)](https://docs.rs/asphere)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/emilk/egui/blob/main/LICENSE-MIT)
[![Apache](https://img.shields.io/badge/license-Apache-blue.svg)](https://github.com/emilk/egui/blob/main/LICENSE-APACHE)

`asphere` implements the evolution scheme of Baumgarte and Shapiro 2007 Chapter 8.4 using `aeon-tk`. Namely it implements equations of motion derived from the Lagrangian, and solving for lapse (α) and the conformal factor (ψ) using a spatial RK4 integrator (this utilizes the simplicity of the equations and spherical symmetry to solve the elliptic constraints very efficiently).
