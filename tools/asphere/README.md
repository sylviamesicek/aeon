# `asphere`: Spherically Symmetric Numerical Relavitity

[![Latest version](https://img.shields.io/crates/v/asphere.svg)](https://crates.io/crates/asphere)
[![Documentation](https://docs.rs/asphere/badge.svg)](https://docs.rs/asphere)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/emilk/egui/blob/main/LICENSE-MIT)
[![Apache](https://img.shields.io/badge/license-Apache-blue.svg)](https://github.com/emilk/egui/blob/main/LICENSE-APACHE)

`asphere` implements the evolution scheme of Baumgarte and Shapiro 2007 Chapter 8.4 using `aeon-tk`. Namely it implements equations of motion derived from the Lagrangian, and solving for lapse (α) and the conformal factor (ψ) using a spatial RK4 integrator (this utilizes the simplicity of the equations and spherical symmetry to solve the elliptic constraints very efficiently).

## Installation

`asphere` can be used via Rust's package manager `cargo`, either by cloning this repository and executing
```bash
# Clone Repo
git clone https://github.com/sylviamesicek/aeon.git
# Build and run asphere from source
cargo run --release --package asphere -- <arguments>
```
or directly installed as a binary
```bash
# Install asphere via cargo
cargo install asphere
# Execute
asphere <arguments>
```

## Usage
Configuration like initial data and evolution parameters for asphere are provided by a set of `.toml` files (examples of these files can be found in the [`config/`](https://github.com/sylviamesicek/aeon/tree/master/config) directory). Executing 
```bash
asphere run simulation
```
will run a simulation using parameters set in `simulation.toml`. Certain arguments support bash-style variable references, allowing values to be passed in through the command-line:
```toml
# In config/sphgauss1.toml
# ...
[[sources]]
mass = 0.0
profile.type = "gaussian"
profile.amplitude = "${amplitude}"
profile.sigma = 5.35
profile.center = 0.0
# ...
```
which can then be executed as
```bash
# Simulates a massless gaussian scalar field with amplitude 0.3
asphere run -Damplitude=0.3 config/sphgauss1
```

Other commands like `search` and `fill` require parameters to be specified in `<name>.search.toml` and `<name>.fill.toml` respectively, in addition to the simulation parameters set in `<name>.toml`. For example, running
```bash
asphere search config/sphgauss1
```
with the following search configuration
```toml
# In config/sphgauss1.search.toml
directory = "output/sphgauss1/search"
parameter = "amplitude"
parallel = 9
start = 0.3
end = 0.4
max_depth = 40
min_error = 1e-16
```
runs 9 simulations between 0.3 and 0.4, determines which simulations collapse and which disperse and recursively updates the start/end values in order to find the "critical amplitude" separating these two states.