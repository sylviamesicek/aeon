# Aeon

`Aeon` is a rust package for solving N-dimensional elliptic and hyperbolic PDEs using the finite difference method and Adaptive Mesh Refinement (AMR).

## Capabilities

- Gradient and Hessian operators up to 6th order accuracy
- Nonuniform quadtree based meshes
- Ghost node filling
    - Coarse-fine interpolating
    - Direct Injection
- Strongly enforced boundary conditions
    - Parity
    - Diritchlet
- Weakly enforced boundary conditions
    - Radiative Sommerfeld
- Adaptive mesh refinement
    - Refine flags
    - Coarsening flags
    - 2:1 balancing
    - Interpolate function between regridding
- Adaptive error heuristics
    - Wavelet-based interpolating basis functions
- Coupled systems, operators, and functions
    - System API with custom `#[derive]` macros
- Method of Lines solver for hyperbolic equations
    - Forward Euler
    - Runge-Kutta 4
- Hyperbolic relaxation solver for elliptic equations
    - Adaptive CFL stepping to accelerate relaxation
- Saving and loading checkpoints of meshes and systems
- Output to `.vtu` files (for viewing in ParaView or VisIt)

## Tools

The bulk of "interesting GR code" lies in the two binary crates `tools/aaxi` and `tools/asphere`. These implement necessary the initial data and evolution code for axisymmetric spacetimes and spherically symmetric spacetimes respectively. Both codebases pull settings from template config files, which support bash-style positional argument references to enable run-time injection of arguments. Example config scripts can be found in the `config/` subdirectory.

### `asphere`

`asphere` implements the evolution scheme of Baumgarte and Shapiro 2007 Chapter 8.4. Namely using equations of motion derived from the Lagrangian, and solving for lapse (α) and the conformal factor (ψ) using a spatial RK4 integrator (this utilizes the simplicity of the equations and spherical symmetry to solve the elliptic constraints very efficiently).

Example `asphere` invokation:
```bash
# Simulates a single massless scalar field. Amplitude argument is passed in
# as first positional argument, and referenced in the config file as `$0`.
cargo run --release --package asphere -- --config="config/sphgauss1.toml" 0.3
```
To run `asphere` in a mode compatible with Cole's critical search code use
```bash
# $0 = 0.3 (amplitude), $1 = 1234 (searial_id)
cargo run --release --package asphere -- --config="config/sphgauss1-cole.toml" 0.3 1234
```

### `aaxi`

`aaxi` adapts the axisymmetric evolution scheme of Rinne 2006 to second-order in space, first-order in time. This scheme is purely hyperbolic during evolution, and solves for initial data using a hyperbolic relaxation solver (modelled after NRPyElliptic's solver). This is significantly more complex and numerically expensive than spherical symmetry. This code also supports critical searches directly via the `search` execution mode.
```toml
# ...
[execution]
mode = "search"
parameter = "amplitude"
start = 0.1
# etc
```

Example `aaxi` invokation:
```bash
# Performs a critical search for a single massless scalar field between
# amplitudes 0.0 and 0.5.
cargo run --release --package aaxi -- --config="config/axiscalar-crit.toml"
```


<!-- [![Build Status](https://github.com/lukazmm/Aeon.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lukazmm/Aeon.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lukazmm/Aeon.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lukazmm/Aeon.jl) -->