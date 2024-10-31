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
- Saving and loading checkpoints of meshes and systems
- Output to `.vtu` files (for viewing in ParaView or VisIt)

## Future Improvements

- [] Adaptive CFL stepping to accelerating hyperbolic relaxation

<!-- [![Build Status](https://github.com/lukazmm/Aeon.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lukazmm/Aeon.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lukazmm/Aeon.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lukazmm/Aeon.jl) -->