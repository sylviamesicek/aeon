# Aeon

`Aeon` is a rust package for solving N-dimensional elliptic and hyperbolic PDEs using the finite difference method and Adaptive Mesh Refinement (AMR).

## To Do

To reach the state of minimum viable product (MVP) the following feature(s) must be completed.

- [x] Stencil application.
- [x] Strong boundary condition API.
- [ ] Weakly enforced (i.e. outgoing) boundary condition API.
- [ ] Adaptive quadtree based meshes
- [ ] Transfer between regridding runs.
- [ ] Restriction/Prolongation.
- [x] Linear iterative solvers.
    - [x] BiCGStab
    - [ ] BiCGStab(l)
- [x] ODE Integrators.
    - [x] Forward Euler
    - [x] RK4
- [x] Hyperbolic Relaxation solver for elliptic equations.
    - [ ] Adaptive CFL scaling for faster convergence.
- [x] Method of Lines solver for hyperbolic equations.
- [x] VTK Output.


<!-- [![Build Status](https://github.com/lukazmm/Aeon.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lukazmm/Aeon.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lukazmm/Aeon.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lukazmm/Aeon.jl) -->
