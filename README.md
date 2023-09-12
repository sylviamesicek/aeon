# Aeon

`Aeon` is a zig package for solving N-dimensional elliptic and hyperbolic PDEs using the finite difference method and block-structured Adaptive Mesh Refinement (AMR).

## To Do

To reach the state of minimum viable product (MVP) the following features must be completed.

- [x] Single level cell clustering using the Berger-Rigoutsos alogorithm.
- [x] Patch compatible clustering algorithm extension.
- [x] Multilevel regridding based on GRChombo.
- [x] Stencil application.
- [ ] Transfer between regridding runs.
- [x] Interior ghost node filling.
- [x] Exterior ghost node filling.
- [x] Restriction operator.
- [x] Prolongation operator.
- [ ] Multigrid elliptic solver using BiCGStab and Jacobi's Method.
- [ ] Hyperbolic solver using 4th order Runge-Kutta and subcylcing.

<!-- [![Build Status](https://github.com/lukazmm/Aeon.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lukazmm/Aeon.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lukazmm/Aeon.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lukazmm/Aeon.jl) -->
