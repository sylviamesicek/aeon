# Aeon

`Aeon` is a rust package for solving N-dimensional elliptic and hyperbolic PDEs using the finite difference method and Adaptive Mesh Refinement (AMR).

## Capabilities

- Gradient and Hessian operators up to 6th order accuracy
- Nonuniform axis-aligned quadtree based meshes
- Inter-cell continuity enforced via ghost nodes
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
- Kreiss-Oliger dissipation
- Finite point method style interpolation on individual cells
- Saving and loading checkpoints of meshes and systems
- Output to `.vtu` files (for viewing in ParaView or VisIt)

## Tools

The bulk of "interesting GR code" lies in the two binary crates `tools/aaxi` and `tools/asphere`. These implement necessary the initial data and evolution code for axisymmetric spacetimes and spherically symmetric spacetimes respectively. Both codebases pull settings from template config files, which support bash-style variable references to enable run-time injection of arguments. Example config scripts can be found in the `config/` subdirectory.

Both `asphere` and `aaxi` can execute a number of different subcommands for more specific usecases. For instance, searching for critical points, or producing mass-fill black hole plots. These subcommands follow the same convention as the standard `run` subcommand, but require additional configuration files. For instance, running a fill command such as `asphere fill <name>` requires there to exist a `<name>.toml` file (storing basic run configuration data) and a `<name>.fill.toml` file containing fill parameters.

### `asphere`

`asphere` implements the evolution scheme of Baumgarte and Shapiro 2007 Chapter 8.4. Namely using equations of motion derived from the Lagrangian, and solving for lapse (α) and the conformal factor (ψ) using a spatial RK4 integrator (this utilizes the simplicity of the equations and spherical symmetry to solve the elliptic constraints very efficiently).

Example `asphere` invokation:
```bash
# Simulates a single massless scalar field. Amplitude argument is passed in
# via "amplitude" variable
cargo run --release --package asphere -- run -Damplitude=0.3 config/sphgauss1
```

### `aaxi`

`aaxi` adapts the axisymmetric evolution scheme of Rinne 2006 to second-order in space, first-order in time. This scheme is purely hyperbolic during evolution, and solves for initial data using a hyperbolic relaxation solver (modelled after NRPyElliptic's solver). This is significantly more complex and numerically expensive than spherical symmetry.

Example `aaxi` invokations:
```bash
# Runs a single simulation for a 0.3 amplitude scalar field.
cargo run --release --package aaxi -- run -Damplitude=0.3 config/axigauss
# Performs a critical search for a single massless scalar field between
# amplitudes 0.0 and 0.5.
cargo run --release --package aaxi -- search config/axigauss
```

#### MPI

I recently implemented MPI support for running axisymmetric searches on multiple cluster nodes. This requires the code be compiled with the mpi feature enabled, i.e.
```bash
cargo build --release --package aaxi --features="mpi"
```
and requires some compatible mpi compiler available (on Windows this includes clang binaries and MSMPI). Running output piped through `mpirun` or `mpiexec` breaks progress bar logging, so I recommend switching to incremental logging by adding this line to your config file.
```toml
# ....
[logging]
style = "incremental"
evolve.steps = 1000  # Number of steps between each log during evolution
initial.steps = 1000 # Number of steps between each log during relaxation
# ....
```
An example of running multiple processes on a single computer for testing purposes
```bash
mpiexec -n 3 ./target/release/aaxi search config/axigauss 
```