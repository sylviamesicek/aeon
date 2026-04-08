# `asphere`: Spherically Numerical Relavitity

`asphere` implements the evolution scheme of Baumgarte and Shapiro 2007 Chapter 8.4 using `aeon-tk`. Namely it implements equations of motion derived from the Lagrangian, and solving for lapse (α) and the conformal factor (ψ) using a spatial RK4 integrator (this utilizes the simplicity of the equations and spherical symmetry to solve the elliptic constraints very efficiently).
