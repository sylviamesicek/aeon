# Design

First one builds a `Mesh`, which splits a domain into a series of `Cell`s and describes
the overall topology of the domain. With this `Mesh` one can initialize a `Method`, which 
defines degrees of freedom across a domain. A domain is split into a set of "independent"
elements, and a field is defined as the combination of its definitions on each element.
To solve a system, the user must provide:
- A matrix-free version of the lhs operator on this element including boundary conditions
- The rhs of the system on this element including boundary conditions

We provide a `System` supertype to provide such an interface. A system has two functions,
an `apply_system(system, element, field)` function and a `compute_rhs(system, element)