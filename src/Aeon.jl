module Aeon

# Dependencies
using LinearAlgebra
using StaticArrays
using WriteVTK

# Exports
export Kind, Domain
export Grid, mkdomain
export VtkOutput, attach_function!, write_vtk


# Includes
include("domain.jl")
include("grid.jl")
include("transform.jl")
include("vtk.jl")

end
