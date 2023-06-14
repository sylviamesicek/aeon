export Method

"""
Contains functions and types common to all methods. Including Meshs, Fields, IO, Function Basis, and stencil approximation.
"""
module Method

# Dependencies

using LinearAlgebra
using StaticArrays

# Other Modules
using Aeon.Tensor
using Aeon.Analytic

##########################
## Includes ##############
##########################

include("mesh.jl")
include("writer.jl")

end