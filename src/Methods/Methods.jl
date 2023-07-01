export Methods

"""
Implements actual numerical method.
"""
module Methods

# Dependencies
using LinearAlgebra
using StaticArrays

# Submodules
using Aeon
using Aeon.Geometry
using Aeon.Operators

# Includes
include("mesh.jl")
include("dofs.jl")
include("interface.jl")

include("writer.jl")

    
end