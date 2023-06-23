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
include("cell.jl")
include("mesh.jl")
include("system.jl")
include("writer.jl")

    
end