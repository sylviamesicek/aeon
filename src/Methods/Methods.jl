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
# include("element.jl")
include("method.jl")

    
end