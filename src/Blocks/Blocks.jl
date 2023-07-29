export Blocks

"""
API for filling and evaluating stencils on blocks. A block is a set of interior cells defined on the unit
square (0, 1)ⁿ, along with ghost cells along the edges to allow for centered stencil application.
"""
module Blocks
    
using StaticArrays
using LinearAlgebra

using Aeon
using Aeon.Geometry

# Basis
include("basis.jl")
include("lagrange.jl")

# Block
include("block.jl")
include("point.jl")
include("derivative.jl")

# ArrayBlock
include("array.jl")

end