export Operators

"""
A module for handling numerical operators in a mesh-independent way. This provides abstracts like, `Block`s, `Field`s
and `Domain`s for treating numerical problems, and computing values, gradients, and hessians of functions. 
"""
module Operators

# Dependencies
using LinearAlgebra
using StaticArrays

using Aeon
using Aeon.Geometry

# Includes
include("basis.jl")
include("lagrange.jl")
include("block.jl")
include("domain.jl")

end