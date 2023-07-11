export Bases

module Bases

# Dependencies
using LinearAlgebra
using StaticArrays

using Aeon
using Aeon.Geometry

# Includes
include("basis.jl")
include("lagrange.jl")
include("block.jl")

end