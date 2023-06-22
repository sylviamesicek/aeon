export Approx

module Approx

# Dependencies

using LinearAlgebra
using StaticArrays

# Modules
using Aeon.Geometry
using Aeon.Analytic

# Includes
include("function.jl")
include("square.jl")
include("wls.jl")
# include("mls.jl")

end