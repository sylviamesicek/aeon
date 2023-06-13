module Aeon

using LinearAlgebra
using StaticArrays

# Includes
include("transform.jl")

# Submodules
include("Analytic/Analytic.jl")
include("Approx/Approx.jl")
include("Method/Method.jl")
include("Grid/Grid.jl")

end
