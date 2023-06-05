module Aeon

using LinearAlgebra
using StaticArrays

# Includes
include("splitarray.jl")
include("stree.jl")
include("transform.jl")

# Submodules
include("Analytic/Analytic.jl")
include("Method/Method.jl")
include("Grid/Grid.jl")
include("Refine/Refine.jl")

end
