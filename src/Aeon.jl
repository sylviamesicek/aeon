module Aeon

using LinearAlgebra
using StaticArrays

# Includes
include("transform.jl")

# Submodules
include("Analytic/Analytic.jl")
include("Space/Space.jl")
include("Engine/Engine.jl")
include("Refine/Refine.jl")

end
