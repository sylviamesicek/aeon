module Aeon

using LinearAlgebra
using StaticArrays

# Includes
include("transform.jl")
include("analytic.jl")
include("basis.jl")

# Submodules
include("Space/Space.jl")
include("Engine/Engine.jl")
include("Refine/Refine.jl")

end
