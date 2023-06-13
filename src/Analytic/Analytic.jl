export Analytic

module Analytic

# Dependencies
using StaticArrays

# Includes
include("function.jl")
include("covariant.jl")

include("monomial.jl")
include("gaussian.jl")
include("basis.jl")

# Aliases

export ∇, ∇²

const ∇ = derivative
const ∇² = curvature

end