export Analytic

module Analytic

# Dependencies
using StaticArrays

# Includes
# include("function.jl")
include("field.jl")
include("operator.jl")
include("basis.jl")

include("monomial.jl")
include("gaussian.jl")

# Aliases

export ∇, ∇²

const ∇ = derivative
const ∇² = curvature

end