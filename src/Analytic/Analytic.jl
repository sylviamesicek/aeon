export Analytic

module Analytic

# Dependencies
using StaticArrays

# Other Modules
using Aeon

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