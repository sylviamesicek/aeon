export Analytic

module Analytic

# Dependencies
using LinearAlgebra
using StaticArrays

# Inludes

include("field.jl")
include("operator.jl")
include("monomial.jl")
include("gaussian.jl")
include("basis.jl")

end