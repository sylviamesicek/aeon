export Analytic

module Analytic

# Dependencies
using LinearAlgebra
using StaticArrays

# Other Modules
using Aeon.Tensor

# Includes
include("function.jl")
include("covariant.jl")

include("monomial.jl")
include("gaussian.jl")
include("basis.jl")

end