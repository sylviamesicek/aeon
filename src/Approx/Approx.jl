export Approx

module Approx

# Dependencies

using LinearAlgebra
using StaticArrays

# Modules
using Aeon
using Aeon.Analytic


# Includes
include("domain.jl")
include("covariant.jl")
include("wls.jl")

end