export Operators

"""
Handles the generation of numerical operators (specifically stencils). The primary type are derivative operators
from the 'summation-by-parts' methods, but it also includes compatible operators for dissipation, prologation,
restriction, etc.
"""
module Operators

# Dependences
using StaticArrays

# Includes
include("stencil.jl")
include("operator.jl")
include("derivative.jl")

include("lagrange.jl")

include("coefficients.jl")
include("MattssonNordström2004.jl")

end