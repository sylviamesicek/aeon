export Operators

"""
Handles the generation of numerical operators (specifically stencils). The primary type of operators
are from the 'summation-by-parts' methods, but it also includes dissipation operators and more
"""
module Operators

# Dependences
using StaticArrays

# Includes
include("stencil.jl")
include("operator.jl")
include("coefficients.jl")

include("MattssonNordström2004.jl")

end