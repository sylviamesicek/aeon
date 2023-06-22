export Grid

"""
Utilities for building and refining grid-like meshs.
"""
module Grid

# Dependencies
using LinearAlgebra
using StaticArrays

# Other Modules
using Aeon
using Aeon.Analytic
using Aeon.Approx
using Aeon.Geometry
using Aeon.Method

################
## Includes ####
################

include("domain.jl")
include("mesh.jl")

end