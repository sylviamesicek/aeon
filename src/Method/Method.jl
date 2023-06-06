export Method

module Method

# Dependencies

using LinearAlgebra
using StaticArrays

# Other Modules
using Aeon
using Aeon.Analytic

##########################
## Exports ###############
##########################

##########################
## Core Types ############
##########################

"""
A mesh represents a division of a space into a collection of cells which may
be refined or coarsed to properly distribute numerical load. The Degrees of 
freedom for the mesh are handled by a corresponding DoFHandler object.
"""
abstract type Mesh end

"""
Manages degrees of freedom defined on a mesh. This includes managing constraints,
hanging points, multiple levels of refinement, etc.
"""
abstract type DoFHandler end

##########################
## Includes ##############
##########################

include("basis.jl")
include("vertex.jl")
include("approx.jl")

end