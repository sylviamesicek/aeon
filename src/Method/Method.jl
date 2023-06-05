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

##########################
## Includes ##############
##########################

include("basis.jl")
include("vertex.jl")
include("approx.jl")

end