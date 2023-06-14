export Grid

module Grid

# Dependencies
using LinearAlgebra
using StaticArrays

# Other Modules
using Aeon.Analytic
using Aeon.Approx
using Aeon.Tensor
using Aeon.Method

################
## Includes ####
################

include("domain.jl")
include("method.jl")

end