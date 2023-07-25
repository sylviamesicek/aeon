export Methods

module Methods

using StaticArrays
using LinearAlgebra
using LinearMaps
using IterativeSolvers

using Aeon
using Aeon.Geometry
using Aeon.Blocks

include("mesh.jl")
include("dofs.jl")
include("transfer.jl")

include("operator.jl")
include("multigrid.jl")
include("writer.jl")

end