export Methods

module Methods

using StaticArrays
using LinearAlgebra

using Aeon
using Aeon.Geometry
using Aeon.Blocks

include("mesh.jl")
include("dofs.jl")
include("transfer.jl")
include("writer.jl")

end