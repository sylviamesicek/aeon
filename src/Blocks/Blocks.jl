export Blocks

module Blocks

# Dependencies
using StaticArrays

abstract type Block{N, T} end

blockcells(block::Block) = error("blockcells is unimplemented for $(typeof(block))")

cellindices(block::Block) = CartesianIndices(blockcells(block))
cellwidths(block::Block{N, T}) where {N, T} = T(1) ./ blockcells(block) 
cellcenter(block::Block{N, T}, index::CartesianIndex{N}) where {N, T} = SVector{N, T}((index.I .- T(1//2)) ./ blockcells(block))

abstract type Stencil end

function stencilparity end
function stencilsupport end

evaluate(point::CartesianIndex, block::Block, func::AbstractArray{N, T}, stencil::Stencil, axis::Int) = error("Unimplemented")



include("lagrange.jl")

end