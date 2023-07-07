export Operators

module Operators

# Dependencies
using StaticArrays
using LinearAlgebra

using Aeon
using Aeon.Geometry

# Includes
include("lagrange.jl")
include("stencil.jl")
include("block.jl")
include("functional.jl")

export gradient

function gradient(cell::CartesianIndex{N}, block::Block{N, T}, value::Operator{T}, deriv::Operator{T}, field::Field{N, T}) where {N, T}
    grad = SVector(
        ntuple(Val(N)) do dim
            opers = ntuple(i -> ifelse(i == dim, deriv, value), Val(N))
            return evaluate(cell, block, opers, field)
        end
    ) 

    grad .* blockcells(block)
end

end