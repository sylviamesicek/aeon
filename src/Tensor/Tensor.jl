export Tensor

module Tensor

# Dependencies
using LinearAlgebra
using StaticArrays

# Core

export STensor, AbstractTensor, order

"""
Alias for an N-dimensional SArray
"""
const STensor{N, T, O, L} = SArray{NTuple{O, N}, T, O, L} where {N, T, O, L}

"""
An abstract tensor of order O
"""
abstract type AbstractTensor{N, T, O} end

order(::AbstractTensor{N, T, O}) where {N, T, O} = O

# Includes
include("transform.jl")
include("affine.jl")
include("covariant.jl")

end