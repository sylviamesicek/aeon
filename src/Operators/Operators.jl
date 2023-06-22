export Operators

"""
Handles the generation of numerical operators (specifically stencils). The primary type of operators
are from the 'summation-by-parts' methods, but it also includes dissipation operators and more
"""
module Operators

# Dependences
using StaticArrays

# Core Exports

export CoefficientSource
export SBPDerivative, SBPBoundaryDerivative, SBPOperator
export SBPDerivatives, SBPBoundaryDerivatives
export sbpoperator, sbpderivatives,sbpboundaryderivatives
export left_boundary_weight, right_boundary_weight

# Core

"""
Represents the source of a coefficient set, most often some SBP paper.
"""
abstract type CoefficientSource end

"""
Represents a single SBP derivative operator, consisting of a interior stencil
and additional blocks for both edges of the domain.
"""
struct SBPDerivative{T, O, Block, Central}
    # Vector of coefficient rows for leftmost degrees of freedom. 
    left_block::Block
    # Vector of coefficient rows for rightmost degrees of freedom. 
    right_block::Block
    # Coefficient of central point on stencil
    central_coefs::Central
end

"""
Represents stencils for lower-rank derivatives on boundaries, used to enforce nuemann conditions
for instance.
"""
struct SBPBoundaryDerivative{T, O, L1, L2} 
    left::SVector{L1, T}
    right::SVector{L2, T}
end

"""
A set of SBP derivative operators, indexed by rank of derivative.
"""
const SBPDerivatives{T, O, L} = NTuple{L, SBPDerivative{T, O}} where {T, O, L}

"""
A set of SBP boundary derivative operators, indexed by rank of derivative
"""
const SBPBoundaryDerivatives{T, O, L} = NTuple{L, SBPBoundaryDerivative{T, O}} where {T, O, L}

"""
Represents a collection of dervative operators, as well as their coefficients.
"""
struct SBPOperator{T, O, Source <: CoefficientSource, Weights <: AbstractVector}
    left_weights::Weights
    central_weight::T
    right_weights::Weights

    source::Source
end

"""
Returns the leftmost boundary weight of the mass matrix.
"""
left_boundary_weight(oper::SBPOperator) = oper.left_weights[begin]

"""
Returns the rightmost boundary weight of the mass matrix.
"""
right_boundary_weight(oper::SBPOperator) = oper.right_weights[end]

sbpoperator(::Val{T}, ::Val{O}, source::CoefficientSource) where {T, O} = error("Unimplemented")
sbpderivatives(::SBPOperator) = error("Unimplemented")
sbpboundaryderivatives(::SBPOperator) = error("Unimplemented")

# Includes
include("MattssonNordström2004.jl")

end