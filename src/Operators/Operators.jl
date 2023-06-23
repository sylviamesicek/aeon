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
export SBPDerivative, SBPOperator
export left_boundary_weight, right_boundary_weight, derivative_left, derivative_right, derivative

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
Represents a collection of dervative operators, as well as their coefficients.
"""
struct SBPOperator{T, O, Source <: CoefficientSource, Weights <: AbstractVector, LeftDerivatives, RightDerivatives, Derivatives}
    left_weights::Weights
    central_weight::T
    right_weights::Weights

    left_derivatives::LeftDerivatives
    right_derivatives::RightDerivatives

    derivatives::Derivatives

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

# Base.length(oper::SBPOperator) = length(oper.derivatives)
# Base.eachindex(oper::SBPOperator) = eachindex(oper.derivatives)
# Base.getindex(oper::SBPOperator, rank::Int) = oper.derivatives[rank]

derivative_left(oper::SBPOperator, rank::Int) = oper.left_derivatives[rank]
derivative_right(oper::SBPOperator, rank::Int) = oper.right_derivatives[rank]
derivative(oper::SBPOperator, rank::Int) = oper.derivatives[rank]

# Includes
include("MattssonNordström2004.jl")

end