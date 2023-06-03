export Analytic

"""
Module for manipulating analytic functions.
"""
module Analytic

##################
## Dependencies ##
##################

using LinearAlgebra
using StaticArrays

# Other Modules
using Aeon

##################
## Exports #######
##################

export AnalyticFunction, AnalyticGradient, AnalyticHessian, AnalyticOperator
export gradient, hessian, laplacian, ∇, ∇², Δ

##################
## Core Types ####
##################

"""
Represents an analytic function, linear combinations of which approximate numerical functions across nodes. An analytic function
returns a scalar value.
"""
abstract type AnalyticFunction end

(func::AnalyticFunction)(::AbstractVector) = error("Value of Analytic Function $(typeof(func)) is undefined")

"""
A analytic gradient function, which returns a vector value.
"""
abstract type AnalyticGradient end

(grad::AnalyticGradient)(::AbstractVector) = error("Value of Analytic Gradient $(typeof(grad)) is undefined")

"""
An analytic hessian function, which returns a matrix value.
"""
abstract type AnalyticHessian end

(hess::AnalyticHessian)(::AbstractVector) = error("Value of Analytic Hessian $(typeof(hess)) is undefined")

"""
Transforms an analytic function into an analytic gradient.
"""
gradient(func::AnalyticFunction) = error("Gradient of Analytic Function $(typeof(func)) is undefined")

"""
Transforms an analytic function into an analytic hessian.
"""
hessian(func::AnalyticFunction) = error("Hessian of Analytic Function $(typeof(func)) is undefined")

"""
Alternative symbol for gradient operator.
"""
const ∇ = gradient

"""
Alternative symbol for hessian operator.
"""
const ∇² = hessian

##########################
## Laplacian #############
##########################

struct Laplacian{H} <: AnalyticFunction
    hess::H
end

(func::Laplacian)(x::AbstractVector) = dot(func.hess(x), I)

"""
Transforms an analytic function into an analytic laplacian.
"""
laplacian(func::AnalyticFunction) = Laplacian(hessian(func))

"""
Alternative symbol for laplacian operator.
"""
const Δ = laplacian

##########################
## Operators #############
##########################

"""
An analytic operator is a function of Analytic Functions. That is, when applied to a analytic function, it yields a new analytic function.
"""
abstract type AnalyticOperator end

(oper::AnalyticOperator)(func::AnalyticFunction) = error("Analytic operation of $(typeof(oper)) on $(typeof(func)) is undefined.")

########################
## Includes ############
########################

# Includes
include("identity.jl")
include("gaussian.jl")
include("monomial.jl")
include("combine.jl")
include("transform.jl")
include("directional.jl")
include("unkown.jl")

end