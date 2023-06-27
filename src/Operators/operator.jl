## Exports

export Operator, evaluate
export MassOperator, CenteredOperator, BoundaryOperator
export ProlongationOperator, RestrictionOperator

"""
Represents an abstract numeric `Operator`. These can be applied to multivariate functions at nodal points.
"""
abstract type Operator{T, O} end

"""
Evaluates an operator at a point, taking the tensor product along the given axes.
"""
evaluate(point::CartesianIndex{N}, ::Operator{T}, func::AbstractArray{T, N}, axes::Int...) where {N, T} = error("Unimplemented.")

"""
Recursive base case for evaluation products.
"""
evaluate_rec(point::CartesianIndex, ::Operator{T}, func::AbstractArray{T, N}, stencils::NTuple{L, Stencil{T}}) = stencil_product(point, func, stencils)

###################
## Mass ###########
###################

"""
A (bi)symmetric diagnol operator which represents numerical quadtrature (ie, the intergral over a given domain).
"""
struct MassOperator{T, O, WL} <: Operator{T, O}
    left::SVector{T, WL}
    right::SVector{T, WL}
end

function evaluate(point::CartesianIndex{N}, oper::MassOperator{T, O}, func::AbstractArray{T, N}, axes::Int...) where {N, T, O}
    weights = ntuple(Val(length(axes))) do adim
        axis = axes[adim]

        index = point[axis]

        leftend = length(oper.left)
        rightbegin = size(func)[axis] - length(oper.right)

        if index ≤ leftend
            return oper.left[index]
        elseif index > rightbegin
            return oper.right[index - rightbegin]
        else
            return T(1)
        end
    end

    prod(weights)
end

"""
Inverts a mass operator.
"""
Base.inv(oper::MassOperator{T, O}) where {T, O} = MassOperator{T, O}(T(1) ./ oper.left, T(1) ./ oper.right)

#######################
## Centered Operator ##
#######################

"""
A operator given by a central difference in the interior of the domain, and a one-sided difference along the boundary of a domain.
"""
struct CenteredOperator{T, O, CL, BP, BL} <: Operator{T, O}
    left::SVector{BP, SVector{BL, T}}
    right::SVector{BP, SVector{T, BO}}
    central::SVector{CL, T}
end

evaluate(point::CartesianIndex{N}, oper::CenteredOperator{T}, func::AbstractArray{T, N}, axes::Int...) = evaluate_rec(point, oper, func, (), axes...)

function evaluate_rec(point::CartesianIndex{N}, oper::CenteredOperator{T}, func::AbstractArray{T, N}, stencils::NTuple{L, Stencil{T}}, remaining::Int...) where {N, T, L}
    axis = first(remaining)

    index = point[axis]
    total = size(func)[axis]

    leftend = length(oper.left)
    rightbegin = total - length(oper.right)

    if index ≤ leftend
        coefrow = index
        stencil = LeftStencil(oper.left[coefrow], axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    elseif index > rightbegin
        coefrow = index - rightbegin
        stencil = RightStencil(oper.right[coefrow], axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    else
        stencil = CenteredStencil(oper.central, axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    end
end

#######################
## Prolonged Operator #
#######################

struct ProlongationOperator{T, O, CL, BP, BL} <: Operator{T, O}
    left::SVector{BP, SVector{BL, T}}
    right::SVector{BP, SVector{BL, T}}
    central::SVector{CL, T}
end

evaluate(point::CartesianIndex{N}, oper::ProlongationOperator{T}, func::AbstractArray{T, N}, axes::Int...) = product_rec(point, oper, func, (), axes...)

function evaluate_rec(point::CartesianIndex{N}, oper::ProlongationOperator{T}, func::AbstractArray{T, N}, stencils::NTuple{L, Stencil{T}}, remaining::Int...) where {N, T, L}
    axis = first(remaining)

    index = point[axis]
    total = size(func)[axis]

    @assert total % 2 == 1

    leftend = length(oper.left)
    rightbegin = 2total - 1 - length(oper.right)

    if index ≤ leftend
        coefrow = index
        stencil = LeftStencil(oper.left[coefrow], axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    elseif index > rightbegin
        coefrow = index - rightbegin
        stencil = RightStencil(oper.right[coefrow], axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    elseif index % 2
        stencil = ProlongedEvenStencil(oper.central, axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    else
        stencil = ProlongedOddStencil(axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    end
end

###################
## Restriction ####
###################

struct RestrictionOperator{T, O, CL, BP, BL} <: Operator{T, O}
    left::SVector{BP, SVector{BL, T}}
    right::SVector{BP, SVector{BL, T}}
    central::SVector{CL, T}
end

evaluate(point::CartesianIndex{N}, oper::RestrictionOperator{T}, func::AbstractArray{T, N}, axes::Int...) = product_rec(point, oper, func, (), axes...)

function evaluate_rec(point::CartesianIndex{N}, oper::RestrictionOperator{T}, func::AbstractArray{T, N}, stencils::NTuple{L, Stencil{T}}, remaining::Int...) where {N, T, L}
    axis = first(remaining)

    index = point[axis]
    total = size(func)[axis]

    @assert total % 2 == 1

    leftend = length(oper.left)
    rightbegin = (total + 1) ÷ 2 - length(oper.right)

    if index ≤ leftend
        coefrow = index
        stencil = LeftStencil(oper.left[coefrow], axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    elseif index > rightbegin
        coefrow = index - rightbegin
        stencil = RightStencil(oper.right[coefrow], axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    else
        stencil = RestrictedStencil(oper.central, axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    end
end

###################
## Boundary #######
###################

"""
A operator only defined on the vertices of a numerical domain. 
"""
struct BoundaryOperator{T, O, BL} <: Operator{T, O}
    left::SVector{BL, T}
    right::SVector{BL, T}
end

evaluate(point::CartesianIndices{N}, oper::BoundaryOperator{T, O}, func::AbstractArray{T, N}, axes::Int...) where {N, T, O} = evaluate_rec(oper, func, point, (), axes...)

function evaluate_rec(point::CartesianIndex{N}, oper::BoundaryOperator{T, O}, func::AbstractArray{T, N}, stencils::NTuple{L, Stencil{T}}, remaining::Int...) where {N, T, O, L}
    axis = first(remaining)

    index = point[axis]
    total = size(func)[axis]

    if index == 1
        stencil = LeftStencil(oper.left, axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    elseif index == total
        stencil = RightStencil(oper.right, axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    else
        return 0
    end
end
