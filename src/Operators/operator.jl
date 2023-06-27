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
evaluate(point::CartesianIndex{N}, oper::Operator{T}, func::AbstractArray{T, N}, axes::Int...) where {N, T} = error("Evaluation of $(typeof(oper)) is unimplemented.")

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
struct MassOperator{T, O, L} <: Operator{T, O}
    weights::SVector{L, T}

    MassOperator{O}(weights::SVector{L, T}) where {L, O, T} = new{T, O, L}(weights)
end

function evaluate(point::CartesianIndex{N}, oper::MassOperator{T, O}, func::AbstractArray{T, N}, axes::Int...) where {N, T, O}
    weights = ntuple(Val(length(axes))) do adim
        axis = axes[adim]

        index = point[axis]
        total = size(func)[axis]

        leftend = length(oper.left)
        rightbegin = total - length(oper.right)

        if index ≤ leftend
            return oper.left[index]
        elseif index > rightbegin
            return oper.right[index_from_right(total, index)]
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
    boundary::SVector{BP, SVector{BL, T}}
    central::SVector{CL, T}

    CenteredOperator{O}(boundary::SVector{BP, SVector{BL, T}}, central::SVector{CL, T}) where {T, BP, BL, CL} = new{T, O, CL, BP, BL}(boundary, central)
end

evaluate(point::CartesianIndex{N}, oper::CenteredOperator{T}, func::AbstractArray{T, N}, axes::Int...) = evaluate_rec(point, oper, func, (), axes...)

function evaluate_rec(point::CartesianIndex{N}, oper::CenteredOperator{T}, func::AbstractArray{T, N}, stencils::NTuple{L, Stencil{T}}, remaining::Int...) where {N, T, L}
    axis = first(remaining)

    index = point[axis]
    total = size(func)[axis]

    leftend = length(oper.boundary)
    rightbegin = total - length(oper.boundary)

    if index ≤ leftend
        stencil = LeftStencil(oper.boundary[index], axis)

        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    elseif index > rightbegin
        stencil = RightStencil(oper.boundary[index_from_right(total, index)], axis, total)

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
    boundary::SVector{BP, SVector{BL, T}}
    central::SVector{CL, T}

    ProlongationOperator{O}(boundary::SVector{BP, SVector{BL, T}}, central::SVector{CL, T}) where {T, BP, BL, CL} = new{T, O, CL, BP, BL}(boundary, central)
end

evaluate(point::CartesianIndex{N}, oper::ProlongationOperator{T}, func::AbstractArray{T, N}, axes::Int...) = product_rec(point, oper, func, (), axes...)

function evaluate_rec(point::CartesianIndex{N}, oper::ProlongationOperator{T}, func::AbstractArray{T, N}, stencils::NTuple{L, Stencil{T}}, remaining::Int...) where {N, T, L}
    axis = first(remaining)

    index = point[axis]
    total = size(func)[axis]

    @assert total % 2 == 1

    leftend = length(oper.boundary)
    rightbegin = coarse_to_refined(total) - length(oper.boundary)

    if index ≤ leftend
        stencil = LeftStencil(oper.boundary[index], axis)

        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    elseif index > rightbegin
        stencil = RightStencil(oper.boundary[index_from_right(coarse_to_refined(total), index)], axis, total)

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
    boundary::SVector{BP, SVector{BL, T}}
    central::SVector{CL, T}

    RestrictionOperator{O}(boundary::SVector{BP, SVector{BL, T}}, central::SVector{CL, T}) where {T, BP, BL, CL} = new{T, O, CL, BP, BL}(boundary, central)
end

evaluate(point::CartesianIndex{N}, oper::RestrictionOperator{T}, func::AbstractArray{T, N}, axes::Int...) = product_rec(point, oper, func, (), axes...)

function evaluate_rec(point::CartesianIndex{N}, oper::RestrictionOperator{T}, func::AbstractArray{T, N}, stencils::NTuple{L, Stencil{T}}, remaining::Int...) where {N, T, L}
    axis = first(remaining)

    index = point[axis]
    total = size(func)[axis]

    @assert total % 2 == 1

    leftend = length(oper.boundary)
    rightbegin = refined_to_coarse(total) - length(oper.boundary)

    if index ≤ leftend
        stencil = LeftStencil(oper.left[index], axis)

        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    elseif index > rightbegin
        stencil = RightStencil(oper.right[index_from_right(refined_to_coarse(total), index)], axis, total)

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
struct BoundaryOperator{T, O, L} <: Operator{T, O}
    boundary::SVector{L, T}
end

evaluate(point::CartesianIndices{N}, oper::BoundaryOperator{T, O}, func::AbstractArray{T, N}, axes::Int...) where {N, T, O} = evaluate_rec(oper, func, point, (), axes...)

function evaluate_rec(point::CartesianIndex{N}, oper::BoundaryOperator{T, O}, func::AbstractArray{T, N}, stencils::NTuple{L, Stencil{T}}, remaining::Int...) where {N, T, O, L}
    axis = first(remaining)

    index = point[axis]
    total = size(func)[axis]

    if index == 1
        stencil = LeftStencil(oper.boundary, axis)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    elseif index == total
        stencil = RightStencil(oper.boundary, axis, total)
        return evaluate_rec(point, oper, func, (stencils..., stencil), Base.tail(remaining)...)
    else
        return 0
    end
end
