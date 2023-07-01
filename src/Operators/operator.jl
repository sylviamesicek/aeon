## Exports

export Operator, evaluate
export MassOperator, CenteredOperator, BoundaryOperator
export ProlongationOperator, RestrictionOperator
export IdentityOperator

"""
Represents an abstract numeric `Operator`. These can be applied to multivariate functions at nodal points.
"""
abstract type Operator{T, O} end

evaluate(point::CartesianIndex{N}, opers::NTuple{N, Operator{T, O}}, func::AbstractArray{T, N}) where {N, T, O} = product_rec(point, (), func, opers...)
evaluate(point::CartesianIndex{N}, oper::Operator{T}, func::AbstractArray{T, N}) where {N, T} = product_rec(point, (), func, ntuple(i -> oper, Val(N))...)

"""
Recursive base case for evaluating operator products.
"""
product_rec(point::CartesianIndex{N}, stencils::NTuple{N, Stencil{T}}, func::AbstractArray{T, N}) where {N, T} = product(point, stencils, func)


###################
## Identity #######
###################

struct IdentityOperator{T, O} <: Operator{T, O} end

function product_rec(point::CartesianIndex{N}, stencils::NTuple{L, Stencil{T}}, func::AbstractArray{T, N}, ::IdentityOperator{T, O}, opers::Operator{T, O}...) where {N, T, L, O}
    stencil = IdentityStencil{T}()

    product_rec(point, (stencils..., stencil), func, opers...)
end

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

function product_rec(point::CartesianIndex{N}, stencils::NTuple{L, Stencil{T}}, func::AbstractArray{T, N}, oper::MassOperator{T, O}, opers::Operator{T, O}...) where {N, T, L, O}
    axis = L + 1
    index = point[axis]
    total = size(func)[axis]

    leftend = length(oper.boundary)
    rightbegin = total - length(oper.boundary)

    if index ≤ leftend
        stencil = ValueStencil(oper.weights[index])

        return product_rec(point, (stencils..., stencil), func, opers...)
    elseif index > rightbegin
        stencil = ValueStencil(oper.weights[index_from_right(total, index)])

        return product_rec(point, (stencils..., stencil), func, opers...)
    else
        stencil = IdentityStencil{T}()

        return product_rec(point, (stencils..., stencil), func, opers...)
    end
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
    right::SVector{BP, SVector{BL, T}}
    central::SVector{CL, T}

    CenteredOperator{O}(left::SVector{BP, SVector{BL, T}}, right::SVector{BP, SVector{BL, T}}, central::SVector{CL, T}) where {T, O, BP, BL, CL} = new{T, O, CL, BP, BL}(left, right, central)
end

function product_rec(point::CartesianIndex{N}, stencils::NTuple{L, Stencil{T}}, func::AbstractArray{T, N}, oper::CenteredOperator{T, O}, opers::Operator{T, O}...) where {N, T, L, O}
    axis = L + 1
    index = point[axis]
    total = size(func)[axis]

    leftend = length(oper.left)
    rightbegin = total - length(oper.right)

    if index ≤ leftend
        stencil = LeftStencil(oper.left[index])

        return product_rec(point, (stencils..., stencil), func, opers...)
    elseif index > rightbegin
        stencil = RightStencil(oper.right[index_from_right(total, index)], total)

        return product_rec(point, (stencils..., stencil), func, opers...)
    else
        stencil = CenteredStencil(oper.central)

        return product_rec(point, (stencils..., stencil), func, opers...)
    end
end

#######################
## Prolonged Operator #
#######################

struct ProlongationOperator{T, O, CL, BP, BL} <: Operator{T, O}
    boundary::SVector{BP, SVector{BL, T}}
    central::SVector{CL, T}

    ProlongationOperator{O}(boundary::SVector{BP, SVector{BL, T}}, central::SVector{CL, T}) where {T, O, BP, BL, CL} = new{T, O, CL, BP, BL}(boundary, central)
end

function product_rec(point::CartesianIndex{N}, stencils::NTuple{L, Stencil{T}}, func::AbstractArray{T, N}, oper::ProlongationOperator{T, O}, opers::Operator{T, O}...) where {N, T, L, O}
    axis = L + 1
    index = point[axis]
    total = size(func)[axis]

    @assert total % 2 == 1

    leftend = length(oper.boundary)
    rightbegin = coarse_to_refined(total) - length(oper.boundary)

    if index ≤ leftend
        stencil = LeftStencil(oper.boundary[index])

        return product_rec(point, (stencils..., stencil), func, opers...)
    elseif index > rightbegin
        stencil = RightStencil(oper.boundary[index_from_right(coarse_to_refined(total), index)], total)

        return product_rec(point, (stencils..., stencil), func, opers...)
    elseif index % 2 == 0
        stencil = ProlongedEvenStencil(oper.central)

        return product_rec(point, (stencils..., stencil), func, opers...)
    else
        stencil = ProlongedOddStencil{T}()

        return product_rec(point, (stencils..., stencil), func, opers...)
    end
end

###################
## Restriction ####
###################

struct RestrictionOperator{T, O, CL, BP, BL} <: Operator{T, O}
    boundary::SVector{BP, SVector{BL, T}}
    central::SVector{CL, T}

    RestrictionOperator{O}(boundary::SVector{BP, SVector{BL, T}}, central::SVector{CL, T}) where {T, O, BP, BL, CL} = new{T, O, CL, BP, BL}(boundary, central)
end

function product_rec(point::CartesianIndex{N}, stencils::NTuple{L, Stencil{T}}, func::AbstractArray{T, N}, oper::RestrictionOperator{T, O}, opers::Operator{T, O}...) where {N, T, L, O}
    axis = L + 1
    index = point[axis]
    total = size(func)[axis]

    @assert total % 2 == 1

    leftend = length(oper.boundary)
    rightbegin = refined_to_coarse(total) - length(oper.boundary)

    if index ≤ leftend
        stencil = LeftStencil(oper.boundary[index])

        return product_rec(point, (stencils..., stencil), func, opers...)
    elseif index > rightbegin
        stencil = RightStencil(oper.boundary[index_from_right(refined_to_coarse(total), index)], total)

        return product_rec(point, (stencils..., stencil), func, opers...)
    else
        stencil = RestrictedStencil(oper.central)

        return product_rec(point, (stencils..., stencil), func, opers...)
    end
end

###################
## Boundary #######
###################

"""
A operator only defined on the vertices of a numerical domain. 
"""
struct BoundaryOperator{T, O, L} <: Operator{T, O}
    left::SVector{L, T}
    right::SVector{L, T}

    BoundaryOperator{O}(left::SVector{L, T}, right::SVector{L, T}) where {T, O, L} = new{T, O, L}(left, right)
end

function product_rec(point::CartesianIndex{N}, stencils::NTuple{L, Stencil{T}}, func::AbstractArray{T, N}, oper::BoundaryOperator{T, O}, opers::Operator{T, O}...) where {N, T, L, O}
    axis = L + 1
    index = point[axis]
    total = size(func)[axis]

    if index == 1
        stencil = LeftStencil(oper.left)
        return product_rec(point, (stencils..., stencil), func , opers...)
    elseif index == total
        stencil = RightStencil(oper.right, total)
        return product_rec(point, (stencils..., stencil), func, opers...)
    else
        return 0
    end
end