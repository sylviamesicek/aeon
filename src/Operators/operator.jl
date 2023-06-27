## Exports

export MassOperator, CenteredOperator, BoundaryOperator
export GradientOperator, HessianOperator
export Operator, evaluate, project

"""
Represents an abstract numeric `Operator`. These can be applied to multivariate functions at nodal points.
"""
abstract type Operator{T, O} end

product_rec(point::CartesianIndex, ::Operator{T}, func::AbstractArray{T, N}, stencils::NTuple{L, (Stencil{T}, Int)}) = stencil_product(point, func, stencils)


"""
Computes the operation of the given operator on a function at a point.
"""
product(operator::Operator{T}, func::AbstractArray{T, N}, point::CartesianIndices{N}) where {N, T} = error("Unimplemented")

# function project(oper::Operator{T}, func::AbstractArray{T, N}, point::CartesianIndex{N}) where {N, T}
#     evaluate(oper, func, point)
# end

# function project(oper::Operator{T}, func::AbstractArray{T, N}, point::CartesianIndex{N}, axes::Int...) where {N, T}
#     @assert issorted(axes)

#     proj = ntuple(Val(N)) do dim
#         any(dim .== axes) ? Colon() : point[dim]
#     end

#     index = ntuple(Val(length(axes))) do dim
#         point[axes[dim]]
#     end

#     projectedfunc::AbstractArray{T, length(axes)} = view(func, proj...)

#     evaluate(oper, projectedfunc, index)
# end

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

"""
Evalates the integral over a domain using a mass operator.
"""
function product(oper::MassOperator{T, O}, func::AbstractArray{T, N}, point::CartesianIndex{N}) where {N, T, O}
    weights = ntuple(Val(N)) do dim
        total = size(func)[dim]
        index = point[dim]

        if index ≤ length(oper.left)
            return oper.left[index]
        elseif index > total - length(oper.right)
            return oper.right[index - total + length(oper.right)]
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

product(point::CartesianIndex{N}, oper::CenteredOperator{T}, func::AbstractArray{T, N}, axes::Int...) = product_rec(point, oper, func, (), axes...)

function product_rec(point::CartesianIndex{N}, oper::CenteredOperator{T}, func::AbstractArray{T, N}, stencils::NTuple{L, Tuple{Stencil{T}, Int}}, remaining::Int...) where {N, T, L}
    axis = first(remaining)

    index = point[axis]
    total = size(func)[axis]

    leftend = length(oper.left)
    rightbegin = total - length(oper.right)

    if index ≤ leftend
        coefrow = index
        return product_rec(point, oper, func, (stencils..., (oper.left[coefrow], axis)), Base.tail(remaining)...)
    elseif index > rightbegin
        coefrow = index - rightbegin
        return product_rec(point, oper, func, (stencils..., (oper.right[coefrow], axis)), Base.tail(remaining)...)
    else
        return product_rec(point, oper, func, (stencils..., (oper.central, axis)), Base.tail(remaining)...)
    end
end

struct ProlongationOperator{T, O, CL, BP, BL} <: Operator{T, O}
    left::SVector{BP, SVector{BL, T}}
    right::SVector{BP, SVector{BL, T}}
    central::SVector{T, CL}
end

product(point::CartesianIndex{N}, oper::ProlongationOperator{T}, func::AbstractArray{T, N}, axes::Int...) = product_rec(point, oper, func, (), axes...)

function product_rec(point::CartesianIndex{N}, oper::ProlongationOperator{T}, func::AbstractArray{T, N}, stencils::NTuple{L, Tuple{Stencil{T}, Int}}, axes::NTuple{L, Int}, remaining::Int...) where {N, T, L}
    axis = first(remaining)

    index = point[axis]
    total = size(func)[axis]

    leftend = length(oper.left)
    rightbegin = 2total - 1 - length(oper.right)

    if index ≤ leftend
        coefrow = index
        return stencil_product(point, oper, func, (stencils..., oper.left[coefrow]), (axes..., axis), Base.tail(remaining)...)
    elseif index > rightbegin
        coefrow = index - rightbegin
        return stencil_product(point, oper, func, (stencils..., oper.right[coefrow]), (axes..., axis), Base.tail(remaining)...)
    else
        return stencil_product(point, oper, func, (stencils..., oper.central), (axes..., axis), Base.tail(remaining)...)
    end
end

function evaluate_rec(oper::ProlongationOperator{T, O}, func::AbstractArray{T, N}, point::CartesianIndex{N}, stencils::Vararg{Stencil{T}, L}) where {N, T, O, L}
    axis = N - L
    intotal = size(func)[axis]
    outtotal = 2intotal - 1
    index = point[axis]

    leftend = length(oper.left)
    rightbegin = outtotal - length(oper.right)

    if index ≤ leftend
        coefrow = index
        stencil = Stencil{T}(oper.left[coefrow], 0)

        return evaluate_rec(oper, func, point, stencil, stencils...)
    elseif index > rightbegin
        coefrow = index - rightbegin
        stencil =  Stencil{T}(oper.right[coefrow], intotal - length(oper.right[coefrow]))

        return evaluate_rec(oper, func, point, stencil, stencils...)
    else
        if index % 2
            new_index = (index + 1) / 2
            stencil = Stencil{T}(SVector{1, T}(1), new_index)
        else
            new_index = (index) / 2
            stencil = Stencil{T}(oper.central, new_index - (length(oper.central) - 1)/2)
        end

        return evaluate_rec(oper, func, point, stencil, stencils...)
    end
end

#######################
## Gradient ###########
#######################

"""
An operator which applies a centered operator along each axis to return a gradient vector.
"""
struct GradientOperator{N, T, O, D1} <: Operator{T, O}
    first::D1

    GradientOperator{N}(first::CenteredOperator{T, O}) where {N, T, O} = new{N, T, O, typeof(first)}(first)
end

function evaluate(oper::GradientOperator{N, T}, func::AbstractArray{T, N}, point::CartesianIndex{N}) where {N, T}
    grad = ntuple(Val(N)) do dim
        project(oper.first, func, point, dim)
    end

    SVector(grad)
end

#######################
## Hessian ############
#######################

"""
An operator which applies a centered operator along each pair of axess to return a hessian matrix.
"""
struct HessianOperator{N, T, O, D1, D2} <: Operator{T, O}
    first::D1
    second::D2

    HessianOperator{N}(first::CenteredOperator{T, O}, second::CenteredOperator{T, O}) where {N, T, O} = new{N, T, O, typeof(first), typeof(second)}(first, second)
end

function hessian_component(oper::HessianOperator{N, T}, func::AbstractArray{T, N}, point::CartesianIndices{N}, axisi::Int, axisj::Int) where {N, T}
    if axisi < axisj
        return project(oper.first, func, point, axisi, axisj)
    elseif axisi > axisj
        return project(oper.first, func, point, axisj, axisi)
    else
        return project(oper.second, func, point, axisi)
    end
end

function evaluate(oper::HessianOperator{N, T}, func::AbstractArray{T, N}, point::CartesianIndices{N}) where {N, T}
    StaticArrays.sacollect(SMatrix{N, N}, hessian_component(oper, func, point, i, j) for i in 1:N, j in 1:N)
end

#######################
## Laplacian ##########
#######################

struct LaplacianOperator{N, T, O, D2} <: Operator{T, O} 
    second::D2

    LaplacianOperator{N}(second::CenteredOperator{T, 2, O}) where {N, T, O} = new{N, T, O, typeof(second)}(second)
end

function evaluate(oper::HessianOperator{N, T}, func::AbstractArray{T, N}, point::CartesianIndices{N}) where {N, T}
    lap = ntuple(Val(N)) do dim
        project(oper.second, func, point, dim)
    end

    sum(lap)
end

###################
## Corner #########
###################

"""
A operator only defined on the vertices of a numerical domain. 
"""
struct CornerOperator{T, O, BL} <: Operator{T, O}
    left::SVector{BL, T}
    right::SVector{BL, T}
end

evaluate(oper::CornerOperator{T, O}, func::AbstractArray{T, N}, point::CartesianIndices{N}) where {N, T, O} = evaluate_rec(oper, func, point)

function evaluate_rec(oper::CornerOperator{T, O}, func::AbstractArray{T, N}, point::CartesianIndex{N}, stencils::Vararg{Stencil{T}, L}) where {N, T, O, L}
    axis = N - L
    total = size(func)[axis]
    index = point[axis]

    if index == 1
        stencil = Stencil(oper.left, 0)
        return evaluate_rec(oper, func, point, stencil, stencils...)
    elseif index == total
        stencil = Stencil(oper.right, total - length(oper.right))
        return evaluate_rec(oper, func, point, stencil, stencils...)
    else
        return 0
    end
end