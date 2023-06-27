#######################
## Exports ############
#######################

export GradientOperator, HessianOperator, LaplacianOperator, evaluate


#######################
## Gradient ###########
#######################

"""
An operator which applies a centered operator along each axis to return a gradient vector.
"""
struct GradientOperator{N, T, O, D1}
    first::D1

    GradientOperator{N}(first::CenteredOperator{T, O}) where {N, T, O} = new{N, T, O, typeof(first)}(first)
end

function evaluate(point::CartesianIndex{N}, oper::GradientOperator{N, T}, func::AbstractArray{T, N}) where {N, T}
    grad = ntuple(Val(N)) do dim
        opers = ntuple(i -> i == dim ? oper.first : IdentityOperator{T, O}, Val(N))
        product(point, opers, func)
    end

    SVector(grad)
end

#######################
## Hessian ############
#######################

"""
An operator which applies a centered operator along each pair of axess to return a hessian matrix.
"""
struct HessianOperator{N, T, O, D1, D2}
    first::D1
    second::D2

    HessianOperator{N}(first::CenteredOperator{T, O}, second::CenteredOperator{T, O}) where {N, T, O} = new{N, T, O, typeof(first), typeof(second)}(first, second)
end

function evaluate(point::CartesianIndices{N}, oper::HessianOperator{N, T}, func::AbstractArray{T, N},) where {N, T}
    hessian = ntuple(Val(N)) do i
        ntuple(Val(N)) do j
            if i == j
                opers = ntuple(k -> k == i ? oper.second : IdentityOperator{T, O}, Val(N))
                product(point, opers, func)
            else
                opers = ntuple(k -> k == i || k == j ? oper.first : IdentityOperator{T, O}, Val(N))
                product(point, opers, func)
            end
        end
    end
    
    StaticArrays.sacollect(SMatrix{N, N}, hessian[i][j] for i in 1:N, j in 1:N)
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
        opers = ntuple(k -> k == i ? oper.second : IdentityOperator{T, O}, Val(N))
        product(point, opers, func)
    end

    sum(lap)
end