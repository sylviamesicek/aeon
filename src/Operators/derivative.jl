#######################
## Exports ############
#######################

export Gradient, Hessian, Laplacian, evaluate


#######################
## Gradient ###########
#######################

"""
An operator which applies a centered operator along each axis to return a gradient vector.
"""
struct Gradient{N, T, O, D1}
    d1::D1

    Gradient{N}(d1::CenteredOperator{T, O}) where {N, T, O} = new{N, T, O, typeof(d1)}(d1)
end

function evaluate(point::CartesianIndex{N}, oper::Gradient{N, T}, func::AbstractArray{T, N}) where {N, T}
    grad = ntuple(Val(N)) do dim
        opers = ntuple(i -> i == dim ? oper.d1 : IdentityOperator{T, O}, Val(N))
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
struct Hessian{N, T, O, D1, D2}
    d1::D1
    d2::D2

    Hessian{N}(d1::CenteredOperator{T, O}, d2::CenteredOperator{T, O}) where {N, T, O} = new{N, T, O, typeof(d1), typeof(d2)}(d1, d2)
end

function evaluate(point::CartesianIndex{N}, oper::Hessian{N, T, O}, func::AbstractArray{T, N}) where {N, T, O}
    hessian = ntuple(Val(N)) do i
        ntuple(Val(N)) do j
            if i == j
                opers = ntuple(k -> k == i ? oper.d2 : IdentityOperator{T, O}(), Val(N))
                product(point, opers, func)
            else
                opers = ntuple(k -> k == i || k == j ? oper.d1 : IdentityOperator{T, O}(), Val(N))
                product(point, opers, func)
            end
        end
    end
    
    StaticArrays.sacollect(SMatrix{N, N}, hessian[i][j] for i in 1:N, j in 1:N)
end

#######################
## Laplacian ##########
#######################

struct Laplacian{N, T, O, D2} <: Operator{T, O} 
    d2::D2

    Laplacian{N}(d2::CenteredOperator{T, O}) where {N, T, O} = new{N, T, O, typeof(d2)}(d2)
end

function evaluate(point::CartesianIndex{N}, oper::Laplacian{N, T, O}, func::AbstractArray{T, N}) where {N, T, O}
    lap = ntuple(Val(N)) do dim
        opers = ntuple(i -> i == dim ? oper.d2 : IdentityOperator{T, O}(), Val(N))
        product(point, opers, func)
    end

    sum(lap)
end