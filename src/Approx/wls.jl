#####################
## Exports ##########
####################

export WLS

#####################
## WLS ##############
#####################

struct WLS{N, T, B}
    matrix::LU{T, Matrix{T}}
    weights::Vector{T}
    basis::B

    WLS{N, T, B}(matrix::LU{T, Matrix{T}}, weights::Vector{T}, basis::B) where {N, T, B <: ABasis{N, T}} = new{N, T, R}(matrix, weights, basis)
end

"""
Uses the Weighted Least Squares method, with a particular weight function and basis to approximate functions on discrete domains.
"""
function wls(domain::Domain{N, T}, basis::B, weight::AFunction{N, T}) where {N, T, B <: ABasis{N, T}}
    nlength = length(domain)
    blength = length(basis)

    matrix = Matrix{T}(undef, blength, nlength)
    weights = Vector{T}(undef, nlength)

    for j in eachindex(domain)
        position = domain[j]
        weights[j] = weight(position)
        for i in eachindex(basis)
            matrix[i, j] = weights[j]  * basis[i](position)
        end
    end

    WLS{N, T, B}(lu(matrix), weights, basis)
end

function approx(wls::WLS{N, T}, operator::AFunctional{N, T, T}, position::SVector{N, T}) where {N, T}
    blength = length(wls.basis)
    nlength = length(wls.weights)

    rhs = Vector(undef, blength)

    for i in eachindex(wls.basis)
        rhs[i] = operator(wls.basis[i], position)
    end

    stencil = Vector(undef, nlength)

    ldiv!(stencil, wls.matrix, rhs)
    stencil *= wls.weights

    ApproxGeneric{N, T, T}(stencil)
end

function approx(wls::WLS{N, T}, operator::AFunctional{N, T, Covariant{N, T, O, L}}, position::SVector{N, T}) where {N, T, O, L}
    blength = length(wls.basis)
    nlength = length(wls.weights)

    tdims = ntuple(_ -> N, Val(O))
    tcoords = CartesianIndices(tdims)

    rhs = StaticArrays.sacollect(SArray{NTuple{O, N}, Vector{T}, O, L}, Vector{T}(undef, blength) for _ in tcoords)

    for i in eachindex(wls.basis)
        values = operator(wls.basis[i], position)

        for coord in tcoords
            rhs[coord][i] = values[coord]
        end
    end

    stencil = StaticArrays.sacollect(SArray{NTuple{O, N}, Vector{T}, O, L}, Vector{T}(undef, nlength) for _ in tcoords)

    for coord in tcoords
        ldiv!(stencil[coord], wls.matrix, rhs[coord])

        stencil[coord] *= wls.weights
    end

    ApproxCovariant(stencil)
end