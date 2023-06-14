#####################
## Exports ##########
####################

export MLS, wls, approx

#####################
## MLS ##############
#####################

"""
Moving least squares method/decomposition of a domain.
"""
struct MLS{N, T, B, W}
    domain::Domain{N, T}
    matrix::LU{T, Matrix{T}}
    basis::B
    weight::W
end

"""
Uses the Moving Least Squares method, with a particular weight function and basis to approximate functions on discrete domains.
"""
function mls(domain::Domain{N, T}, basis::ABasis{N, T}, weight::AFunction{N, T}) where {N, T}
    nlength = length(domain)
    blength = length(basis)

    matrix = Matrix{T}(undef, blength, nlength)

    for j in eachindex(domain)
        position = domain[j]
        for i in eachindex(basis)
            matrix[i, j] = basis[i](position)
        end
    end

    MLS{N}(domain, lu(matrix), basis, weight)
end

"""
Approximates a simple scalar functional using MLS.
"""
function approx(mls::MLS{N, T}, operator::AScalar{N, T}, position::SVector{N, T}) where {N, T}
    nlength = length(mls.domain)
    blength = length(mls.basis)

    rhs = Vector(undef, blength)
    weights = Vector(undef, nlength)

    for i in eachindex(mls.domain)
        weights[i] = mls.weight(position - mls.domain[i])
    end

    for i in eachindex(mls.basis)
        rhs[i] = operator(wls.basis[i], position)
    end

    

    stencil = Vector(undef, nlength)

    ldiv!(stencil, wls.matrix, rhs)
    stencil *= wls.weights

    ApproxScalar{N, T}(stencil)
end

"""
Approximates a covariant functional
"""
function approx(wls::WLS{N, T}, operator::ACovariant{N, T, O, L}, position::SVector{N, T}) where {N, T, O, L}
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