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

    WLS{N, T, B}(matrix::LU{T, Matrix{T}}, weights::Vector{T}, basis::B) where {N, T, B <: AnalyticBasis{N, T}} = new{N, T, R}(matrix, weights, basis)
end

"""
Uses the Weighted Least Squares method, with a particular weight function and basis to approximate functions on discrete domains.
"""
function wls(domain::Domain{N, T}, basis::B, weight::AnalyticField{N, T, 0}) where {N, T, B <: AnalyticBasis{N, T}}
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

function approx(wls::WLS{N, T}, operator::AnalyticOperator{N, T, R}, position::SVector{N, T}) where {N, T, R}
    blength = length(wls.basis)
    nlength = length(wls.weights)

    tdims = ntuple(_ -> N, Val(R))
    tcoords = CartesianIndices(tdims)

    rhs = Array(undef, blength, tdims...)

    for i in eachindex(wls.basis)
        values = operator(wls.basis[i])(position)

        for coord in tcoords
            rhs[i, Tuple(coord)...] = values[coord]
        end
    end

    stencil = Array(undef, nlength, tdims...)

    for coords in tcoords
        sview = @view stencil[:, coords...]
        rhsview = @view rhs[:, coords...]

        ldiv!(sview, wls.matrix, rhsview)

        sview *= wls.weights
    end

    ApproxOperator{N, T, R}(stencil)
end