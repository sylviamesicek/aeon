#####################
## Exports ##########
#####################

export NodeSpace, stencil
export WLS, wls

#####################
## Interface ########
#####################

"""
Represents a support structure in some abstract NodeSpace. These are passed to approximation engines via the nodespace! function.
"""
struct NodeSpace{N, T}
    support::Vector{SVector{N, T}}

    function NodeSpace(support::Vector{SVector{N, T}}) where {N, T}
        new{N, T}(support)
    end
end

Base.length(nodespace::NodeSpace) = length(nodespace.support)
Base.eachindex(nodespace::NodeSpace) = eachindex(nodespace.support)
Base.getindex(nodespace::NodeSpace, i) = nodespace.support[i]

#####################
## WLS ##############
#####################

struct WLS{N, T, B <: Basis} 
    matrix::LU{T, Matrix{T}}
    weights::Vector{T}
    basis::B
end

"""
Uses the Weighted Least Squares method, with a particular weight function and basis to approximate functions on discrete domains.
"""
function wls(nodespace::NodeSpace{N, T}, basis::B, weight::AnalyticFunction) where {N, T, B <: Basis}
    nlength = length(nodespace)
    blength = length(basis)

    matrix = Matrix{T}(undef, blength, nlength)
    weights = Vector{T}(undef, nlength)

    for j in eachindex(nodespace)
        position = nodespace[j]
        weights[j] = weight(position)
        for i in eachindex(basis)
            matrix[i, j] = weights[j]  * basis[i](position)
        end
    end

    WLS{N, T, B}(lu(matrix), weights, basis)
end

function stencil(wls::WLS{N, T}, operator::AnalyticOperator, position::SVector{N, T}) where {N, T}
    blength = length(wls.basis)

    rhs = Vector(undef, blength)

    for i in eachindex(wls.basis)
        rhs[i] = operator(wls.basis[i])(position)
    end

    stencil::Vector{T} = wls.matrix \ rhs
    stencil .*= wls.weights

    stencil
end