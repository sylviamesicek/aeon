#####################
## Exports ##########
####################

export WLSEngine, approx

#####################
## WLS ##############
#####################

"""
Weighted least squares method/decomposition of a domain.
"""
struct WLSEngine{N, T, B, M}
    matrix::M
    weights::Vector{T}
    basis::B
end

"""
Uses the Weighted Least Squares method, with a particular weight function and basis to approximate functions on discrete domains.
"""
function WLSEngine(domain::Domain{N, T}, basis::B, weight::AFunction{N, T}) where {N, T, B <: ABasis{N, T}}
    nlength = length(domain)
    blength = length(basis)

    matrix = Matrix{T}(undef, blength, nlength)
    weights = Vector{T}(undef, nlength)

    for j in eachindex(domain)
        weights[j] = weight(domain[j])
        for i in eachindex(basis)
            matrix[i, j] = weights[j] * basis[i](domain[j])
        end
    end

    fact = qr(matrix)

    WLSEngine{N, T, typeof(basis), typeof(fact)}(fact, weights, basis)
end

"""
Approximates a simple functional (which transforms preserves type) using WLS. The result is a `ApproxGeneric`.
"""
function approx(engine::WLSEngine{N, T}, operator::AScalar{N, T}, position::SVector{N, T}) where {N, T}
    blength = length(engine.basis)
    nlength = length(engine.weights)

    rhs = Vector{T}(undef, blength)

    for i in eachindex(engine.basis)
        rhs[i] = operator(engine.basis[i], position)
    end

    stencil = Vector{T}(undef, nlength)

    ldiv!(stencil, engine.matrix, rhs)
    stencil .*= engine.weights

    ApproxScalar{N, T}(stencil)
end

"""
Approximates a covariant functional
"""
function approx(engine::WLSEngine{N, T}, operator::ACovariant{N, T, O, L}, position::SVector{N, T}) where {N, T, O, L}
    blength = length(engine.basis)
    nlength = length(engine.weights)

    tdims = ntuple(_ -> N, Val(O))
    tcoords = CartesianIndices(tdims)

    rhs = StaticArrays.sacollect(STensor{N, Vector{T}, O, L}, Vector{T}(undef, blength) for _ in tcoords)

    for i in eachindex(engine.basis)
        values = operator(engine.basis[i], position)

        for coord in tcoords
            rhs[coord][i] = values.inner[coord]
        end
    end

    stencil = StaticArrays.sacollect(STensor{N, Vector{T}, O, L}, Vector{T}(undef, nlength) for _ in tcoords)

    for coord in tcoords
        ldiv!(stencil[coord], engine.matrix, rhs[coord])

        stencil[coord] .*= engine.weights
    end

    ApproxCovariant(nlength, stencil)
end