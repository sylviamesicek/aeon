#####################
## Exports ##########
####################

export SquareEngine, approx

#####################
## WLS ##############
#####################

"""
Weighted least squares method/decomposition of a domain.
"""
struct SquareEngine{N, T, B, M}
    matrix::M
    basis::B
end

"""
Uses the Weighted Least Squares method, with a particular weight function and basis to approximate functions on discrete domains.
"""
function SquareEngine(domain::Domain{N, T}, basis::B) where {N, T, B <: ABasis{N, T}}
    nlength = length(domain)
    blength = length(basis)

    if nlength ≠ blength
        error("For square engine, nlength and blength must match.")
    end

    matrix = Matrix{T}(undef, blength, nlength)

    for j in eachindex(domain)
        for i in eachindex(basis)
            matrix[i, j] = basis[i](domain[j])
        end
    end

    fact = lu(matrix)

    SquareEngine{N, T, typeof(basis), typeof(fact)}(fact, basis)
end

"""
Approximates a simple functional (which transforms preserves type) using WLS. The result is a `ApproxGeneric`.
"""
function approx(engine::SquareEngine{N, T}, operator::AScalar{N, T}, position::SVector{N, T}) where {N, T}
    blength = length(engine.basis)

    rhs = Vector{T}(undef, blength)

    for i in eachindex(engine.basis)
        rhs[i] = operator(engine.basis[i], position)
    end

    stencil = Vector{T}(undef, blength)

    ldiv!(stencil, engine.matrix, rhs)

    ApproxScalar{N, T}(stencil)
end

"""
Approximates a covariant functional
"""
function approx(engine::SquareEngine{N, T}, operator::ACovariant{N, T, O, L}, position::SVector{N, T}) where {N, T, O, L}
    blength = length(engine.basis)

    tdims = ntuple(_ -> N, Val(O))
    tcoords = CartesianIndices(tdims)

    rhs = StaticArrays.sacollect(STensor{N, Vector{T}, O, L}, Vector{T}(undef, blength) for _ in tcoords)

    for i in eachindex(engine.basis)
        values = operator(engine.basis[i], position)

        for coord in tcoords
            rhs[coord][i] = values.inner[coord]
        end
    end

    stencil = StaticArrays.sacollect(STensor{N, Vector{T}, O, L}, Vector{T}(undef, blength) for _ in tcoords)

    for coord in tcoords
        ldiv!(stencil[coord], engine.matrix, rhs[coord])
    end

    ApproxCovariant(blength, stencil)
end