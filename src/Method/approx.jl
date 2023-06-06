#####################
## Exports ##########
#####################

export ApproxEngine, stencil
export WLSEngine

#####################
## Interface ########
#####################

abstract type ApproxEngine end

stencil(engine::ApproxEngine, ::AnalyticOperator, ::Vertex) = error("Stencil is unimplemented for $(typeof(engine))")

#####################
## WLS ##############
#####################

struct WLSEngine{N, T, B <: Basis, W <: AnalyticFunction} <: ApproxEngine
    basis::B
    weight::W

    WLSEngine{N, T}(basis::B, weight::W) where {N, T, B, W} = new{N, T, B, W}(basis, weight)
end

function stencil(wls::WLSEngine{N, T, B}, operator::AnalyticOperator, vertex::Vertex) where {N, T, B}
    vlength = length(vertex)
    blength = length(wls.basis)

    m::Matrix{T} = Matrix(undef, blength, vlength)
    w::Vector{T} = Vector(undef, vlength)
    rhs::Vector{T} = Vector(undef, blength)

    for j in eachindex(vertex)
        position = vertex[j]
        w[j] = wls.weight(position)
        for i in eachindex(wls.basis)
            m[i, j] = w[j] * wls.basis[i](position)
        end
    end

    center = center(vertex)

    for i in eachindex(wls.basis)
        rhs[i] = operator(wls.basis[i])(center)
    end

    stencil::Vector{T} = m \ rhs
    stencil .*= w

    stencil
end
