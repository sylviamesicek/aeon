#####################
## Exports ##########
#####################

export ApproxStencil, ApproxEngine, stencil
export WLSEngine, WLSStencil

#####################
## Interface ########
#####################

"""
A stencil for approximating an analytic operator acting on a function defined on the support of a vertex.
"""
abstract type ApproxStencil end

abstract type ApproxEngine end

#####################
## WLS ##############
#####################

struct WLSStencil{D, F, B <: Basis}
    stencil::Vector{F}
end

struct WLSEngine{D, F, B <: Basis, W <: AnalyticFunction} <: ApproxEngine
    basis::B
    weight::W

    WLSEngine{D, F}(basis::B, weight::W) where {D, F, B, W} = new{D, F, B, W}(basis, weight)
end

function stencil(wls::WLSEngine{D, F, B}, vertex::Vertex{D, F}, operator::AnalyticOperator) where {D, F, B}
    center = vertex[vertex.primary]

    vlength = length(vertex)
    blength = length(wls.basis)

    m::Matrix{F} = Matrix(undef, blength, vlength)
    w::Vector{F} = Vector(undef, vlength)
    rhs::Vector{F} = Vector(undef, blength)

    for j in eachindex(vertex)
        position = vertex[j]
        w[j] = wls.weight(position)
        for i in eachindex(wls.basis)
            m[i, j] = w[j] * wls.basis[i](position)
        end
    end

    for i in eachindex(wls.basis)
        rhs[i] = operator(wls.basis[i])(center)
    end

    display(m)
    display(rhs)

    stencil = m \ rhs
    stencil .*= w

    WLSStencil{D, F, B}(stencil)
end
