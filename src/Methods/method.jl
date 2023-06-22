###############
## Exports ####
###############


###############
## Elements ###
###############

"""
An individual element of a fdmethod.
"""
struct Element{N, T}
    # Bounds of the element
    bounds::HyperBox{N, T}
    # Faces of this element
    faces::NTuple{N, NTuple{2, Face}}
    # Number of dofs along each dimension
    dofs::NTuple{N, Int}
end

"""
The values of a field on an fdelement.
"""
struct Field{N, T} 
    values::Array{T, N}
end

function element_to_world(element::Element{N, T}, ::NTuple{N, T}) where {N, T}
    UniformScaleTransform{N, T}(element.bounds.widths[1] / element.dofs[1])
end

# Generic Functions

function stencil_with_offset(field::Field{N, T}, point::NTuple{N, Int}, di::Int, D::SBPDerivative{T, O}) where {N, T, O}
    dimoffset = size(field)[di] + 1

    if point[di] ≤ length(D.left_block)
        coefficientrow = point[di]
        stenciloff = 0
        stencil = D.left_block[coefficientrow]
    elseif point[di] ≥ dimoffset - length(D.right_block)
        coefficientrow = point[di] - (dimoffset - length(D.right_block)) + 1
        stenciloff = dimoffset - length(D.right_block)
        stencil = D.right_block[coefficientrow]
    else
        stenciloff = point[di] - O - 1
        stencil = D.central_coefs
    end

    return stencil, stenciloff
end

function derivative(field::Field{N, T}, point::NTuple{N, Int}, di::Int, D::SBPDerivative{T, O}) where {N, T, O}
    stencil, stenciloff = stencil_with_offset(field, point, di, D)

    result = zero(T)

    for i in eachindex(stencil)
        off = ntuple(Val(N)) do p
            ifelse(p == di, stenciloff + i, point[p])
        end
        result += stencil[i] * field.values[off...]
    end

    result
end

function derivative_mixed(field::Field{N, T}, point::NTuple{N, Int}, di::Int, dj::Int, D::SBPDerivative{T, O}) where {N, T, O}
    stencili, stenciloffi = stencil_with_offset(field, point, di, D)
    stencilj, stenciloffj = stencil_with_offset(field, point, dj, D)

    result = zero(T)

    for i in eachindex(stencili)
        offi = ntuple(Val(N)) do p
            ifelse(p == di, stenciloffi + i, point[p])
        end

        for j in eachindex(stencilj)
            off = ntuple(Val(N)) do p
                ifelse(p == dj, stenciloffj + j, offi[p])
            end

            result += stencili[i] * stencilj[j] * field.values[offi...]
        end
    end

    result
end


# Gradient
function gradient(::Element{N, T}, field::Field{N, T}, point::NTuple{N, Int}, derivatives::SBPDerivatives{T, O}) where {N, T, O}
    grad = ntuple(Val(N)) do dim
        derivative(field, point, dim, D[1])
    end

    SVector(grad)
end

# Second derivatives
function hessian_component(field::Field{N, T}, point::NTuple{N, Int}, di::Int, dj::Int, derivatives::SBPDerivatives{T, O}) where {N, T, O}
    if di == dj
        return derivative(field, point, di, derivatives[2])
    else
        return derivative_mixed(field, point, di, dj, derivatives[1])
    end
end

function hessian(::Element{N, T}, field::Field{N, T}, point::NTuple{N, Int}, derivatives::SBPDerivatives{T, O}) where {N, T, O}
    StaticArrays.sacollect(SMatrix{N, N, T, N^2}, hessian_component(field, point, i, j, derivatives) for i in 1:N, j in 1:N)
end

struct Method{N, T}
    # The tree of elements
    elements::Vector{Element{N, T}}
    # Base number of DoFs along the edges of an element
    basedofs::Int
end

function Method(mesh::Mesh{N, T}, basedofs::Int) where {N, T}
    elements = map(mesh.cells) do c
        Element(c.bounds, c.faces, basedofs)
    end
    Method(elements, basedofs)
end


##################
## Method ########
##################