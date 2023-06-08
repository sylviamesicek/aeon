#########################
## Exports ##############
#########################

export Mesh, kind, nodeindices
export NodeKind, interior, boundary, ghost, constraint


#########################
## Mesh #################
#########################

@enum NodeKind interior = 0 boundary = 1 ghost = 2 constraint = 3

"""
A `Mesh` is simply a matrix of node positions along with a node kind. Additional context/data is provided by the individual `Method`
"""
struct Mesh{N, T}
    positions::Vector{SVector{N, T}}
    kinds::Vector{NodeKind}

    function Mesh(positions::Vector{SVector{N, T}}, kinds::Vector{NodeKind}) where{N, T}
        if length(positions) ≠ length(kinds)
            error("Position matrix length must equal the length of the kinds vector.")
        end

        new{N, T}(positions, kinds)
    end
end

Base.length(mesh::Mesh) = length(mesh.positions)
Base.eachindex(mesh::Mesh) = eachindex(mesh.kinds)
Base.getindex(mesh::Mesh, i) = mesh.positions[i]
Base.similar(mesh::Mesh{N, T}) where {N, T} = Vector{T}(undef, length(mesh))

function kind(mesh::Mesh, i)
    mesh.kinds[i]
end

function nodeindices(mesh::Mesh, kind::NodeKind)
    nodes = Vector{Int}()

    for i in eachindex(mesh)
        if kind(mesh, i) == kind
            push!(nodes, i)
        end
    end

    nodes
end