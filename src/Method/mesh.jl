#########################
## Exports ##############
#########################

export Mesh, filterindices

#########################
## Mesh #################
#########################

"""
A `Mesh` is simply a vector of points: consisting of a position, a kind, and a tag. It stores data in a SoA format for maximal cache efficiency.
"""
struct Mesh{N, T}
    positions::Vector{SVector{N, T}}
    kinds::Vector{Int}
    tags::Vector{Int}

    function Mesh(positions::Vector{SVector{N, T}}, kinds::Vector{Int}, tags::Vector{Int}) where{N, T}
        @assert length(positions) == length(kinds) == length(tags)
        new{N, T}(positions, kinds, tags)
    end
end

Base.length(mesh::Mesh) = length(mesh.positions)
Base.eachindex(mesh::Mesh) = eachindex(mesh.kinds)
Base.similar(mesh::Mesh{N, T}) where {N, T} = Vector{T}(undef, length(mesh))

"""
Filters the points in a `Mesh` by its kind. This returns a vector which can be iterated to yield to 
"""
function filterindices(mesh::Mesh, kind::Int)
    nodes = Vector{Int}()

    for i in eachindex(mesh)
        if mesh.kinds[i] == kind
            push!(nodes, i)
        end
    end

    nodes
end