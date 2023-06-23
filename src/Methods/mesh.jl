export Mesh, hyperprism

"""
The overall topology of a domain. The mesh class
provides a means of discretizing, building, and manpulating numerical domains.
It can be iterated to yield the individual cells of the mesh. 
"""
struct Mesh{N, T}
    cells::Vector{Cell{N, T}}
end

Base.length(mesh::Mesh) = length(mesh.cells)
Base.eachindex(mesh::Mesh) = eachindex(mesh.cells)
Base.getindex(mesh::Mesh, i::Int) = getindex(mesh.cells, i)

"""
    hyperprism(origin, cells, width)

Builds a mesh consisting of coordinate aligned cells of the given width. This starts at the origin,
and extends by a number of cells (given by the `cells` vector) in each direction.
"""
function hyperprism(origin::SVector{N, T}, cells::SVector{N, Int}, widths::SVector{N, T}, dofs::SVector{N, Int}) where {N, T}
    @assert N > 0

    # Indexing objects
    cellindices = CartesianIndices(Tuple(cells))
    carttolinear = LinearIndices(Tuple(cells))
    
    # Vector of final cells for the mesh.
    meshcells = Vector{Cell{N, T}}(undef, length(cellindices))

    for i in cellindices
        cartindex = Tuple(i)
        # Convert to linear index
        linearindex = carttolinear[i]
        # Compute position of origin
        position = origin + SVector{N, T}((cartindex .- 1) .* widths)
        # Compute face tuple
        faces = ntuple(Val(N)) do dim
            if cartindex[dim] == 1
                negface = Face(boundaryface, 1)
            else
                negcoord = cartindex .- ntuple(i -> i == dim, Val(N))
                negindex = carttolinear[negcoord...]
                negface = Face(cellface, negindex)
            end

            if cartindex[dim] == cells[dim]
                posface = Face(boundaryface, 1)
            else
                poscoord = cartindex .+ ntuple(i -> i == dim, Val(N))
                posindex = carttolinear[poscoord...]
                posface = Face(cellface, posindex)
            end

            (negface, posface)
        end
        # Add cell to mesh
        meshcells[linearindex] = Cell(HyperBox(position, widths), faces, dofs)
    end

    Mesh{N, T}(meshcells)
end