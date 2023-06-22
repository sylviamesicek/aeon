export Cell, Face, Mesh
export cellface, boundaryface
export hyperprism

"""
A face between two `Cell`s.
"""
const cellface::Int = 0
"""
A face between a `Cell` and boundary.
"""
const boundaryface::Int = 1

"""
A face in a mesh.
"""
struct Face
    # Indicates what type of face this is.
    kind::Int
    # An index, either into the boundary array or the cell vector
    index::Int
end

"""
Represents a cell in a mesh. Eventually this will be extended to arbitrary 
hyper-quadralaterials, but for now it is simply a hyperrectangle with a certain
width and origin.
"""
struct Cell{N, T} 
    bounds::HyperBox{N, T}
    faces::NTuple{N, NTuple{2, Face}}
end

"""
The overall topology of a domain, unconnected to any DoFs or interpretation. The mesh class
provides a Method-agnostic means of discretizing, building, and manpulating numerical domains.
It can be iterated to yield the individual cells of the mesh. `Mesh`s are usually passed to
individual `Method`s which handle DoFs, refinement, ect.
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
function hyperprism(origin::SVector{N, T}, cells::SVector{Int, T}, width::T) where {N, T}
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
        position = origin + SVector{N, T}((cartindex .- 1) .* width)
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
        meshcells[linearindex] = Cell(HyperBox(position, width), faces)
    end

    Mesh{N, T}(meshcells)
end