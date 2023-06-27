export Mesh, hyperprism, refine!

"""
The overall topology of a domain. The mesh class
provides a means of discretizing, building, and manpulating numerical domains.
It can be iterated to yield the individual cells of the mesh. 
"""
mutable struct Mesh{N, T}
    cells::Vector{Cell{N, T}}
    baserefine::SVector{N, Int}
    doftotal::Int
end

Base.length(mesh::Mesh) = length(mesh.cells)
Base.eachindex(mesh::Mesh) = eachindex(mesh.cells)
Base.getindex(mesh::Mesh, i::Int) = getindex(mesh.cells, i)
Base.setindex!(mesh::Mesh, cell::Cell, i::Int) = setindex!(mesh.cells, cell, i)

"""
    hyperprism(origin, cells, width)

Builds a mesh consisting of coordinate aligned cells of the given width. This starts at the origin,
and extends by a number of cells (given by the `cells` vector) in each direction.
"""
function hyperprism(origin::SVector{N, T}, cells::SVector{N, Int}, widths::SVector{N, T}, baserefine::SVector{N, Int}) where {N, T}
    @assert N > 0

    # Indexing objects
    cellindices = CartesianIndices(Tuple(cells))
    carttolinear = LinearIndices(Tuple(cells))
    
    # Vector of final cells for the mesh.
    meshcells = Vector{Cell{N, T}}(undef, length(cellindices))

    dofoffset = 0

    for i in cellindices
        cartindex = Tuple(i)
        # Convert to linear index
        linearindex = carttolinear[i]
        # Compute position of origin
        position = origin + SVector{N, T}((cartindex .- 1) .* widths)
        # Compute face tuple
        faces = ntuple(Val(N)) do dim
            if cartindex[dim] == 1
                negface = 0
            else
                negcoord = cartindex .- ntuple(i -> i == dim, Val(N))
                negface = carttolinear[negcoord...]
                # negface = Face(cellface, negindex)
            end

            if cartindex[dim] == cells[dim]
                # posface = Face(boundaryface, 1)
                posface = 0
            else
                poscoord = cartindex .+ ntuple(i -> i == dim, Val(N))
                posface = carttolinear[poscoord...]
                # posface = Face(cellface, posindex)
            end

            (negface, posface)
        end

        cell = Cell(HyperBox(position, widths), faces, baserefine, 0, dofoffset)
        # Increment offset
        dofoffset += length(cell)
        # Add cell to mesh
        meshcells[linearindex] = cell
    end

    Mesh{N, T}(meshcells, baserefine, dofoffset)
end

"""
Refine a mesh using a refinement vector. For each `true` in the `shouldrefine` vector, the number of dofs on the corresponding cell is doubled along each
axis.
"""
function refine!(mesh::Mesh, shouldrefine::Vector{Bool})
    smooth = false
    while !smooth
        smooth = smoothrefine!(mesh, shouldrefine)
    end

    # We now have smoothed the shouldrefine vector. Recompute cells.

    dofoffset = 0

    for cell in eachindex(mesh)
        if shouldrefine[cell]
            c = mesh[cell]
            mesh[cell] = Cell(c.bounds, c.faces, mesh.baserefine, c.refinement + 1, dofoffset)
        else
            c = mesh[cell]
            mesh[cell] = Cell(c.bounds, c.faces, mesh.baserefine, c.refinement, dofoffset)
        end

        dofoffset += length(mesh[cell])
    end

    mesh.doftotal = dofoffset
end

"""
Refines a mesh based on a functional predicate.
"""
function refine!(pred::Function, mesh::Mesh)
    shouldrefine = [pred(mesh[cell]) for cell in eachindex(mesh)]
    refine!(mesh, shouldrefine)
end

"""
Smooths the `shouldrefine` vector. Returns true if the vector is already smooth.
"""
function smoothrefine!(mesh::Mesh, shouldrefine::Vector{Bool})
    smooth = true

    for cell in eachindex(mesh)
        if shouldrefine[cell]
            for face in mesh[cell].faces
                # Left case
                left = face[1]
                if left > 0 && !shouldrefine[left] && mesh[left].refinement < mesh[cell].refinement
                    smooth = false
                    shouldrefine[left] = true
                end

                # Right case
                right = face[2]
                if right > 0 && !shouldrefine[right] && mesh[right].refinement < mesh[cell].refinement
                    smooth = false
                    shouldrefine[right] = true
                end
            end
        end
    end

    smooth
end