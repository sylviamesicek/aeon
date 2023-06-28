export Mesh, hyperprism, refine!

"""
The overall topology of a domain. The mesh class
provides a means of discretizing, building, and manpulating numerical domains.
It can be iterated to yield the individual cells of the mesh. 
"""
mutable struct Mesh{N, T}
    cells::Vector{Cell{N, T}}
    celldims::NTuple{N, Int}
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
function hyperprism(origin::SVector{N, T}, widths::SVector{N, T}, celldims::NTuple{N, Int}, baserefinement::Int) where {N, T}
    @assert N > 0

    # Indexing objects
    cellindices = CartesianIndices(celldims)
    carttolinear = LinearIndices(celldims)
    
    # Vector of final cells for the mesh.
    meshcells = Vector{Cell{N, T}}(undef, length(cellindices))

    dofoffset = 0

    for i in cellindices
        cartindex = Tuple(i)
        # Convert to linear index
        linearindex = carttolinear[i]
        # Compute position of origin
        position = origin + SVector{N, T}((cartindex .- 1) .* widths)
        # Build cell
        cell = Cell(HyperBox(position, widths), baserefinement, dofoffset)
        # Increment offset
        dofoffset += length(cell)
        # Add cell to mesh
        meshcells[linearindex] = cell
    end

    Mesh{N, T}(meshcells, celldims, dofoffset)
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
            mesh[cell] = Cell(c.bounds, c.refinement + 1, dofoffset)
        else
            c = mesh[cell]
            mesh[cell] = Cell(c.bounds, c.refinement, dofoffset)
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
function smoothrefine!(mesh::Mesh{N, T}, shouldrefine::Vector{Bool}) where {N, T}
    smooth = true

    cellindices = CartesianIndices(mesh.celldims)
    carttolinear = LinearIndices(mesh.celldims)

    for cell in cellindices
        celllinear = carttolinear[cell]

        if shouldrefine[celllinear]
            for dim in 1:N
                left = CartesianIndex(index.I .- ntuple(i -> i == dim, Val(N)))
                leftlinear = carttolinear[left]

                if left[dim] > 0 && !shouldrefine[leftlinear] && mesh[leftlinear].refinement < mesh[celllinear].refinement
                    smooth = false
                    shouldrefine[leftlinear] = true
                end

                right = CartesianIndex(index.I .+ ntuple(i -> i == dim, Val(N)))
                rightlinear = carttolinear[right]

                if right[dim] > 0 && !shouldrefine[rightlinear] && mesh[rightlinear].refinement < mesh[celllinear].refinement
                    smooth = false
                    shouldrefine[rightlinear] = true
                end
            end
        end
    end

    smooth
end