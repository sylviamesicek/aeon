export Mesh, hyperprism, refine!

"""
The overall topology of a domain. The mesh class
provides a means of discretizing, building, and manpulating numerical domains.
It can be iterated to yield the individual cells of the mesh. 
"""
mutable struct Mesh{N, T}
    cells::Array{Cell{N, T}, N}
    doftotal::Int
end

Base.length(mesh::Mesh) = length(mesh.cells)
Base.eachindex(mesh::Mesh) = eachindex(mesh.cells)
Base.getindex(mesh::Mesh, i) = getindex(mesh.cells, i)
Base.setindex!(mesh::Mesh, cell::Cell, i) = setindex!(mesh.cells, cell, i)

"""
    hyperprism(origin, cells, width)

Builds a mesh consisting of coordinate aligned cells of the given width. This starts at the origin,
and extends by a number of cells (given by the `cells` vector) in each direction.
"""
function hyperprism(origin::SVector{N, T}, widths::SVector{N, T}, celldims::NTuple{N, Int}, baserefinement::Int) where {N, T}
    @assert N > 0

    # Array of final cells for the mesh.
    meshcells = Array{Cell{N, T}, N}(undef, celldims...)

    dofoffset = 0

    for index in CartesianIndices(celldims)
        # Compute position of origin
        position = origin + SVector{N, T}((index.I .- 1) .* widths)
        # Build cell
        cell = Cell(HyperBox(position, widths), baserefinement, dofoffset)
        # Increment offset
        dofoffset += length(cell)
        # Add cell to mesh
        meshcells[index] = cell
    end

    Mesh{N, T}(meshcells, dofoffset)
end

"""
Refine a mesh using a refinement vector. For each `true` in the `shouldrefine` vector, the number of dofs on the corresponding cell is doubled along each
axis.
"""
function refine!(mesh::Mesh{N}, shouldrefine::Array{Bool, N}) where N
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
    shouldrefine = Array{Bool}(undef, size(mesh.cells)...)

    for cell in eachindex(shouldrefine)
        shouldrefine[cell] = pred(cell)
    end

    refine!(mesh, shouldrefine)
end

"""
Smooths the `shouldrefine` vector. Returns true if the vector is already smooth.
"""
function smoothrefine!(mesh::Mesh{N, T}, shouldrefine::Array{Bool, N}) where {N, T}
    smooth = true

    for cell in CartesianIndices(size(mesh.cells))
        if shouldrefine[cell]
            for dim in 1:N
                left = CartesianIndex(cell.I .- ntuple(i -> i == dim, Val(N)))

                if left[dim] > 0 && !shouldrefine[left] && mesh[left].refinement < mesh[cell].refinement
                    smooth = false
                    shouldrefine[left] = true
                end

                right = CartesianIndex(cell.I .+ ntuple(i -> i == dim, Val(N)))

                if right[dim] > 0 && !shouldrefine[right] && mesh[right].refinement < mesh[cell].refinement
                    smooth = false
                    shouldrefine[right] = true
                end
            end
        end
    end

    smooth
end