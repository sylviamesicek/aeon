export DoFLevel, DoFHandler
export nodeoffset, leveltotal, dofstotal

struct DoFLevel{N, T}
    offsets::Vector{Int}
    total::Int
end

struct DoFHandler{N, T} 
    levels::Vector{DoFLevel{N, T}}

    DoFHandler(levels::Vector{DoFLevel{N, T}}) where {N, T} = new{N, T}(levels)
end

function DoFHandler(mesh::Mesh{N, T}) where {N, T}
    levels = DoFLevel{N, T}[]

    total = 0
    dofs_per_node = prod(nodecells(mesh))

    for i in eachindex(mesh)
        level = mesh.levels[i]

        offsets = Vector{Int}(undef, length(level))

        for node in eachindex(level)
            if level.children[node] == -1
                offsets[node] = total
                total += dofs_per_node
            end
        end

        tmp = total

        for node in eachindex(level)
            if level.children[node] ≠ -1
                offsets[node] = tmp
                tmp += dofs_per_node
            end
        end

        push!(levels, DoFLevel{N, T}(offsets, tmp))
    end

    DoFHandler(levels)
end

"""
Returns the offset into a global dof vector for a given node.
"""
nodeoffset(dofs::DoFHandler, level::Int, node::Int) = dofs.levels[level].offsets[node]

"""
Returns the total number of dofs of a level and all coarser levels.
"""
leveltotal(dofs::DoFHandler, level::Int) = dofs.levels[level].total

"""
Returns the total number of dofs of a mesh, including the most refined level.
"""
dofstotal(dofs::DoFHandler) = leveltotal(dofs, length(dofs.levels))

#############################
## Projection ###############
#############################

export project, project!

function project(f::Function, mesh::Mesh{N, T}, dofs::DoFHandler{N, T}) where {N, T} 
    total = dofstotal(dofs)
    v = Vector{T}(undef, total)
    project!(f, mesh, dofs, v)
    return v
end

function project!(f::Function, mesh::Mesh{N, T}, dofs::DoFHandler{N, T}, v::AbstractVector{T}) where {N, T}
    for level in eachindex(mesh)
        for node in eachleafnode(mesh, level)
            transform = nodetransform(mesh, level, node)
            offset = nodeoffset(dofs, level, node)

            for (i, cell) in enumerate(cellindices(mesh))
                lpos = cellposition(mesh, cell)
                gpos = transform(lpos)

                v[offset + i] = f(gpos)
            end
        end
    end
end