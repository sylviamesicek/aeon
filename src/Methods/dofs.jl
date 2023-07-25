export DoFLevel, DoFManager
export nodeoffset, leveltotal, dofstotal

struct DoFLevel{N, T}
    offsets::Vector{Int}
    total::Int
end

"""
Stores per level dof offsets for a mesh.
"""
struct DoFManager{N, T} 
    levels::Vector{DoFLevel{N, T}}
    # DoFHandler(levels::Vector{DoFLevel{N, T}}, refinement::Int, coarserefinement) where {N, T} = new{N, T}(levels, refinement)
end

function DoFManager(mesh::Mesh{N, T}) where {N, T}
    levels = DoFLevel{N, T}[]

    total = 0

    for i in eachindex(mesh)
        dofs_per_node = prod(nodecells(mesh, i))

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

    DoFManager(levels)
end

"""
Returns the offset into a global dof vector for a given node.
"""
nodeoffset(dofs::DoFManager, level::Int, node::Int) = dofs.levels[level].offsets[node]

"""
Returns the total number of dofs of a level and all coarser levels.
"""
leveltotal(dofs::DoFManager, level::Int) = dofs.levels[level].total

"""
Returns the total number of dofs of a mesh, including the most refined level.
"""
dofstotal(dofs::DoFManager) = leveltotal(dofs, length(dofs.levels))

#############################
## Projection ###############
#############################

export project, project!

function project(f::Function, mesh::Mesh{N, T}, dofs::DoFManager{N, T}) where {N, T} 
    total = dofstotal(dofs)
    v = Vector{T}(undef, total)
    project!(f, mesh, dofs, v)
    return v
end

function project!(f::Function, mesh::Mesh{N, T}, dofs::DoFManager{N, T}, v::AbstractVector{T}) where {N, T}
    foreachleafnode(mesh) do level, node
        transform = nodetransform(mesh, level, node)
        offset = nodeoffset(dofs, level, node)
        for (i, cell) in enumerate(cellindices(mesh))
            lpos = cellposition(mesh, cell)
            gpos = transform(lpos)
            v[offset + i] = f(gpos)
        end
    end
end

##############################
## Block Manager #############
##############################
export BlockManager

struct BlockManager{N, T, O} 
    block::ArrayBlock{N, T, O}
    base::Vector{ArrayBlock{N, T, O}}
end

function BlockManager{O}(mesh::Mesh{N, T}) where {N, T, O}
    b = baselevels(mesh)
    block = ArrayBlock{T, O}(nodecells(mesh, b)...)
    base = [ArrayBlock{T, O}(nodecells(mesh, c)...) for c in 1:(b - 1)]

    BlockManager{N, T, O}(block, base)
end

Base.getindex(blocks::BlockManager, i::Int) = i ≤ length(block.base) ? blocks.base[i] : blocks.block