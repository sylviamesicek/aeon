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

"""
The same as `project!` but allocates an appropriately sized vector. 
"""
function project(f::Function, mesh::Mesh{N, T}, dofs::DoFManager{N, T}) where {N, T} 
    total = dofstotal(dofs)
    v = Vector{T}(undef, total)
    project!(f, v, mesh, dofs)
    return v
end

"""
Projects an analytic function into a numerical vector `v` using the given mesh and dofs manager.
"""
function project!(f::Function, v::AbstractVector{T}, mesh::Mesh{N, T}, dofs::DoFManager{N, T},) where {N, T}
    foreachleafnode(mesh) do level, node
        transform = nodetransform(mesh, level, node)
        offset = nodeoffset(dofs, level, node)
        for (i, cell) in enumerate(cellindices(mesh, level))
            lpos = cellposition(mesh, level, cell)
            gpos = transform(lpos)
            v[offset + i] = f(gpos)
        end
    end
end

##############################
## Block Manager #############
##############################
export BlockManager

"""
Associates an appropriately sized `ArrayBlock` for each level of a mesh. 
"""
struct BlockManager{N, T, O} 
    block::ArrayBlock{N, T, O}
    base::Vector{ArrayBlock{N, T, O}}

    """
    Constructs a block manager compatible with the given mesh.
    """
    function BlockManager{O}(mesh::Mesh{N, T}) where {N, T, O}
        block = ArrayBlock{T, O}(undef, nodecells(mesh, mesh.base + 1)...)
        base = [ArrayBlock{T, O}(undef, nodecells(mesh, c)...) for c in 1:mesh.base]
    
        new{N, T, O}(block, base)
    end
end

Base.getindex(blocks::BlockManager, i::Int) = i ≤ length(blocks.base) ? blocks.base[i] : blocks.block