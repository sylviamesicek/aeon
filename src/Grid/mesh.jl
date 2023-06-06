##########################
## Exports ###############
##########################

export GridMesh, GridLevel

##########################
## Grid ##################
##########################

struct GridLevel{N, T}
    positions::Vector{SVector{N, T}}
    # locals_to_globals::Vector{Vector{Int}}

    # The length of this field encodes the number of vertices that are also on the previous grid level.
    # False indicates they are unrefined and use coarsewidth, while true indicates they use refinedwidth.
    # The vertices from length(refined):length(vertices) are implicitly refined, as they did not appear
    # on the previous level.
    refined::Vector{Bool}

    coarsewidth::T
    refinedwidth::T
end

Base.length(level::GridLevel) = length(level.positions)
Base.eachindex(level::GridLevel) = eachindex(level.positions)
Base.getindex(level::GridLevel, i) = getindex(level.positions, i)
Base.similar(level::GridLevel{N, T}) where {N, T} = Vector{T}(undef, length(level))

coarselength(level::GridLevel) = length(level.refined)
coarsevertices(level::GridLevel) = eachindex(level.refined)
coarseisrefined(level::GridLevel, i) = level.refined[i]

"""
Represents a mesh as a hyper-rectangular grid of vertices. This allows for many optimatizations
including, given the regularity of the system, for the stencil and constraints to only be computed once, 
and cached across degrees of freedom. This makes matrix assembly essentially free, and allows 
the matrix-free algorithm to be immensly quicker.

Remember that the base size given to the grid at construction is the coarsest level of the grid.
This means that this level must be solvable without a preconditioner.
"""
struct GridMesh{N, T}
    # Vertices organized such that the first N vertices constitute the coarsest mesh
    levels::Vector{GridLevel{N, T}}

    function GridMesh(origin::SVector{N, T}, dims::SVector{N, Int}, width::T, ::Int) where {N, T}
        vertexcount = prod(dims)
        positions = Vector{SVector{N, T}}(undef, vertexcount)
        refined = Vector{Bool}(undef, vertexcount)

        for (i, I) in enumerate(CartesianIndices(Tuple(dims) .+ 1))
            positions[i] = origin .+ (Tuple(I) .- 1) .* width
            refined[i] = false
        end

        new{N, T}([GridLevel(positions, refined, width, width)])
    end
end

Base.length(grid::GridMesh) = length(grid.levels)
Base.eachindex(grid::GridMesh) = eachindex(grid.levels)
Base.getindex(grid::GridMesh, i) = getindex(grid.levels, i)
Base.firstindex(grid::GridMesh) = firstindex(grid.levels)
Base.lastindex(grid::GridMesh) = lastindex(grid.levels)
