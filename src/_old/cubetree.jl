#####################
## Exports ##########
#####################

export CubeTree
export isleaf, children, parent, center, vertices
export alltrees, allleaves, allparents, findleaf, split!

######################
## CubeTree ##########
######################

mutable struct CubeTree{N, T, L, Data}
    origin::SVector{N, T}
    width::T
    data::Data
    children::Union{SplitArray{N, CubeTree{N, T, L, Data}, L}, Nothing}
    parent::Union{CubeTree{N, T, L, Data}, Nothing}
    depth::Int
end

function CubeTree(origin::SVector{N, T}, width::SVector{N, T}, data::Data) where {N, T, Data}
    CubeTree{N, T, 2^N, Data}(origin, width, data, nothing, 0)
end

@inline isleaf(tree::CubeTree) = tree.children === nothing
@inline children(tree::CubeTree) = tree.children
@inline parent(tree::CubeTree) = tree.parent
@inline center(tree::CubeTree) = tree.origin .+ tree.width/2

@generated function vertices(tree::CubeTree{N}) where N
    verts = Expr[]
    for I in CartesianIndices(tuple([2 for _ in 1:N]...))
        push!(verts,
            Expr(:call, :SVector, [I[i] == 1 ? :(tree.origin[$i]) : :(tree.origin[$i] .+ tree.width[$i]) for i in 1:N]...)
        )
    end
    Expr(:call, :SplitArray, verts...)
end

Base.show(io::IO, tree::CubeTree) = print(io, "CubeTree: origin = $(tree.origin), width = $(tree.width)")

Base.getindex(tree::CubeTree) = tree
Base.getindex(tree::CubeTree, ::CartesianIndex{0}) = tree
Base.getindex(tree::CubeTree, i) = getindex(tree.children, i)
Base.getindex(tree::CubeTree, i...) = getindex(tree.children, i...)

function alltrees(tree::CubeTree)
    Channel() do c
        queue = [tree]
        while !isempty(queue)
            current = pop!(queue)
            put!(c, current)
            if !isleaf(current)
                append!(queue, children(current))
            end
        end
    end
end

function allleaves(tree::CubeTree)
    Channel() do c
        for child in alltrees(tree)
            if isleaf(child)
                put!(c, child)
            end
        end
    end
end

function allparents(tree::CubeTree)
    Channel() do c
        queue = [tree]
        while !isempty(queue)
            current = pop!(queue)
            p = parent(current)
            if ! (p === nothing)
                put!(c, p)
                push!(queue, p)
            end
        end
    end
end

split!(tree::CubeTree{N}) where N = split!(tree, (c, I) -> tree.data)
split!(tree::CubeTree, child_data_function::Function) = split!(tree, map_children(child_data_function, tree))

@generated split!(tree::CubeTree{N}, child_data::AbstractArray) where N = split!_impl(tree, child_data, Val(N))

@generated function findleaf(tree::CubeTree{N, T, L, Data}, point::AbstractVector) where {N, T, L, Data}
    quote
        while true
            if isleaf(tree)
                return tree
            end
            length(point) == $N || throw(DimensionMismatch("expected a point of length $N"))
            @inbounds tree = $(Expr(:ref, :tree, [:(ifelse(point[$i] >= center(tree)[$i], 2, 1)) for i in 1:N]...))
        end
    end
end

child_indices(::CubeTree{N, T, L, Data}) where {Data, N, T, L} = child_indices(Val(N))

##############################
## Implementation ############
##############################

@generated function map_children(f::Function, tree::CubeTree{N}) where N
    Expr(:call, :SplitArray, 
        Expr(:tuple,[:(f(tree, $(I.I))) for I in CartesianIndices(ntuple(_ -> 2, Val(N)))]...)
    )
end

@generated function child_indices(::Val{N}) where N
    Expr(:call, :SplitArray, 
        Expr(:tuple, [I.I for I in CartesianIndices(ntuple(_ -> 2, Val(N)))]...)
    )
end

function build_child(tree::CubeTree, indices, data)
    half_width = tree.width / 2
    origin = tree.origin .+ (SVector(indices) .- 1) .* half_width

    CubeTree(origin, half_width, data)
end

function split!_impl(::Type{C}, child_data, ::Val{N}) where {C <: CubeTree, N}
    child_exprs = [:(build_child(tree, $(I.I), child_data[$i])) for (i, I) in enumerate(CartesianIndices(ntuple(_ -> 2, Val(N))))]
    quote
        @assert isleaf(tree)
        tree.children = $(Expr(:call, :SplitArray, Expr(:tuple, child_exprs...)))
        for child in tree.children
            child.parent = tree
            child.depth = tree.depth + 1
        end
        tree
    end
end

