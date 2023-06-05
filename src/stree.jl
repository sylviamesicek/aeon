#####################
## Exports ##########
#####################

export STree, isleaf, children, parent, alltrees, allleaves, allparents, split!, findleaf
export HyperRectangle, vertices, center

#####################
## Rectangle ########
#####################

struct HyperRectangle{N, T}
    origin::SVector{N, T}
    widths::SVector{N, T}
end

@generated function vertices(rect::HyperRectangle{N}) where N
    verts = Expr[]
    for I in CartesianIndices(tuple([2 for i in 1:N]...))
        push!(verts,
        Expr(:call, :SVector, [I[i] == 1 ? :(rect.origin[$i]) : :(rect.origin[$i] + rect.widths[$i]) for i in 1:N]...))
    end
    Expr(:call, :SplitArray, verts...)
end

center(rect::HyperRectangle) = rect.origin + 0.5 * rect.widths

######################
## STree #############
######################

mutable struct STree{N, T, L, Data}
    boundary::HyperRectangle{N, T}
    data::Data
    center::SVector{N, T}
    children::Union{SplitArray{N, STree{N, T, L, Data}, L}, Nothing}
    parent::Union{STree{N, T, L, Data}, Nothing}
end

function STree(origin::SVector{N, T}, widths::SVector{N, T}, data::Data=nothing) where {N, T, Data}
    STree(HyperRectangle(origin, widths), data)
end

function STree(boundary::HyperRectangle{N, T}, data::Data=nothing) where {N, T, Data}
    STree{N, T, 2^N, Data}(boundary, data, center(boundary), nothing, nothing)
end

@inline isleaf(tree::STree) = tree.children === nothing
@inline children(tree::STree) = tree.children
@inline parent(tree::STree) = tree.parent
@inline center(tree::STree) = tree.center
@inline vertices(tree::STree) = vertices(tree.boundary)

Base.show(io::IO, tree::STree) = print(io, "STree: $(tree.boundary)")

Base.getindex(tree::STree) = tree
Base.getindex(tree::STree, ::CartesianIndex{0}) = tree
Base.getindex(tree::STree, i) = getindex(tree.children, i)
Base.getindex(tree::STree, i...) = getindex(tree.children, i...)

function alltrees(tree::STree)
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

function allleaves(tree::STree)
    Channel() do c
        for child in alltrees(tree)
            if isleaf(child)
                put!(c, child)
            end
        end
    end
end

function allparents(tree::STree)
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

split!(tree::STree{N}) where {N} = split!(tree, (c, I) -> tree.data)
split!(tree::STree, child_data_function::Function) = split!(tree, map_children(child_data_function, tree))

@generated split!(tree::STree{N}, child_data::AbstractArray) where {N} = split!_impl(tree, child_data, Val(N))

@generated function findleaf(tree::STree{N, T, L, Data}, point::AbstractVector) where {N, T, L, Data}
    quote
        while true
            if isleaf(tree)
                return tree
            end
            length(point) == $N || throw(DimensionMismatch("expected a point of length $N"))
            @inbounds tree = $(Expr(:ref, :tree, [:(ifelse(point[$i] >= tree.center[$i], 2, 1)) for i in 1:N]...))
        end
    end
end

##############################
## Implementation ############
##############################

@generated function map_children(f::Function, cell::STree{N}) where N
    Expr(:call, :SplitArray, 
        Expr(:tuple,[:(f(cell, $(I.I))) for I in CartesianIndices(ntuple(_ -> 2, Val(N)))]...)
    )
end

function child_boundary(tree::STree, indices)
    half_widths = tree.boundary.widths ./ 2
    HyperRectangle(
        tree.boundary.origin .+ (SVector(indices) .- 1) .* half_widths,
        half_widths)
end

@generated function child_indices(::Val{N}) where N
    Expr(:call, :SplitArray, 
        Expr(:tuple, [I.I for I in CartesianIndices(ntuple(_ -> 2, Val(N)))]...)
    )
end

child_indices(::STree{N, T, L, Data}) where {Data, N, T, L} = child_indices(Val(N))

function split!_impl(::Type{C}, child_data, ::Val{N}) where {C <: STree, N}
    child_exprs = [:(STree(child_boundary(tree, $(I.I)),
                          child_data[$i])) for (i, I) in enumerate(CartesianIndices(ntuple(_ -> 2, Val(N))))]
    quote
        @assert isleaf(tree)
        tree.children = $(Expr(:call, :SplitArray, Expr(:tuple, child_exprs...)))
        for child in tree.children
            child.parent = tree
        end
        tree
    end
end

