export HyperTree
export isleaf, isroot
export child_indices, split!

mutable struct HyperTree{N, T, L}
    children::Union{SplitArray{N, HyperTree{N, T, L},  L}, Nothing}
    parent::Union{HyperTree{N, T, L}, Nothing}
    data::T
end

function HyperTree{N}(data::T) where {N, T}
    HyperTree{N, T, 2^N}(nothing, nothing, data)
end

@inline isleaf(node::HyperTree) = node.children === nothing
@inline isroot(node::HyperTree) = node.parent === nothing

child_indices(::HyperTree{N}) where N = child_indices(Val(N))
child_indices(::Val{N}) where N = CartesianIndices(ntuple(_ -> 2, Val(N)))

@generated function map_children(f::Function, node::HyperTree{N, T}) where {N, T}
    child_exprs = [:(f(node.children[$I].data, $(I))) for I in child_indices(Val(N))]
    
    quote
        @assert !isleaf(node)
        
        SArray{NTuple{N, 2}, T, N}($(child_exprs)...)
    end
end

function split!(node::HyperTree{N, T}, data::SArray{NTuple{N, 2}, T, N}) where {N, T}
    @assert isleaf(node)

    node.children = map(HyperTree{N}, data)
    for child in node.children
        child.parent = node
    end
    node
end

@generated function split!(f::Function, node::HyperTree{N, T}) where {N, T}
    child_exprs = [:(f(node.data, $(I))) for I in child_indices(Val(N))]

    quote
        @assert isleaf(node)
        
        SArray{NTuple{N, 2}, T, N}($(child_exprs)...)
    end
end


# @generated function split!(node::HyperTree{N, T}, data::SArray{NTuple{N, 2}, T, N}) where {N, T}
#     child_exprs = [:(HyperTree{N}(data[$I])) for I in child_indices(Val(N))]
#     quote
#         @assert isleaf(node)
        
#         node.children = StaticArrays.sacollect(SArray{NTuple{N, 2}, T, N}, $(child_exprs)...)
#         for child in node.children
#             child.parent = node
#         end
#         node
#     end
# end

