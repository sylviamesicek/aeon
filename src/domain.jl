const pnull = typemax(UInt64)

@enum Kind begin
    boundary=1
    interior=2
    ghost=3
end

struct Domain{S, F<:Real}
    positions::Vector{SVector{S, F}}
    kinds::Vector{Kind}
    
    bmap::Vector{UInt}
    normals::Vector{SVector{S, F}}

    function Domain{S, F}(positions::Vector{SVector{S, F}}, kinds::Vector{Kind}, bmap::Vector{UInt}, normals) where {S, F<:Real}
        new{S, F}(positions, kinds, bmap, normals)
    end
end

function Base.length(domain::Domain{S, F})::UInt where {S, F<:Real}
    length(domain.positions)
end

function Base.eachindex(domain::Domain{S, F}) where {S, F<:Real}
    eachindex(domain.positions)
end

