## Exports
export Domain

## Core 
struct Domain{N, T}
    positions::Vector{SVector{N, T}}

    Domain(positions::Vector{SVector{N, T}}) where {N, T} = new{N, T}(positions)
end

Base.length(domain::Domain) = length(domain.positions)
Base.eachindex(domain::Domain) = eachindex(domain.positions)
Base.getindex(domain::Domain, i) = domain.positions[i]

