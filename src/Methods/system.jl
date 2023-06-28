# Exports
export System

# Interface
abstract type System{N, T} end

# apply_operator!(::AbstractVector{T}, mesh::Mesh{N, T}, system::System{N, T}) where {N, T} = error("Operator application is unimplemented for $(typeof(system))")
# compute_rhs!(::AbstractVector{T}, mesh::Mesh{N, T}, system::System{N, T}) where {N, T} = error("RHS is unimplemented for $(typeof(system))")