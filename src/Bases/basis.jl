struct CellStencil end


abstract type Basis{N, T} end

stencil_order(::Basis) = error("Unimplemented")
value_stencil(::Basis) = error("Unimplemented")
prolong_stencil(::Basis) = error("Unimplemented")
subcell_stencil(::Basis) = error("Unimplemented")

derivative_stencil(::Basis, ::Val{R}) where R = error("Unimplemented")

exterior_order(basis::Basis) = stencil_order(basis)
interior_order(basis::Basis) = 2stencil_order(basis)
boundary_value_stencils(::Basis) where R = error("Unimplemented")