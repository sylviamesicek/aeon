export CoefficientSource, mass, derivative, boundary_derivative, dissipation


"""
Represents the source of a coefficient set, most often some SBP paper.
"""
abstract type CoefficientSource end

mass(::Val{T}, ::Val{O}, source::CoefficientSource) = error("Unimplemented.")
derivative(::Val{T}, ::Val{O}, ::Val{R}, source::CoefficientSource) where {T, O, R} = error("Unimplemented")
boundary_derivative(::Val{T}, ::Val{O}, ::Val{R}, source::CoefficientSource) where {T, O, R} = error("Unimplemented")
dissipation(::Val{T}, ::Val{O}, source::CoefficientSource) where {T, O} = error("Unimplemented")
