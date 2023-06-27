export CoefficientSource, mass_operator, derivative_operator, boundary_derivative_operator, dissipation_operator


"""
Represents the source of a coefficient set, most often some SBP paper.
"""
abstract type CoefficientSource{T, O} end

mass_operator(source::CoefficientSource) = error("Unimplemented.")
derivative_operator(source::CoefficientSource, ::Val{R}) where { R} = error("Unimplemented")
boundary_derivative_operator(source::CoefficientSource, ::Val{R}) where {R} = error("Unimplemented")
dissipation_operator(source::CoefficientSource) = error("Unimplemented")
prologation_operator(source::CoefficientSource) = error("Unimplemented")
restriction_operator(source::CoefficientSource) = error("Unimplemented")