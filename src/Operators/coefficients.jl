export CoefficientSource, mass_operator, derivative_operator, boundary_derivative_operator
export dissipation_operator, prolongation_operator, restriction_operator


"""
Represents the source of a coefficient set, most often some SBP paper.
"""
abstract type CoefficientSource{T, O} end

"""
Builds the mass operator using a given coefficient source.
"""
mass_operator(source::CoefficientSource) = error("Mass operator unimplemented for $(typeof(source)).")

"""
Builds a derivative operator of a given rank for a coefficient source.
"""
derivative_operator(source::CoefficientSource, ::Val{R}) where { R} = error("Derivative operator of rank $(R) is unimplemented for $(typeof(source)).")

"""
Builds a boundary derivative operator of a given rank for a coefficient source. Used to enforce Nuemann boundary conditions.
"""
boundary_derivative_operator(source::CoefficientSource, ::Val{R}) where {R} = error("Boundary derivative operator of rank $(R) is unimplemented for $(typeof(source)).")

"""
Builds a dissupation operator using the coefficient source.
"""
dissipation_operator(source::CoefficientSource) = error("Dissipation operator for $(typeof(source)) is unimplemented.")

"""
Builds a prologation operator using the coefficient source.
"""
prolongation_operator(source::CoefficientSource) = error("Prolongation operator for $(typeof(source)) is unimplemented.")

"""
Builds a restriction operator using the coefficient source.
"""
restriction_operator(source::CoefficientSource) = error("Restriction operator for $(typeof(source)) is unimplemented.")