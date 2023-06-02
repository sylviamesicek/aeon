export Space

"""
`Space` defines a set of types and variables for interacting and discritizing
problem spaces. This includes tools for visiualizing functions on domains, 
generating domains using various algorithms (uniform grid, uniform fill, etc.), 
and defining functions on domains
"""
module Space
    
# Dependencies
using LinearAlgebra
using StaticArrays

# Includes
include("domain.jl")
include("grid.jl")
include("writer.jl")

end