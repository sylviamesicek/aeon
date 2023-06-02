export Engine

"""
The `Engine` module contains code for solving PDEs on domains.
"""
module Engine

# Dependencies
using Distances
using LinearAlgebra
using NearestNeighbors
using StaticArrays
using SparseArrays

# Other Modules
using Aeon
using Aeon.Space

# Includes
include("wls.jl")
    
end