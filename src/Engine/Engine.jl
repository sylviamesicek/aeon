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
using Aeon.Analytic
using Aeon.Space

# Core

abstract type Mesh end

abstract type MeshOperator end

abstract type MeshFunction end



prepare_matrix!(mesh, matrix) = 0
prepare_rhs!(mesh, rhs) = 0

assemble!(func, mesh, system, rhs) = 0

struct System{Func<:Function, F}
    operator::Func
    rhs::F
end


# Includes
include("basis.jl")
include("wls.jl")
    
end