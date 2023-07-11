export Geometry

"""
A module for handling various Geometrical objects in an `N` dimensional space. This includes coordinate transformations,
axis aligned bounding boxes (`HyperBox`s), and index manipulation for faces and subdivisions of a box.
"""
module Geometry
    
using StaticArrays
using LinearAlgebra

# Includes
include("index.jl")
include("transform.jl")
include("affine.jl")
include("box.jl")

end