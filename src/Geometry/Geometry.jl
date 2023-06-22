export Geometry

"""
A module for generically handling various geometric concepts. Including coordinate transformations
tensors of arbitrary rank and dimension, axis-aligned bounding boxes and hyper-octrees.
"""
module Geometry
    
# Dependences
using StaticArrays
using LinearAlgebra

# Includes
include("transform.jl")
include("affine.jl")
include("tensor.jl")

include("splitarray.jl")
include("box.jl")
include("tree.jl")

end