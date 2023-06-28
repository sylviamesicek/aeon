
struct Interface{T, O, P, R, B} 
    prolong::P
    restrict::R
    boundary::B
    strength::T

    function Interface(prolong::ProlongationOperator{T, O}, restrict::RestrictionOperator{T, O}, boundary::Operator{T, O}, strength::T) where {T, O}
        new{T, O, typeof(prolong), typeof(restrict), typeof(boundary)}(prolong, restrict, boundary, strength)
    end
end

function evaluate_prolong(point::CartesianIndex{N}, oper::Interface{T, O}, func::AbstractArray{T, N}, other::AbstractArray{T, N}, axis::Int, side::Bool) where {N, T, O}
    opers_prolong = ntuple(dim -> dim == meta.axis ? oper.boundary : oper.prolong, Val(N))
    # opers_restrict = ntuple(dim -> dim == meta.axis ? oper.boundary : oper.restrict, Val(N))
    opers_identity = ntuple(dim -> dim == meta.axis ? oper.boundary : IdentityOperator{T, O}(), Val(N))

    thisedge = side ? 1 : size(func)[axis]
    otheredge = side ? size(other)[axis] : 1

    if point[axis] == thisedge
        thisvalue = product(point, opers_identity, func)

        otherpoint = ntuple(dim -> ifelse(dim == axis, size(other)[dim], point[dim]), Val(N))
        othervalue = product(otherpoint, opers_prolong, other)

        return oper.strength * (thisvalue - othervalue)
    end
end