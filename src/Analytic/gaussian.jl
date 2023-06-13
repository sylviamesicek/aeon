###########################
## Exports ################
###########################

export AGaussian

###########################
## Gausian ################
###########################

"""
A Gaussian distribution function centered at the origin.
"""
struct AGaussian{N, T} <: AFunction{N, T}
    simga::T
end

AGaussian{N}(sigma::T) where {N, T} = AGaussian{N, T}(sigma)

function (gauss::AGaussian{N, T})(x::SVector{N, T}) where {N, T}
    power = -dot(x, x)/(gauss.simga * gauss.simga)
    return ℯ^power
end

function (::ADerivative{N, T})(gauss::AGaussian{N, T}, position::SVector{N, T}) where {N, T}
    power = -dot(position, position)/(gauss.simga * gauss.simga)
    scale = 1/(gauss.simga * gauss.simga) * ℯ^power
    return Covariant(-scale * position)
end