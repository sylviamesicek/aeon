###########################
## Exports ################
###########################

export Gaussian

###########################
## Gausian ################
###########################

"""
A Gaussian distribution function centered at the origin.
"""
struct Gaussian{N, T} <: AnalyticField{N, T, 0}
    simga::T

    Gaussian{N}(sigma::T) where {N, T} = new{N, T}(sigma)
end

function (gauss::Gaussian{N, T})(x::SVector{N, T}) where {N, T}
    power = -dot(x, x)/(gauss.simga * gauss.simga)
    return ℯ^power
end


struct GaussianGradient{N, T} <: AnalyticField{N, T, 1}
    simga::T

    GaussianGradient{N}(sigma::T) where {N, T} = new{N, T}(sigma)
end

(::GradientOperator)(gauss::Gaussian) = GaussianGradient(gauss.simga)

function (gauss::GaussianGradient{N, T})(x::SVector{N, T}) where {N, T}
    power = -dot(x, x)/(gauss.simga * gauss.simga)
    scale = 1/(gauss.simga * gauss.simga) * ℯ^power
    return -scale * x
end