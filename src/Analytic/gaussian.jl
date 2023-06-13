###########################
## Exports ################
###########################

export Gaussian, GaussianGradient

###########################
## Gausian ################
###########################

"""
A Gaussian distribution function centered at the origin.
"""
struct Gaussian{N, T} <: AnalyticField{N, T, 1}
    simga::T
end

Gaussian{N}(sigma::T) where {N, T} = Gaussian{N, T}(sigma)

function (gauss::Gaussian{N, T})(x::SVector{N, T}) where {N, T}
    power = -dot(x, x)/(gauss.simga * gauss.simga)
    return ℯ^power
end

"""
The gradient of a gassian distribution.
"""
struct GaussianGradient{N, T} <: AnalyticField{N, T, 1}
    simga::T
end

(::AnalyticDerivative{N, T})(gauss::Gaussian{N, T}) where {N, T} = GaussianGradient{N, T}(gauss.simga)

function (gauss::GaussianGradient{N, T})(x::SVector{N, T}) where {N, T}
    power = -dot(x, x)/(gauss.simga * gauss.simga)
    scale = 1/(gauss.simga * gauss.simga) * ℯ^power
    return -scale * x
end