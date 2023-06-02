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
struct Gaussian{F} <: AnalyticFunction
    amplitude::F
    simga::F

    Gaussian(amplitude::F, sigma::F) where {F} = new{F}(amplitude, sigma)
end

(gauss::Gaussian)(x::AbstractVector) = gauss.amplitude * ℯ^(-dot(x, x)/(gauss.simga * gauss.simga))