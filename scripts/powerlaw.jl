# A simple script for plotting the result of a massfill operation.

using CSV
using DataFrames
using LaTeXStrings
using Plots
using TOML

function massfill(historyfile::String, infofile::String)::AbstractPlot
    # Load dataframe
    df = DataFrame(CSV.File(historyfile))
    # Load infofile
    info = TOML.parsefile(infofile)
    # Get pstar
    pstar = Float64(info["pstar"])
    # Load parameters and masses
    param::Vector{Float64} = df[:, :param]
    mass::Vector{Float64} = df[:, :mass]
    # Plot resulting data
    sc = scatter(param .- pstar, mass, label="Final Mass", markershape=:cross, markersize=3)
    plot!(sc, xscale=:log10, yscale=:log10, minorgrid=true)
    title!(sc, "Mass Scaling Relation")
    xlabel!(sc, L"|p - p_*|")
    ylabel!(sc, "Mass [Natural Units]")
    xlims!(sc, 10^(-16), 10^(-4))
    ylims!(sc, 10^(-6), 10^(0))
    return sc
end

has_collapsed(status) = status == "Collapse"

function masssearch(historyfile::String)::AbstractPlot
    # As well as dataframe
    df = DataFrame(CSV.File(historyfile))
    # Filters data frame for data which collapsed
    df = filter(:status => has_collapsed, df)
    # Best approximation of pstar
    pstar = minimum(df[!, :param]) - 1e-16
    # Load parameters and masses
    param::Vector{Float64} = df[:, :param]
    mass::Vector{Float64} = df[:, :mass]
    # Plot resulting data
    sc = scatter(param .- pstar, mass, label="Final Mass", markershape=:cross, markersize=3)
    plot!(sc, xscale=:log10, yscale=:log10, minorgrid=true)
    title!(sc, "Mass Scaling Relation")
    xlabel!(sc, L"|p - p_*|")
    ylabel!(sc, "Mass [Natural Units]")
    xlims!(sc, 10^(-16), 10^(-1))
    ylims!(sc, 10^(-6), 1)
    return sc
end