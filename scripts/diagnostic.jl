using CSV
using DataFrames
using LaTeXStrings
using Plots
using TOML

function timevsdofs(diagfile::String)::AbstractPlot
    # Load dataframe
    df = DataFrame(CSV.File(diagfile))
    # Load parameters and masses
    proper_time::Vector{Float64} = df[:, :proper_time]
    dofs::Vector{Float64} = df[:, :dofs]
    nodes::Vector{Float64} = df[:, :nodes]

    # Plot resulting data
    sc = plot(proper_time, dofs, label="Dofs (no ghost)")
    plot!(sc, proper_time, nodes, label="Nodes (with ghost)")

    plot!(sc, minorgrid=true)
    title!(sc, "Nodes vs Proper Time")
    xlabel!(sc, "Proper Time ($(L"\tau"))")
    ylabel!(sc, "Degrees of Freedom")
    return sc
end
