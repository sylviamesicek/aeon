# Exports
export GridWriter, ScalarAttribute, IndexAttribute, attrib!, write_vtk

# Dependencies
using WriteVTK

# Code

struct NormalAttribute end
struct IndexAttribute end

struct ScalarAttribute{T}
    name::String
    values::Vector{T}
end

mutable struct GridWriter{N, T}
    grid::GridLevel{N, T}
    indices::Bool
    scalars::Vector{ScalarAttribute{T}}
end

GridWriter(grid::GridLevel{N, T}) where {N, T} = GridWriter{N, T}(grid, false, Vector{ScalarAttribute{T}}())

function attrib!(writer::GridWriter, attrib::ScalarAttribute)
    push!(writer.scalars, attrib)
end

function attrib!(writer::GridWriter, ::IndexAttribute)
    writer.indices = true
end

function write_vtk(writer::GridWriter, filename)
    points = position_array(writer.grid)

    # Produce VTK Grid
    vtk_grid(filename, points, Vector{MeshCell}()) do vtk
        if writer.indices
            indices = [i for i in eachindex(writer.grid)]
            vtk["builtin:indices", VTKPointData()] = indices
        end
        
        # Scalars
        for i in eachindex(writer.scalars)
            scalar = writer.scalars[i]
            vtk[scalar.name, VTKPointData()] = scalar.values
        end
    end
end

function position_array(level::GridLevel{N, T}) where {N, T}
    array = Array{T, 2}(undef, N, length(level.positions))

    for i in eachindex(level)
        for j in 1:N
            array[j, i] = level[i][j]
        end
    end

    array
end