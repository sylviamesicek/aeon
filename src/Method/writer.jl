# Exports
export MeshWriter, ScalarAttribute, IndexAttribute, attrib!, write_vtk

# Dependencies
using WriteVTK

# Code

struct NormalAttribute end
struct IndexAttribute end

struct ScalarAttribute{T}
    name::String
    values::Vector{T}
end

mutable struct MeshWriter{N, T}
    mesh::Mesh{N, T}
    indices::Bool
    scalars::Vector{ScalarAttribute{T}}
end

MeshWriter(mesh::Mesh{N, T}) where {N, T} = MeshWriter{N, T}(mesh, false, Vector{ScalarAttribute{T}}())

function attrib!(writer::MeshWriter, attrib::ScalarAttribute)
    if length(attrib.values) != length(writer.level)
        error("Scalar Attribute length $(length(attrib.values)) does not match with length of Level $(length(writer.level))")
    end

    push!(writer.scalars, attrib)
end

function attrib!(writer::MeshWriter, ::IndexAttribute)
    writer.indices = true
end

function write_vtk(writer::MeshWriter, filename)
    positions = position_array(writer.mesh)

    # Produce VTK Grid
    vtk_grid(filename, positions, Vector{MeshCell}()) do vtk
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

function position_array(mesh::Mesh{N, T}) where {N, T}
    matrix = Matrix{T}(N, length(mesh))

    for j in eachindex(mesh)
        for i in 1:N
            matrix[i, j] = mesh.positions[j][i]
        end
    end
end