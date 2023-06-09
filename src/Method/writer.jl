# Exports
export MeshWriter, ScalarAttribute, IndexAttribute, KindAttribute, IntAttribute, attrib!, write_vtk

# Dependencies
using WriteVTK

# Code

struct NormalAttribute end
struct IndexAttribute end
struct KindAttribute end

struct ScalarAttribute{T}
    name::String
    values::Vector{T}
end

struct IntAttribute
    name::String
    values::Vector{Int}
end

mutable struct MeshWriter{N, T}
    mesh::Mesh{N, T}
    indices::Bool
    kinds::Bool
    scalars::Vector{ScalarAttribute{T}}
    ints::Vector{IntAttribute}
end

MeshWriter(mesh::Mesh{N, T}) where {N, T} = MeshWriter{N, T}(mesh, false, false, Vector{ScalarAttribute{T}}(), Vector{IntAttribute}())

function attrib!(writer::MeshWriter, attrib::ScalarAttribute)
    if length(attrib.values) != length(writer.mesh)
        error("Scalar Attribute length $(length(attrib.values)) does not match with length of Level $(length(writer.level))")
    end

    push!(writer.scalars, attrib)
end

function attrib!(writer::MeshWriter, attrib::IntAttribute)
    if length(attrib.values) != length(writer.mesh)
        error("Scalar Attribute length $(length(attrib.values)) does not match with length of Level $(length(writer.level))")
    end

    push!(writer.ints, attrib)
end

function attrib!(writer::MeshWriter, ::IndexAttribute)
    writer.indices = true
end

function attrib!(writer::MeshWriter, ::KindAttribute)
    writer.kinds = true
end

function write_vtk(writer::MeshWriter, filename)
    positions = position_array(writer.mesh)

    # Produce VTK Grid
    vtk_grid(filename, positions, Vector{MeshCell}()) do vtk
        if writer.indices
            indices = [i for i in eachindex(writer.mesh)]
            vtk["builtin:indices", VTKPointData()] = indices
        end

        if writer.kinds
            kinds = collect(map(Integer, writer.mesh.kinds))
            vtk["builtin:kinds", VTKPointData()] = kinds
        end
        
        # Scalars
        for i in eachindex(writer.scalars)
            scalar = writer.scalars[i]
            vtk[scalar.name, VTKPointData()] = scalar.values
        end

        # Ints
        for i in eachindex(writer.ints)
            scalar = writer.ints[i]
            vtk[scalar.name, VTKPointData()] = scalar.values
        end
    end
end

function position_array(mesh::Mesh{N, T}) where {N, T}
    matrix = Matrix{T}(undef, N, length(mesh))

    for j in eachindex(mesh)
        for i in 1:N
            matrix[i, j] = mesh.positions[j][i]
        end
    end

    matrix
end