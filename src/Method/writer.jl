# Exports
export MeshWriter, ScalarAttribute, IndexAttribute, KindAttribute, IntegerAttribute, TagAttribute, attrib!, write_vtk

# Dependencies
using WriteVTK

# Code

"""
Adds `builtin::indices` data to VTK file.
"""
struct IndexAttribute end

"""
Adds `builtin::kinds` data to VTK file.
"""
struct KindAttribute end

"""
Adds `builtin::tags` data to VTK file.
"""
struct TagAttribute end

"""
Adds a scalar field of the given name to 
"""
struct ScalarAttribute{T}
    name::String
    values::Vector{T}
end

struct IntegerAttribute
    name::String
    values::Vector{Int}
end

mutable struct MeshWriter{N, T}
    mesh::Mesh{N, T}
    indices::Bool
    kinds::Bool
    tags::Bool
    scalars::Vector{ScalarAttribute{T}}
    integers::Vector{IntegerAttribute}
end

MeshWriter(mesh::Mesh{N, T}) where {N, T} = MeshWriter{N, T}(mesh, false, false, false, Vector{ScalarAttribute{T}}(), Vector{IntegerAttribute}())

function attrib!(writer::MeshWriter, attrib::ScalarAttribute)
    if length(attrib.values) != length(writer.mesh)
        error("Scalar Attribute length $(length(attrib.values)) does not match with length of mesh $(length(writer.mesh))")
    end

    push!(writer.scalars, attrib)
end

function attrib!(writer::MeshWriter, attrib::IntegerAttribute)
    if length(attrib.values) != length(writer.mesh)
        error("Integer Attribute length $(length(attrib.values)) does not match with length of mesh $(length(writer.mesh))")
    end

    push!(writer.integers, attrib)
end

function attrib!(writer::MeshWriter, ::IndexAttribute)
    writer.indices = true
end

function attrib!(writer::MeshWriter, ::KindAttribute)
    writer.kinds = true
end

function attrib!(writer::MeshWriter, ::TagAttribute)
    writer.tags = true
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
            vtk["builtin:kinds", VTKPointData()] = writer.mesh.kinds
        end

        if writer.tags
            vtk["builtin:tags", VTKPointData()] = writer.mesh.tags
        end
        
        # Scalars
        for i in eachindex(writer.scalars)
            scalar = writer.scalars[i]
            vtk["scalar:$(scalar.name)", VTKPointData()] = scalar.values
        end

        # Integers
        for i in eachindex(writer.integers)
            integers = writer.integers[i]
            vtk["integer:$(integers.name)", VTKPointData()] = integers.values
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