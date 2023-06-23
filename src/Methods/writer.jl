# Exports
export MeshWriter, ScalarAttribute, IndexAttribute, KindAttribute, TagAttribute, attrib!, write_vtk

# Dependencies
using WriteVTK

# Code

"""
Adds `builtin::indices` data to VTK file.
"""
struct IndexAttribute end

# """
# Adds `builtin::kinds` data to VTK file.
# """
# struct KindAttribute end

"""
Adds a scalar field of the given name to 
"""
struct ScalarAttribute{N, T}
    name::String
    fields::Vector{Field{N, T}}
end

mutable struct MeshWriter{N, T}
    mesh::Mesh{N, T}
    indices::Bool
    scalars::Vector{ScalarAttribute{N, T}}
end

MeshWriter(mesh::Mesh{N, T}) where {N, T} = MeshWriter{N, T}(mesh, false, Vector{ScalarAttribute{N, T}}())

function attrib!(writer::MeshWriter, attrib::ScalarAttribute)
    if length(attrib.values) != length(writer.mesh)
        error("Scalar Attribute length $(length(attrib.values)) does not match with length of mesh $(length(writer.mesh))")
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
            vtk["builtin:indices", VTKPointData()] = indices_array(writer.mesh)
        end
        
        # Scalars
        for i in eachindex(writer.scalars)
            scalar = writer.scalars[i]
            vtk["scalar:$(scalar.name)", VTKPointData()] = scalar_array(writer.mesh, scalar)
        end
    end
end

function position_array(mesh::Mesh{N, T}) where {N, T}
    totaldofs = sum(map(length, mesh.cells))
    matrix = Matrix{T}(undef, N, totaldofs)

    offset = 1

    for cindex in eachindex(mesh)
        cell = mesh[cindex]
        for pindex in eachindex(cell)
            pos = position(cell, pindex)

            for i in 1:N
                matrix[i, offset] = pos[i]
            end

            offset += 1
        end
    end

    matrix
end

function indices_array(mesh::Mesh{N, T}) where {N, T}
    totaldofs = sum(map(length, mesh.cells))
    vector = Vector{T}(undef, totaldofs)

    offset = 1

    for cindex in eachindex(mesh)
        cell = mesh[cindex]

        linear = LinearIndices(Tuple(cell.dofs))

        for point in eachindex(cell)
            vector[offset] = linear[point]
            offset += 1
        end
    end

    vector
end

function scalar_array(mesh::Mesh{N, T}, attribute::ScalarAttribute{N, T}) where {N, T}
    totaldofs = sum(map(length, mesh.cells))
    vector = Vector{T}(undef, totaldofs)

    offset = 1

    for cindex in eachindex(mesh)
        cell = mesh[cindex]
        field = attribute.fields[cindex]
        for point in eachindex(cell)
            vector[offset] = value(cell, point, field)
            offset += 1
        end
    end

    vector
end