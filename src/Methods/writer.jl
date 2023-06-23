# Exports
export MeshWriter, ScalarAttribute, IndexAttribute, CellAttribute, TagAttribute, attrib!, write_vtk

# Dependencies
using WriteVTK

# Code

"""
Adds `builtin::indices` data to VTK file.
"""
struct IndexAttribute end

"""
Adds `builtin::cells` data to VTK file.
"""
struct CellAttribute end

# """
# Adds `builtin::kinds` data to VTK file.
# """
# struct KindAttribute end

"""
Adds a scalar field of the given name to 
"""
struct ScalarAttribute{N, T}
    name::String
    field::Vector{T}
end

mutable struct MeshWriter{N, T}
    mesh::Mesh{N, T}
    indices::Bool
    cells::Bool
    scalars::Vector{ScalarAttribute{N, T}}
end

MeshWriter(mesh::Mesh{N, T}) where {N, T} = MeshWriter{N, T}(mesh, false, false, Vector{ScalarAttribute{N, T}}())

function attrib!(writer::MeshWriter, attrib::ScalarAttribute)
    if length(attrib.values) != length(writer.mesh)
        error("Scalar Attribute length $(length(attrib.values)) does not match with length of mesh $(length(writer.mesh))")
    end

    push!(writer.scalars, attrib)
end

function attrib!(writer::MeshWriter, ::IndexAttribute)
    writer.indices = true
end

function attrib!(writer::MeshWriter, ::CellAttribute)
    writer.cells = true
end

function write_vtk(writer::MeshWriter, filename)
    positions = position_array(writer.mesh)

    # Produce VTK Grid
    vtk_grid(filename, positions, Vector{MeshCell}()) do vtk
        if writer.indices
            vtk["builtin:indices", VTKPointData()] = indices_array(writer.mesh)
        end

        if writer.cells
            vtk["builtin:cells", VTKPointData()] = cells_array(writer.mesh)
        end
        
        # Scalars
        for i in eachindex(writer.scalars)
            vtk["scalar:$(scalar.name)", VTKPointData()] = writer.scalars[i].field
        end
    end
end

function position_array(mesh::Mesh{N, T}) where {N, T}
    matrix = Matrix{T}(undef, N, mesh.doftotal)

    for cell in eachindex(mesh)
        # @show mesh[cell], length(mesh[cell])
        for point in eachindex(mesh[cell])
            pos = position(mesh[cell], point)
            ptr = local_to_global(mesh[cell], point)


            for i in 1:N
                matrix[i, ptr] = pos[i]
            end
        end
    end

    matrix
end

function indices_array(mesh::Mesh{N, T}) where {N, T}
    vector = Vector{T}(undef, mesh.doftotal)

    for cell in eachindex(mesh)
        linear = LinearIndices(Tuple(mesh[cell].dofs))
        for point in eachindex(mesh[cell])
            ptr = local_to_global(mesh[cell], point)

            vector[ptr] = linear[point]
        end
    end

    vector
end

function cells_array(mesh::Mesh{N, T}) where {N, T}
    vector = Vector{T}(undef, mesh.doftotal)

    for cell in eachindex(mesh)
        for point in eachindex(mesh[cell])
            ptr = local_to_global(mesh[cell], point)
            vector[ptr] = cell
        end
    end

    vector
end