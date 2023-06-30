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

    ScalarAttribute{N}(name::String, field::Vector{T}) where {N, T} = new{N, T}(name, field)
end

mutable struct MeshWriter{N, T}
    mesh::TreeMesh{N, T}
    indices::Bool
    cells::Bool
    scalars::Vector{ScalarAttribute{N, T}}
end

MeshWriter(mesh::TreeMesh{N, T}) where {N, T} = MeshWriter{N, T}(mesh, false, false, Vector{ScalarAttribute{N, T}}())

function attrib!(writer::MeshWriter, attrib::ScalarAttribute)
    push!(writer.scalars, attrib)
end

function attrib!(writer::MeshWriter, ::IndexAttribute)
    writer.indices = true
end

function attrib!(writer::MeshWriter, ::CellAttribute)
    writer.cells = true
end

function write_vtk(writer::MeshWriter{N, T}, dofs::DoFManager{N, T}, filename::String) where {N, T}
    mesh = writer.mesh

    positions = Matrix{T}(undef, N, dofs.total)

    for active in dofs.active
        trans = nodetransform(mesh, active)
        offset = dofs.offsets[active]

        for (i, point) in enumerate(nodepoints(mesh, active))
            lpos = pointposition(mesh, active, point)
            gpos = trans(lpos)

            for dim in 1:N
                positions[dim, i + offset] = gpos[dim]
            end
        end
    end

    # Produce VTK Grid
    vtk_grid(filename, positions, Vector{MeshCell}()) do vtk
        if writer.indices
            indices = Vector{T}(undef, dofs.total)

            for active in dofs.active
                offset = dofs.offsets[active]

                for (i, _) in enumerate(nodepoints(mesh, active))
                    indices[offset + i] = i
                end
            end

            vtk["builtin:indices", VTKPointData()] = indices
        end

        if writer.cells
            cells = Vector{T}(undef, dofs.total)

            for active in dofs.active
                offset = dofs.offsets[active]

                for (i, _) in enumerate(nodepoints(mesh, active))
                    cells[offset + i] = active
                end
            end

            vtk["builtin:cells", VTKPointData()] = cells
        end
        
        # Scalars
        for i in eachindex(writer.scalars)
            @assert length(writer.scalars[i].field) == dofs.total

            vtk["scalar:$(writer.scalars[i].name)", VTKPointData()] = writer.scalars[i].field
        end
    end
end