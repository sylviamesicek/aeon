# Exports
export DomainWriter, ScalarAttribute, NormalAttribute, IndexAttribute, attrib!, write_vtk

# Dependencies
using WriteVTK

# Code

struct NormalAttribute end
struct IndexAttribute end

struct ScalarAttribute{F}
    name::String
    values::Vector{F}
end

mutable struct DomainWriter{D, F}
    const domain::Domain{D, F}
    normals::Bool
    indices::Bool
    scalars::Vector{ScalarAttribute{F}}
end

DomainWriter(domain::Domain{D, F}) where {D, F} = DomainWriter{D, F}(domain, false, false, Vector{ScalarAttribute{F}}())

function attrib!(writer::DomainWriter, attrib::ScalarAttribute)
    push!(writer.scalars, attrib)
end

function attrib!(writer::DomainWriter, ::NormalAttribute)
    writer.normals = true
end

function attrib!(writer::DomainWriter, ::IndexAttribute)
    writer.indices = true
end

function write_vtk(writer::DomainWriter, filename)
    points = position_array(writer.domain)

    # Produce VTK Grid
    vtk_grid(filename, points, Vector{MeshCell}()) do vtk
        if writer.normals
            normals = normal_array(writer.domain)
            vtk["normals", VTKPointData()] = normals
        end

        if writer.indices
            indices = [i for i in eachindex(writer.domain)]
            vtk["indices", VTKPointData()] = indices
        end
        
        # Scalars
        for i in eachindex(writer.scalars)
            scalar = writer.scalars[i]
            vtk[scalar.name, VTKPointData()] = scalar.values
        end
    end
end