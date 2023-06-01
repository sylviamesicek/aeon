# Exports
export DomainWriter, ScalarAttribute, attrib!, write_vtk

# Dependencies
using WriteVTK

# Code

struct ScalarAttribute{F}
    name::String
    values::Vector{F}
end

struct DomainWriter{S, F}
    domain::Domain{S, F}
    scalars::Vector{ScalarAttribute{F}}
end

DomainWriter{S, F}(domain::Domain{S, F}) where {S, F} = DomainWriter{S, F}(domain, Vector{ScalarAttribute{F}}())

function attrib!(writer::DomainWriter, attrib::ScalarAttribute)
    push!(writer.scalars, attrib)
end

function write_vtk(writer::DomainWriter{S, F}, filename) where {S, F}
    count = length(writer.domain)

    # Build points matrix
    points = reduce(hcat, writer.domain.positions)

    # Build normal matrix
    normals = zeros(count, S)

    for i in eachindex(writer.domain.bounds)
        boundary = writer.domain.bounds[i]
        for j in 1:S
            normals[boundary.prev, j] = boundary.normal[j]
        end
    end

    # Produce VTK Grid
    vtk_grid(filename, points, Vector{MeshCell}()) do vtk
        vtk["normals", VTKPointData()] = normals'
        # Scalars
        for i in eachindex(writer.scalars)
            scalar = writer.scalars[i]
            vtk[scalar.name, VTKPointData()] = scalar.values
        end
    end
end