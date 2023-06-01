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

function write_vtk(writer::DomainWriter, filename)
    points = reduce(hcat, vec(map(point -> point.position, writer.domain.points)))

    vtk_grid(filename, points, Vector{MeshCell}()) do vtk
        # Scalars
        for i in eachindex(writer.scalars)
            scalar = writer.scalars[i]
            vtk[scalar.name, VTKPointData()] = scalar.values
        end
    end
end