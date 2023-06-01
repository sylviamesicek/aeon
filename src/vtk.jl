struct VtkFunction{F<:Real}
    name::String
    values::Vector{F}
end

struct VtkOutput{S, F<:Real}
    domain::Domain{S, F}
    functions::Vector{VtkFunction{F}}
end

VtkOutput(domain::Domain{S, F}) where {S, F<:Real} = VtkOutput{S, F}(domain, Vector{VtkFunction{F}}())

function attach_function!(vtk_output::VtkOutput{S, F}, name::String, values::Vector{F}) where {S, F<:Real}
    push!(vtk_output.functions, VtkFunction(name, values))
end

function write_vtk(vtk_output::VtkOutput{S, F}, filename) where {S, F<:Real}
    points = reduce(hcat, vtk_output.domain.positions)

    vtk = vtk_grid(filename, points, Vector{MeshCell}())

    for n in eachindex(vtk_output.functions)
        func = vtk_output.functions[n]
        vtk[func.name, VTKPointData()] = func.values
    end

    vtk_save(vtk)
end