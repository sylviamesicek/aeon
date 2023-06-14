using Aeon.Analytic
using Aeon.Approx
using Aeon.Method
using Aeon.Grid
using Aeon.Tensor

# using LinearAlgebra
using StaticArrays

# Main code
function main()
    method = GridMethod(SVector(0.0, 0.0), 1.0, SVector(8, 8), 1)

    domain = griddomain(method)

    basis = monomials(Val(2), Val(Float64), 2)
    weight = AGaussian{2, Float64}(1.0)

    engine = SquareEngine(domain, basis, weight)

    stencil = approx(engine, AIdentity{2, Float64}(), zero(SVector{2, Float64}))

    display(stencil)

    mesh = finest(method)

    field = similar(mesh)

    for i in eachindex(mesh)
        field[i] = mesh[i][1]
    end
    
    writer = MeshWriter(mesh)
    attrib!(writer, IndexAttribute())
    attrib!(writer, KindAttribute())
    attrib!(writer, ScalarAttribute("test", field))
    write_vtk(writer, "output")
end

function test1()
    method = GridMethod(SVector(0.0), 1.0, SVector(8), 2)

    domain = griddomain(method)

    basis = monomials(Val(1), Val(Float64), 4)
    weight = AGaussian{1, Float64}(1.0)

    engine = WLSEngine(domain, basis, weight)

    stencil = approx(engine, AIdentity{1, Float64}(), zero(SVector{1, Float64}))

    @show stencil
end

function test2()
    method = GridMethod(SVector(0.0, 0.0), 1.0, SVector(8, 8), 2)

    domain = griddomain(method)

    basis = monomials(Val(2), Val(Float64), 4)
    weight = AGaussian{2, Float64}(1.0)

    engine = SquareEngine(domain, basis)

    stencil = approx(engine, AIdentity{2, Float64}(), zero(SVector{2, Float64}))

    @show stencil
end

test2()