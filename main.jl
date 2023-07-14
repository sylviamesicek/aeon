using Aeon
using Aeon.Geometry
using Aeon.Operators
using Aeon.Methods

using IterativeSolvers
using LinearMaps
using StaticArrays

function gunlach_seed(pos::SVector{2, T}, A::T, σ::T) where {T}
    ρ = pos[1]
    z = pos[2]

    A * (ρ/σ)^2 * ℯ^(-(ρ^2 + z^2) / σ^2)
end

function gunlach_laplacian(pos::SVector{2, T}, A::T, σ::T) where T
    ρ = pos[1]
    z = pos[2]

    2A*(2ρ^4 - 6ρ^2*σ^2 + σ^4 + 2ρ^2 * z^2) * ℯ^(-(ρ^2 + z^2) / σ^2) / σ^6
end

# Main code
function main()
    # Function basis
    basis = LagrangeBasis{Float64}()

    # Mesh

    mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SA{Float64}[4.0, 4.0]), 7)
    surface = TreeSurface(mesh)

    # Boundary conditions

    boundary = HyperFaces(BC(Nuemann, 0.0), BC(Nuemann, 0.0), BC(Flatness, -1.0), BC(Flatness, -1.0))

    # Seed function

    seed = TreeField(undef, surface, boundary)

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)

        for cell in cellindices(block)
            lpos = cellcenter(block, cell)
            gpos = trans(lpos)

            v = gunlach_laplacian(gpos, 1.0, 1.0)
            setvalue!(seed, v, block, cell)
        end
    end

    # Hemholtz Operator

    hemholtz = LinearMap(surface.total) do y, x
        yfield = TreeField(y, boundary)
        xfield = TreeField(x, boundary)

        domain = Domain{2, Float64, 2}(undef)

        for active in surface.active
            block = TreeBlock(surface, active)
            trans = blocktransform(block)

            if blockcells(block) ≠ domaincells(domain)
                domain = Domain{2}(undef, block)
            end

            transfer_block_to_domain!(domain, xfield, block, basis)

            for cell in cellindices(block)
                lpos = cellcenter(block, cell)
                # gpos = trans(lpos)
                j = inv(jacobian(trans, lpos))

                lhess = domainhessian(domain, cell, basis)
                ghess = j' * lhess * j

                gvalue = domainvalue(domain, cell)
                scale = value(seed, block, cell)

                setvalue!(yfield,  ghess[1, 1] + ghess[2, 2] + gvalue * scale, block, cell)
            end
        end
    end

    # Initial guess is flatness

    Ψ = TreeField(undef, surface, boundary)

    for active in surface.active
        block = TreeBlock(surface, active)
        for cell in cellindices(block)
            setvalue!(Ψ, 0.0, block, cell)
        end
    end

    println("Solving")

    _, history = bicgstabl!(Ψ.values, hemholtz, seed.values, 2; log=true, max_mv_products=4000)

    # testvalues = hemholtz * seed.values
    # Ψ = TreeField(testvalues, boundary)
    
    @show history

    writer = MeshWriter(surface)
    attrib!(writer, BlockAttribute())
    attrib!(writer, ScalarAttribute("seed", seed))
    attrib!(writer, ScalarAttribute("Ψ", Ψ))
    write_vtu(writer, "output")
end

# # Main code
# function run_profile()
#     mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SA{Float64}[π, π]), 3)

#     mark_global_refine!(mesh)
#     prepare_and_execute_refinement!(mesh)

#     surface = TreeSurface(mesh)

#     # Right hand side
#     analytic_laplacian = similar(surface)
#     analytic_value = similar(surface)

#     for active in surface.active
#         block = TreeBlock(surface, active)
#         trans = blocktransform(block)

#         for cell in cellindices(block)
#             lpos = cellcenter(block, cell)
#             gpos = trans(lpos)

#             ana = sin(gpos.x) * sin(gpos.y)
#             fun = -2sin(gpos.x)sin(gpos.y)

#             setfieldvalue!(cell, block, analytic_value, ana)
#             setfieldvalue!(cell, block, analytic_laplacian, fun)
#         end
#     end

#     # _precomp = laplacian * analytic_value.values

#     # @show length(_precomp)

#     value = LagrangeValue{Float64, 2}()
#     derivative = LagrangeDerivative{Float64, 2}()
#     derivative2 = LagrangeDerivative2{Float64, 2}()
#     hessian = HessianFunctional{2}(value, derivative, derivative2)
#     block = TreeBlock(surface, 2)

#     evaluate(CartesianIndex(4, 4), block, analytic_value)
#     evaluate(CartesianIndex(4, 4), block, (derivative, derivative), analytic_value)
#     allocated = @allocated evaluate(CartesianIndex(4, 4), block, (derivative, derivative), analytic_value)
#     println(allocated)
# end

# run_profile()

# Execute
main()