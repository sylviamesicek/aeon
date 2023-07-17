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

    boundary = HyperFaces(BC(Nuemann, 0.0), BC(Nuemann, 0.0), BC(Flatness, 0.0), BC(Flatness, 0.0))

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

            # subdomain = domain.inner[1:5, end-5:end]
            # display(subdomain)

            for cell in cellindices(block)
                lpos = cellcenter(block, cell)
                gpos = trans(lpos)
                j = inv(jacobian(trans, lpos))

                lgrad = domaingradient(domain, cell, basis)
                ggrad = j * lgrad

                lhess = domainhessian(domain, cell, basis)
                ghess = j' * lhess * j

                gvalue = domainvalue(domain, cell)
                scale = value(seed, block, cell)

                lap = ghess[1, 1] + ghess[2, 2] + ggrad[1]/gpos[1]

                setvalue!(yfield, -lap - gvalue * scale, block, cell)
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


# Main code
function main2()
    # Function basis
    basis = LagrangeBasis{Float64}()

    # Mesh

    mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SA{Float64}[2π, 2π]), 7)
    surface = TreeSurface(mesh)

    # Boundary conditions

    boundary = HyperFaces(BC(Nuemann, -1.0), BC(Nuemann, -1.0), BC(Nuemann, 1.0), BC(Nuemann, 1.0))

    # Seed function
    func = TreeField(undef, surface, boundary)
    analytic = TreeField(undef, surface, boundary)

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)

        for cell in cellindices(block)
            lpos = cellcenter(block, cell)
            gpos = trans(lpos)

            v = sin(gpos.x) + sin(gpos.y)
            setvalue!(func, v, block, cell)

            v = -sin(gpos.x) - sin(gpos.y)
            setvalue!(analytic, v, block, cell)

            # v = sin(gpos.x) * sin(gpos.y)
            # setvalue!(func, v, block, cell)

            # v = -2sin(gpos.x)*sin(gpos.y)
            # setvalue!(analytic, v, block, cell)
        end
    end

    # Hemholtz Operator

    laplacian = LinearMap(surface.total) do y, x
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

            # subdomain = domain.inner[1:5, 1:5]
            # display(subdomain)

            # sten1 = vertex_value_stencil(basis, Val(2), Val(2), Val(true))
            # sten2 = vertex_value_stencil(basis, Val(2), Val(2), Val(true))
            # value = domain_stencil_product(domain, CartesianIndex((1, 1)), (sten1, sten2))
            # @show value

            for cell in cellindices(block)
                lpos = cellcenter(block, cell)
                gpos = trans(lpos)
                j = inv(jacobian(trans, lpos))

                lhess = domainhessian(domain, cell, basis)
                ghess = j' * lhess * j

                # gvalue = domainvalue(domain, cell)
                # scale = value(seed, block, cell)

                setvalue!(yfield, ghess[1, 1] + ghess[2, 2], block, cell)

                # if cell[1] < 5 && cell[2] < 5
                #     @show cell
                #     lap = ghess[1, 1] + ghess[2, 2]
                #     @show lap
                #     ana = -2sin(gpos.x)*sin(gpos.y)
                #     @show ana
                # end
            end
        end
    end

    # Initial guess is flatness

    println("Application")

    # _, history = bicgstabl!(Ψ.values, hemholtz, seed.values, 2; log=true, max_mv_products=4000)

    testvalues = laplacian * func.values
    numeric = TreeField(testvalues, boundary)

    error = TreeField(numeric.values .- analytic.values, boundary)
    
    # @show history

    writer = MeshWriter(surface)
    attrib!(writer, BlockAttribute())
    attrib!(writer, ScalarAttribute("func", func))
    attrib!(writer, ScalarAttribute("analytic", analytic))
    attrib!(writer, ScalarAttribute("numeric", numeric))
    attrib!(writer, ScalarAttribute("error", error))
    write_vtu(writer, "output")
end


# Main code
function main3()
    # Function basis
    basis = LagrangeBasis{Float64}()

    # Mesh

    mesh = TreeMesh(HyperBox(SA[0.0, 0.0], SA{Float64}[2π, 2π]), 7)
    surface = TreeSurface(mesh)

    # Boundary conditions

    boundary = HyperFaces(BC(Diritchlet, 0.0), BC(Diritchlet, 0.0), BC(Diritchlet, 0.0), BC(Diritchlet, 0.0))

    # Seed function
    func = TreeField(undef, surface, boundary)

    for active in surface.active
        block = TreeBlock(surface, active)
        trans = blocktransform(block)

        for cell in cellindices(block)
            lpos = cellcenter(block, cell)
            gpos = trans(lpos)

            v = sin(gpos.x) + sin(gpos.y)
            setvalue!(func, v, block, cell)
        end
    end

    block = TreeBlock(surface, 1)
    domain = Domain{2}(undef, block)

    transfer_block_to_domain!(domain, func, block, basis)

    v = domainhessian(domain, CartesianIndex(7, 7), basis)
    @show v

    @time domainhessian(domain, CartesianIndex(7, 7), basis)
end


# Execute
main()