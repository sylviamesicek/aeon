using Symbolics
using TensorOperations
using LinearAlgebra

function gradient(tensor::Array{T,N}, coords::Vector{T}) where {T,N}
    # result = zeros(size(tensor)..., 2)
    result = Array{T,N + 1}(undef, size(tensor)..., 2)

    for i in CartesianIndices(tensor)
        for j in 1:2
            # result[i..., j] = Differential(coords[j])(tensor[i]).doit()
            result[i, j] = Symbolics.derivative(tensor[i], coords[j])
        end
    end

    return result
end

function main()
    # Declare coordinate chart
    @variables ρ z

    coords = [ρ, z]

    # Declare metric
    @variables g₁₁(ρ, z), g₁₂(ρ, z), g₂₂(ρ, z)
    G = [g₁₁ g₁₂; g₁₂ g₂₂]
    Ginv = inv(g)
    Gdet = det(g)

    Ggrad = gradient(G, coords)

    display(Ggrad)

    # Compute Christoffel symbolcs
    Γ = zeros(2, 2, 2)

    @tensor begin
        Γ[λ, μ, ν] = ginv[λ, σ] * (Ggrad[ν, σ, μ] + Ggrad[σ, μ, ν] - Ggrad[μ, ν, σ])
    end


end

main()