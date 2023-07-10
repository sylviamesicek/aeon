export GradientFunctional, HessianFunctional

struct GradientFunctional{N, T, VOper, DOper}
    value::VOper
    derivative::DOper

    function GradientFunctional{N}(value::Operator{T}, derivative::Operator{T}) where {N, T}
        new{N, T, typeof(value), typeof(derivative)}(value, derivative)
    end
end

function evaluate(cell::CartesianIndex{N}, block::Block{N, T}, fal::GradientFunctional{N, T}, field::Field{N, T}) where {N, T}
    cells = blockcells(block)

    SVector(
        ntuple(Val(N)) do i
            opers = ntuple(dim -> ifelse(i == dim, fal.deriv, fal.value), Val(N))
            return evaluate(cell, block, opers, field) * cells[i]
        end
    ) 
end

struct HessianFunctional{N, T, VOper, DOper, D2Oper}
    value::VOper
    derivative::DOper
    derivative2::D2Oper

    function HessianFunctional{N}(value::Operator{T}, derivative::Operator{T}, derivative2::Operator{T}) where {N, T}
        new{N, T, typeof(value), typeof(derivative), typeof(derivative2)}(value, derivative, derivative2)
    end
end

function evaluate(cell::CartesianIndex{N}, block::Block{N, T}, fal::HessianFunctional{N, T}, field::Field{N, T}) where {N, T}
    cells = blockcells(block)

    hess = ntuple(Val(N * N)) do index
        i = (index - 1) ÷ N + 1
        j = (index - 1) % N + 1

        if i == j
            opers = ntuple(dim -> ifelse(i == dim, fal.derivative2, fal.value), Val(N))
            return evaluate(cell, block, opers, field) * cells[i]^2
        else
            opers = ntuple(dim -> ifelse(i == dim || j == dim, fal.derivative, fal.value), Val(N))
            return evaluate(cell, block, opers, field) * cells[i] * cells[j]
        end
    end

    SMatrix{N, N, T}(hess)
end