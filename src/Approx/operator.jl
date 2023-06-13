export ApproxOperator

struct ApproxOperator{N, T, R, D}
    values::Array{T, D}

    function ApproxOperator{N, T, R}(values::Array{T, D}) where {N, T, R, D}
        if D ≠ R + 1
            error("Incorrect dimensions for approximate operator.")
        end

        new{N, T, R, D}(values)
    end
end