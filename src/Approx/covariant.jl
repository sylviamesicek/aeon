export ApproxCovariant

struct ApproxCovariant{N, T, O, D}
    values::Array{T, D}

    function ApproxCovariant{N, T, O}(values::Array{T, D}) where {N, T, O, D}
        if D ≠ O + 1
            error("Incorrect dimensions for approximate operator.")
        end

        new{N, T, O, D}(values)
    end
end