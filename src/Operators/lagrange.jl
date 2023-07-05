export lagrange

function lagrange(positions::AbstractVector{T}, point::T) where T
    coefficients = similar(positions)

    for j in eachindex(positions)
        result = one(T)

        for i in eachindex(positions)
            if i != j
                result *= (point - positions[i]) / (positions[j] - positions[i])
            end
        end

        coefficients[j] = result
    end

    coefficients
end

function lagrange_derivative(positions::AbstractVector{T}, point::T) where T
    coefficients = similar(positions)

    for j in eachindex(positions)
        r1 = zero(T)

        for l in eachindex(positions)
            if l != j
                r2 = one(T)

                for m in eachindex(positions)
                    if m != l && m != j
                        r2 *= (point - positions[m]) / (positions[j] - positions[m])
                    end
                end

                r1 += 1/(positions[j] - positions[l]) * r2
            end
        end

        coefficients[j] = r1
    end

    coefficients
end

function lagrange_derivative_2(positions::AbstractVector{T}, point::T) where T
    coefficients = similar(positions)

    for j in eachindex(positions)
        r1 = zero(T)

        for l in eachindex(positions)
            if l != j
                r2 = zero(T)

                for m in eachindex(positions)
                    if m != l && m != j
                        r3 = one(T)

                        for k in eachindex(positions)
                            if k != j && k != l && k != m
                                r3 *= (point - positions[k])/(positions[j] - positions[k])
                            end
                        end

                        r2 += r3 * 1/(positions[j] - positions[m])
                    end
                end

                r1 += r2 * 1/(positions[j] - positions[l])
            end
        end

        coefficients[j] = r1
    end

    coefficients
end