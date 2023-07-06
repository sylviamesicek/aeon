export lagrange, lagrange_derivative, lagrange_derivative_2

lagrange(grid::NTuple{N, T}, point::T) where {N, T} = ntuple(Val(N)) do i
    r1 = ntuple(Val(N)) do j
        if i != j
            return (point - grid[j]) / (grid[i] - grid[j])
        else
            return one(T)
        end
    end

    prod(r1)
end

lagrange_derivative(grid::NTuple{N, T}, point::T) where {N, T} = ntuple(Val(N)) do i
    r1 = ntuple(Val(N)) do j
        if i != j
            r2 = ntuple(Val(N)) do k
                if k != i && k != j
                    return (point - grid[k]) / (grid[i] - grid[k])
                else
                    return one(T)
                end
            end

            return  1/(grid[i] - grid[j]) * prod(r2)
        else
            return zero(T)
        end
    end

    sum(r1)
end

lagrange_derivative_2(grid::NTuple{N, T}, point::T) where {N, T} = ntuple(Val(N)) do i
    r1 = ntuple(Val(N)) do j
        if i != j
            r2 = ntuple(Val(N)) do k
                if k != i && k != j
                    r3 = ntuple(Val(N)) do l
                        if l != i && l != j && l != k
                            return (point - grid[l])/(grid[i] - grid[l])
                        else
                            return one(T)
                        end
                    end

                    return 1/(grid[i] - grid[k]) * prod(r3)
                else
                    return zero(T)
                end
            end

            return  1/(grid[i] - grid[j]) * sum(r2)
        else
            return zero(T)
        end
    end

    sum(r1)
end