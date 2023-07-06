export lagrange, lagrange_derivative, lagrange_derivative_2
export boundary_value_left, boundary_value_right
export boundary_derivative_left, boundary_derivative_right

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

boundary_grid_left(::Val{L}, ::Val{O}) where {L, O} = ntuple(Val(L)) do l
    ntuple(Val(O + l)) do i
        i - l - 1//2
    end
end

boundary_grid_right(::Val{L}, ::Val{O}) where {L, O} = ntuple(Val(L)) do l
    ntuple(Val(O + l)) do i
        i - O - 1//2
    end
end

boundary_value_left(::Val{T}, ::Val{L}, ::Val{O}) where {T, L, O} = map(boundary_grid_left(Val(L), Val(O))) do grid
    map(T, lagrange(grid, 0//1))
end

boundary_value_right(::Val{T}, ::Val{L}, ::Val{O}) where {T, L, O} = map(boundary_grid_right(Val(L), Val(O))) do grid
    map(T, lagrange(grid, 0//1))
end

boundary_derivative_left(::Val{T}, ::Val{L}, ::Val{O}) where {T, L, O} = map(boundary_grid_left(Val(L), Val(O))) do grid
    map(T, lagrange_derivative(grid, 0//1))
end

boundary_derivative_right(::Val{T}, ::Val{L}, ::Val{O}) where {T, L, O} = map(boundary_grid_right(Val(L), Val(O))) do grid
    map(T, lagrange_derivative(grid, 0//1))
end
