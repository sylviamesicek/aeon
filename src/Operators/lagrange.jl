export LagrangeBasis

#####################
## Lagrange coefs ###
#####################

lagrange_value(grid::NTuple{N, T}, point::T) where {N, T} = ntuple(Val(N)) do i
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

cell_centered_grid(L, R) = ntuple(L + R + 1) do i
    i - L - 1//1
end

vertex_centered_grid(L, R) = ntuple(L + R) do i
    i - L - 1//2
end

########################
## Lagrange Basis ######
########################

struct LagrangeBasis{T} <: AbstractBasis{T} end

# Helpers
function lagrange_cell_stencil(f::Function, ::Val{T}, ::Val{L}, ::Val{R}, point) where {T, L, R}
    grid = cell_centered_grid(L, R)
    stencil = f(grid, point)
    left = map(T, ntuple(i -> stencil[L + 1 - i], Val(L)))
    right = map(T, ntuple(i -> stencil[L + 1 + i], Val(R)))
    center = T(stencil[L + 1])
    Stencil(left, center, right)
end

function lagrange_vertex_stencil(f::Function, ::Val{T}, ::Val{L}, ::Val{R}, ::Val{S}) where {T, L, R, S}
    grid = vertex_centered_grid(L, R)
    stencil = f(grid, 0//1)
    left = map(T, ntuple(i -> stencil[L + S - i], Val(L - !S)))
    right = map(T, ntuple(i -> stencil[L + S + i], Val(R - S)))
    Stencil(left, T(stencil[L + S]), right)
    
end

# Values 
function cell_value_stencil(::LagrangeBasis{T}, ::Val{L}, ::Val{R}) where {T, L, R} 
    lagrange_cell_stencil(lagrange_value, Val(T), Val(L), Val(R), 0//1)
end

function subcell_value_stencil(::LagrangeBasis{T}, ::Val{L}, ::Val{R}, ::Val{S}) where {S, T, L, R}
    lagrange_cell_stencil(lagrange_value, Val(T), Val(L), Val(R), ifelse(S, 1//2, -1//2))
end

function vertex_value_stencil(::LagrangeBasis{T}, ::Val{L}, ::Val{R}, ::Val{S}) where {T, L, R, S} 
    lagrange_vertex_stencil(lagrange_value, Val(T), Val(L), Val(R), Val(S))
end

# Derivative
function cell_derivative_stencil(::LagrangeBasis{T}, ::Val{L}, ::Val{R}) where {T, L, R} 
    lagrange_cell_stencil(lagrange_derivative, Val(T), Val(L), Val(R), 0//1)
end

function subcell_derivative_stencil(::LagrangeBasis{T}, ::Val{L}, ::Val{R}, ::Val{S}) where {S, T, L, R}
    lagrange_cell_stencil(lagrange_derivative, Val(T), Val(L), Val(R), ifelse(S, 1//2, -1//2))
end

function vertex_derivative_stencil(::LagrangeBasis{T}, ::Val{L}, ::Val{R}, ::Val{S}) where {T, L, R, S} 
    lagrange_vertex_stencil(lagrange_derivative, Val(T), Val(L), Val(R), Val(S))
end

# General cell centered stencils
function value_stencil(::LagrangeBasis{T}, ::Val{O}, ::Val{0}) where {T, O}
    lagrange_cell_stencil(lagrange_value, Val(T), Val(0), Val(0), 0//1)
end

function value_stencil(::LagrangeBasis{T}, ::Val{O}, ::Val{1}) where {T, O}
    lagrange_cell_stencil(lagrange_derivative, Val(T), Val(O), Val(O), 0//1)
end

function value_stencil(::LagrangeBasis{T}, ::Val{O}, ::Val{2}) where {T, O}
    lagrange_cell_stencil(lagrange_derivative_2, Val(T), Val(O), Val(O), 0//1)
end