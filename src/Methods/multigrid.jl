export MultiGridManager
export multigrid!

struct MultiGridLevel{T}
    b::Vector{T}
    x::Vector{T}
    i::Vector{T}
    diag::Vector{T}
end

MultiGridLevel{T}(total::Int) where T = MultiGridLevel{T}(Vector{T}(undef, total), Vector{T}(undef, total), Vector{T}(undef, total), Vector{T}(undef, total))

"""
Associates multigrid scratch data with each level of a mesh.
"""
struct MultiGridManager{N, T}
    levels::Vector{MultiGridLevel{T}}
    smoothing::Int
end

"""
Constructs a `MultiGridManager` from a DoFManager.
"""
function MultiGridManager(dofs::DoFManager{N, T}, smoothing::Int = 10) where {N, T}
    levels = MultiGridLevel[]

    for level in dofs.levels
        push!(levels, MultiGridLevel{T}(level.total))
    end

    MultiGridManager(levels, smoothing)
end

"""
Performs a multigrid solve using an `AbstractOperator`.
"""
function multigrid!(x::AbstractVector{T}, A::AbstractOperator{T}, b::AbstractVector{T}, manager::MultiGridManager{N, T}) where {N, T}
    scratch = last(manager.scratch)

    # Stage data into recursive chain
    copy!(scratch.b, b)
    copy!(scratch.x, x)

    # Recurse
    _multigrid_solve_rec!(A, manager, lastindex(manager.levels))

    # Copy data back into vectors
    copy!(b, scratch.b)
    copy!(x, scratch.x)
end

function _multigrid_solve_rec!(A::AbstractOperator{T}, manager::MultiGridManager{N, T}, level::Int) where {N, T}
    if level == 1
        # Solve directly 
        base = level > 0 ? manager.levels[begin] : manager.coarse[end]
        fill!(base.x, zero(T))
        # Build operator
        operator = LinearMap(length(refined.b), length(refined.x)) do y, x
            for l in 1:level
                operator_apply!(y, A, x, l)
            end
        end
        # Solve with BiCGStab
        bicgstabl!(refined.x, operator, refined.b, 2)
        # End recursion
        return
    end
    # There exists a coarser level

    # Level data
    refined = manager.levels[level]
    coarse = manager.levels[level - 1]

    # Clear solution data
    fill!(refined.x, zero(T))
    # End recursion if on coarsest level
    # Perform v-cycle
    # 1. Compute Diagonal
    operator_diagonal!(refined.diag, A, level)
    # 2. Presmoothing
    for _ in 1:manager.smoothing
        operator_apply!(refined.i, A, refined.x, level)
        manager.x .+= (refined.b .- refined.i) ./ refined.diag
    end
    # 3. Residual Computation
    operator_apply!(refined.i, A, refined.x, level)
    refined.i .= refined.b .- refined.i
    # 4. Restriction
    operator_restrict!(coarse.b, A, refined.i, level)
    # 5. Recursion
    _multigrid_solve_rec!(A, manager, level - 1)
    # 6. Prolongation
    operator_prolong!(refined.i, A, coarse.x, level)
    # 7. Correction
    refined.x .+= refined.i
    # 8. Post smoothing
    for _ in 1:manager.smoothing
        operator_apply!(refined.i, A, refined.x, level)
        manager.x .+= (refined.b .- refined.i) ./ refined.diag
    end
end