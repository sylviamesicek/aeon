# A script for computing stencils for various operations using rational arithmatic (to minimize
# floating point round-off).

using LinearAlgebra

function vandermonde(grid::Vector{Rational{Int128}})::Matrix{Rational{Int128}}
    N = length(grid)
    return [
        grid[j]^i for i in 0:(N-1), j in 1:N
    ]
end

function operator(M::Int, order::Int)
    result = zeros(Rational{Int64}, M)
    result[order+1] = factorial(order)
    return result
end

vertexgrid(L::Int, R::Int)::Vector{Rational{Int128}} = [Rational(i) for i in -L:R]
cellgrid(L::Int, R::Int)::Vector{Rational{Int128}} = vcat([(2i + 1) // 2 for i in -L:-1], [(2i - 1) // 2 for i in 1:R])


function operators(N::Int)
    println("Stencils for $(2N) order.")
    grid = vertexgrid(N, N)

    println("Interior Grid $(grid)")

    mat = inv(vandermonde(grid))

    println(mat)


    value = mat * operator(length(grid), 0)
    derivative = mat * operator(length(grid), 1)
    second_derivative = mat * operator(length(grid), 2)

    println("Value Stencil: $(value)")
    println("Derivative Stencil: $(derivative)")
    println("Second Derivative Stencil: $(second_derivative)")
end

function interpolation(N::Int)
    println("Stencils for $(2N - 1) order.")

    grid = cellgrid(N, N)

    mat = inv(vandermonde(grid))
    stencil = mat * operator(length(grid), 0)

    println("Interior Stencil: $(stencil)")

    for left in 1:(N-1)
        grid = cellgrid(left, 2N - left)
        mat = inv(vandermonde(grid))
        stencil = mat * operator(length(grid), 0)
        println("Negative Stencil Left=$(left): $(stencil)")
    end

    for right in reverse(1:(N-1))
        grid = cellgrid(2N - right, right)
        mat = inv(vandermonde(grid))
        stencil = mat * operator(length(grid), 0)
        println("Positive Stencil Right=$(right): $(stencil)")
    end

end

# function main()




# end