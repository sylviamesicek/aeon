export MattssonNordström2004

"""
	MattssonNordström2004()

Coefficients of the SBP operators given in
- Mattsson, Nordström (2004)
	Summation by parts operators for finite difference approximations of second derivatives.
	Journal of Computational Physics 199, pp. 503-540.
"""
struct MattssonNordström2004 <: CoefficientSource end

function Base.show(io::IO, ::MattssonNordström2004)
	print(io,
		 "Mattsson, Nordström (2004) \n",
			"  Summation by parts operators for finite difference approximations of second \n",
			"    derivatives. \n",
			"  Journal of Computational Physics 199, pp. 503-540.")
end

#############################
## First Order ##############
#############################

function mattson_derivative(::Val{T}, ::Val{1}, ::Val{1}) where {T}
	left_block = [SVector(-1//1, 1//1)]
	right_block = reverse(map(reverse, left_block))

	central_coefs = SVector{3, T}(-1//2, 0, 1//2)

	SBPDerivative(left_block, right_block, central_coefs)
end

function mattson_derivative(::Val{T}, ::Val{1}, ::Val{2}) where {T}
	left_block = [SVector(1, -2, 1)]
	right_block = reverse(map(reverse, left_block))

	central_coefs = SVector{3, T}(1, -2, 1)

	SBPDerivative(left_block, right_block, central_coefs)
end

function SBPOperator{T, 1}(source::MattssonNordström2004) where {T}
	left_weights = SVector{1, T}(1//2)
	central_weight = T(1//1)
	right_weights = SVector{1, T}(1//2)

	left_derivatives = (SVector{3, T}(3//2, -2//1, 1//2),)
	right_derivatives = map(reverse, left_derivatives)

	derivatives = (
		mattson_derivative(Val(T), Val(1), Val(1)),
		mattson_derivative(Val(T), Val(1), Val(2))
	)

	SBPOperator{T, 1}(left_weights, central_weight, right_weights, left_derivatives, right_derivatives, derivatives, source)
end

#######################
## Second Order #######
#######################

function mattson_derivative(::Val{T}, ::Val{2}, ::Val{1}) where {T}
	left_block =
		[
			SVector(-24//17, 59//34, -4//17, -3//34, 0, 0),
			SVector(-1//2, 0, 1//2, 0, 0, 0),
			SVector(4//43, -59//86, 0, 59//86, -4//43, 0),
			SVector(3//98, 0,-59//98, 0, 32//49, -4//49),
		]
	right_block = reverse(map(reverse, left_block))

	central_coefs = SVector{5, T}(1//12, -2//3, 0, 2//3, -1//12)

	SBPDerivative(left_block, right_block, central_coefs)
end

function mattson_derivative(::Val{T}, ::Val{2}, ::Val{2}) where {T}
	left_block = 
		[
			SVector(2, -5, 4, -1, 0, 0),
			SVector(1, -2, 1, 0, 0, 0),
			SVector(-4//43, 59//43, -110//43, 59//43, -4//43, 0),
			SVector(-1//49, 0, 59//49, -118//49, 64//49, -4//43)
		]
	right_block = reverse(map(reverse, left_block))

	central_coefs = SVector{5, T}(-1//12, 4//3, -5//2, 4//3, 1//12)

	SBPDerivative(left_block, right_block, central_coefs)
end

function SBPOperator{T, 2}(source::MattssonNordström2004) where {T}
	left_weights = SVector{4, T}(17//48, 59//48, 43/48, 49/48)
	central_weight = T(1)
	right_weights = reverse(right_weights)

	left_derivatives = (SVector{4, T}(11//6, -3, 3//2, -1//3),)
	right_derivatives = map(reverse, left_derivatives)

	derivatives = (
		mattson_derivative(Val(T), Val(2), Val(1)),
		mattson_derivative(Val(T), Val(2), Val(2))
	)

	SBPOperator{T, 1}(left_weights, central_weight, right_weights, left_derivatives, right_derivatives, derivatives, source)
end