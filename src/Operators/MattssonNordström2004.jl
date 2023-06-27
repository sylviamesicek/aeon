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

function mass(::Val{T}, ::Val{1}, ::MattssonNordström2004) where {T}
	left = SVector{1, T}(1//2)
	right = left

	MassOperator{T, 1}(left, right)
end

function derivative(::Val{T}, ::Val{1}, ::Val{1}, ::MattssonNordström2004) where {T}
	left = SA[SVector{2, T}(-1//1, 1//1)]
	right = reverse(map(reverse, left))
	central = SVector{3, T}(-1//2, 0, 1//2)

	CenteredOperator{T, 1}(left, right, central)
end

function derivative(::Val{T}, ::Val{1}, ::Val{2}, ::MattssonNordström2004) where {T}
	left = SA[SVector{3, T}(1, -2, 1)]
	right = reverse(map(reverse, left))
	central = SVector{3, T}(1, -2, 1)

	CenteredOperator{T, 1}(left, right, central)
end

function boundary_derivative(::Val{T}, ::Val{1}, ::Val{1}, ::MattssonNordström2004) where {T} 
	left = SVector{3, T}(3//2, -2//1, 1//2)
	right = reverse(left)

	CornerOperator{T, 1}(left, right)
end

#######################
## Second Order #######
#######################

function mass(::Val{T}, ::Val{2}, ::MattssonNordström2004) where {T}
	left = SVector{4, T}(17//48, 59//48, 43/48, 49/48)
	right = reverse(left)

	MassOperator{T, 2}(left, right)
end

function derivative(::Val{T}, ::Val{2}, ::Val{1}, ::MattssonNordström2004) where {T}
	left = SA[
			SVector{6, T}(-24//17, 59//34, -4//17, -3//34, 0, 0),
			SVector{6, T}(-1//2, 0, 1//2, 0, 0, 0),
			SVector{6, T}(4//43, -59//86, 0, 59//86, -4//43, 0),
			SVector{6, T}(3//98, 0,-59//98, 0, 32//49, -4//49),
		]
	right = reverse(map(reverse, left))
	central = SVector{5, T}(1//12, -2//3, 0, 2//3, -1//12)

	CenteredOperator{T, 2}(left, right, central)
end

function derivative(::Val{T}, ::Val{2}, ::Val{2}, ::MattssonNordström2004) where {T}
	left = SA[
			SVector{6, T}(2, -5, 4, -1, 0, 0),
			SVector{6, T}(1, -2, 1, 0, 0, 0),
			SVector{6, T}(-4//43, 59//43, -110//43, 59//43, -4//43, 0),
			SVector{6, T}(-1//49, 0, 59//49, -118//49, 64//49, -4//43)
		]
	right = reverse(map(reverse, left_block))
	central = SVector{5, T}(-1//12, 4//3, -5//2, 4//3, 1//12)

	CenteredOperator{T, 2}(left, right, central)
end

function boundary_derivative(::Val{T}, ::Val{1}, ::Val{1}, ::MattssonNordström2004) where {T} 
	left = SVector{4, T}(11//6, -3, 3//2, -1//3)
	right = map(reverse, left)

	CornerOperator{T, 2}(left, right)
end