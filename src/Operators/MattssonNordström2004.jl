export MattssonNordström2004

"""
	MattssonNordström2004()

Coefficients of the SBP operators given in
- Mattsson, Nordström (2004)
	Summation by parts operators for finite difference approximations of second derivatives.
	Journal of Computational Physics 199, pp. 503-540.
"""
struct MattssonNordström2004{T, O} <: CoefficientSource{T, O} end

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

function mass_operator(::MattssonNordström2004{T, 1}) where {T}
	left = SVector{1, T}(1//2)
	right = left

	MassOperator{T, 1}(left, right)
end

function derivative_operator(::MattssonNordström2004{T, 1}, ::Val{1}) where {T}
	left = SA[SVector{2, T}(-1//1, 1//1)]
	right = reverse(map(reverse, left))
	central = SVector{3, T}(-1//2, 0, 1//2)

	CenteredOperator{T, 1}(left, right, central)
end

function derivative_operator(::MattssonNordström2004{T, 1}, ::Val{2}) where {T}
	left = SA[SVector{3, T}(1, -2, 1)]
	right = reverse(map(reverse, left))
	central = SVector{3, T}(1, -2, 1)

	CenteredOperator{T, 1}(left, right, central)
end

function boundary_derivative_operator(::MattssonNordström2004{T, 1}, ::Val{1}) where {T} 
	left = SVector{3, T}(3//2, -2//1, 1//2)
	right = reverse(left)

	BoundaryOperator{T, 1}(left, right)
end

function prologation_operator(::MattssonNordström2004{T, 1}) where {T}
	left = SVector{0, SVector{0, T}}()
	right = left
	central = SVector{2, T}(1//2, 1//2)

	ProlongationOperator{T, 1}(left, right, central)
end

function restriction_operator(::MattssonNordström2004{T, 1})
	left = SA[
		SVector{2, T}(1, 1)
	]
	right = reverse(map(reverse, left))
	central = SVector{3, T}(1//2, 1, 1//2)

	RestrictionOperator{T, 1}(left, right, central)
end

#######################
## Second Order #######
#######################

function mass_operator(::MattssonNordström2004{T, 2}) where {T}
	left = SVector{4, T}(17//48, 59//48, 43/48, 49/48)
	right = reverse(left)

	MassOperator{T, 2}(left, right)
end

function derivative_operator(::MattssonNordström2004{T, 2}, ::Val{1}) where {T}
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

function derivative_operator(::MattssonNordström2004{T, 2}, ::Val{2}) where {T}
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

function boundary_derivative_operator(::MattssonNordström2004{T, 2}, ::Val{1}) where {T} 
	left = SVector{4, T}(11//6, -3, 3//2, -1//3)
	right = map(reverse, left)

	BoundaryOperator{T, 2}(left, right)
end

function prologation_operator(::MattssonNordström2004{T, 2}) where {T}
	left = SA[
		SVector{7, T}(1, 0, 0, 0, 0, 0, 0),
		SVector{7, T}(429//944, 279//472, -43//944, 0, 0, 0, 0),
		SVector{7, T}(0, 1, 0, 0, 0, 0, 0),
		SVector{7, T}(-103//784, 549//784, 387//784, -1//16, 0, 0, 0),
		SVector{7, T}(-5//48, 5//24, 43//48, 0, 0, 0, 0),
		SVector{7, T}(-9//256, 5//256, 129//256, 147//256, -1//16, 0, 0),
		SVector{7, T}(1//24, -1//16, 0, 49//48, 0, 0, 0),
		SVector{7, T}(23//768, -37//768, -43//768, 147//256, 9//16, -1//16),
		SVector{7, T}(0, 0, 0, 0, 1, 0, 0),
		SVector{7, T}(-1//384, 1//256, 0, -49//768, 9//16, 9/16, -1//16),
	]
	right = reverse(map(reverse, left))
	central = SVector{2, T}(-1//16, 9//16, 9//16, -1//16)

	ProlongationOperator{T, 2}(left, right, central)
end

function restriction_operator(::MattssonNordström2004{T, 2})
	left = SA[
		SVector{10, T}(1//2, 429//544, 0, -103//544, -5//34, -27//544, 1//17, 23//544, 0, -1//272),
		SVector{10, T}(0, 279//944, 43//118, 549//1888, 5//59, 15//1888, -3//118, -37//1888, 0, 3//1888)
	]
	right = reverse(map(reverse, left))
	central = SVector{7, T}(-1//32, 0, 9//32, 1//2, 9//32, 0, -1//32)

	RestrictionOperator{T, 1}(left, right, central)
end