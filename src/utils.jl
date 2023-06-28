export tuple_project_on_axis, tuple_splice_on_axis
export index_project_on_axis, index_splice_on_axis

tuple_project_on_axis(t::Tuple, ::Val{A}) where A = tuple(t[begin:(A-1)]..., t[(A + 1):end]...)
tuple_splice_on_axis(t::Tuple, ::Val{A}, value) where A = tuple(t[begin:(A-1)]..., value, t[(A + 1):end]...)

index_project_on_axis(i::CartesianIndex, ::Val{A}) where A = CartesianIndex(tuple_project_on_axis(i.I, Val(A)))
index_splice_on_axis(i::CartesianIndex, ::Val{A}, value::Int) where A = CartesianIndex(tuple_splice_on_axis(i.I, Val(A), value))