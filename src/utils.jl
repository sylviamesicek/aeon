export map_tuple_with_index

map_tuple_with_index_helper(f, t::Tuple, index) = (@inline; (f(index, t[1]), map_tuple_with_index_helper(f, Base.tail(t), index + 1)...))
map_tuple_with_index_helper(f, t::Tuple{Any,}, index) = (@inline; (f(index, t[1])))
map_tuple_with_index_helper(f, ::Tuple{}, index) = ()

map_tuple_with_index(f, t::Tuple) = map_tuple_with_index_helper(f, t, 1)