

abstract type Element{N, T} end
abstract type Field{N, T} end

fieldtype(element::Element) = error("`fieldtype` unimplemented for $(typeof(element))")
