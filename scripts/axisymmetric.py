import sympy

# Qualified imports
from sympy import Rational, symbols
from sympy.diffgeom import Manifold, Patch, CoordSystem, TensorProduct as TP, CovarDerivativeOp, metric_to_Christoffel_2nd

# Start pretty printing
sympy.init_printing()

r, z = symbols('r, z')

m = Manifold('M', 2)
p = Patch('P', m)

Cyl = CoordSystem('Cylindrical', p, symbols=[r, z])

fr, fz = Cyl.base_scalars()
e_r, e_z = Cyl.base_vectors()
dr, dz = Cyl.base_oneforms()

metric = TP(dr, dr) + Cyl.coord_function(r)**2 * TP(dz, dz)
ch = metric_to_Christoffel_2nd(metric)

cvd = CovarDerivativeOp(e_r, ch)

sympy.pretty_print(cvd(metric))