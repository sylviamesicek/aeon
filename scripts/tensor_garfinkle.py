import numpy as np
import sympy as sym

r, z = sym.symbols('r z')

psi = sym.Function('psi')(r, z)
seed = sym.Function('s')(r, z)

A = psi**4 * sym.exp(2 * r * seed)
lam = r * psi**2

# A = sym.Function('A')(r, z)
# lam = sym.Function('lambda')(r, z)

lapse = sym.Function('alpha')(r, z)
shiftr = sym.Function('beta_r')(r, z)
shiftz = sym.Function('beta_z')(r, z)

h = np.array([[A, 0], [0, A]])
hinv = np.array([[1/A, 0], [0, 1/A]])

def par(f, i):
    if i == 0:
        return sym.diff(f, r)
    else:
        return sym.diff(f, z)

def connection(a, b, c):
    result = 0

    for d in range(0, 2):
        scale = hinv[a, d] / sym.sympify(2)
        term = par(h[c, d], b) + par(h[b, d], c) - par(h[b, c], d)
        result += scale * term

    return sym.simplify(result)

# Computes R^a_{bcd}
def riemann(a, b, c, d):
    result = par(connection(a, b, d), c) - par(connection(a, b, c), d)
    
    for e in range(0, 2):
        result += connection(a, e, c) * connection(e, b, d)
        result -= connection(a, e, d) * connection(e, b, c)

    return sym.simplify(result)


# Computes R_{ab}
def comp_ricci_tmp(a, b):
    result = 0

    for c in range(0, 2):
        result += riemann(c, a, c, b)

    return sym.simplify(result)

# Computes R^a_b
def comp_ricci(a, b):
    result = 0

    for c in range(0, 2):
        result += hinv[a, c] * comp_ricci_tmp(c, b)

    return sym.simplify(result)

ricci = np.array([[comp_ricci(0, 0), comp_ricci(0, 1)], [comp_ricci(1, 0), comp_ricci(1, 1)]])
rtrace = sym.simplify(ricci[0, 0] + ricci[1, 1])

def covariant_tmp(f, a, b):
    result = par(par(f, b), a)
    
    for c in range(0, 2):
        result -= connection(c, b, a) * par(f, c)

    return sym.simplify(result)

def covariant(f, a, b):
    result = 0

    for c in range(0, 2):
        result += hinv[a, c] * covariant_tmp(f, c, b)

    return sym.simplify(result)

# print(covariant(lam, 0, 0))

# exit(0)

# sym.pretty_print(rtrace)

# exit(0)


# Extrinsic curvature

u = sym.Function('U')(r, z)
w = sym.Function('W')(r, z)
y = sym.Function('Y')(r, z)

ext = np.array([
    [(r * y - u) / sym.sympify(3), w], 
    [w, 2 * (u + r*y/2) / sym.sympify(3) ]])

exttrace = (2*r*y + u) / sym.sympify(3)

source = np.array([
    [sym.Function('Srr')(r, z), sym.Function('Srz')(r, z)], 
    [sym.Function('Szr')(r, z), sym.Function('Szz')(r, z)]])

strace = source[0, 0] + source[1, 1]

rhoh = sym.Function('rho_h')(r, z)
tau = sym.Function('tau')(r, z)
kappa = sym.Symbol('kappa')

def ext_normal(a, b):
    result = ricci[a, b] - covariant(lam, a, b) / lam - covariant(lapse, a, b) / lapse 
    result -= kappa * source[a, b]
    if a == b:
        result -= kappa * (rhoh - strace - tau) / 2
    return result

# Evolution

w_normal = sym.simplify(ext_normal(0, 1))
w_beta = shiftr * par(w, 0) + shiftz * par(w, 1) + (par(shiftz, 0) - par(shiftr, 1)) * u / 2
w_evolution =  lapse * w_normal + w_beta

u_normal = sym.simplify(ext_normal(1, 1) - ext_normal(0, 0))
u_beta = shiftr * par(u, 0) + shiftz * par(u, 1) + 2 * w * (par(shiftr, 1) - par(shiftz, 0))
u_evolution = lapse * u_normal + u_beta

l_normal = - (covariant(lam, 0, 0) + covariant(lam, 1, 1) ) / lam - (par(lam, 0) * par(lapse, 0) +  par(lam, 1) * par(lapse, 1)) / (A * lam * lapse) - kappa / 2 * (rhoh - strace + tau)

y_normal = sym.simplify((ext_normal(0, 0) - l_normal) / r)
y_beta = shiftr * par(r*y, 0) / r + shiftz * par(y, 1) + w/r * (par(shiftz, 0) - par(shiftr, 1))
y_evolution = sym.simplify(lapse * y_normal) + y_beta

ext_combination = (ext[0, 0])**2 + 2*(ext[0, 1])**2 + (ext[1, 1])**2

## Analytic

y_evolution_analytic1 = sym.exp(-2*r*seed) / (psi**4) * (-par(par(lapse, 0) / r, 0) - par(seed, 1) * par(lapse, 1) + (seed / r + par(seed, 0) + 4 *  (r * psi)**-1 * par(psi, 0)) * par(lapse, 0))
y_evolution_analytic2 = -lapse / (r**2 * psi**6) * sym.exp(-2*r*seed) * (r**2 * psi**2 * (par(par(seed, 0), 0) + par(par(seed, 1), 1) + par(seed, 0) / r) + 2 * r**2 * psi * par(seed, 1) * par(psi, 1) - seed * psi**2 - 6 * r * par(psi, 0)**2 + 2 * r * psi * par(par(psi, 0), 0) + par(psi, 0) * (-2 * r * r * psi * par(seed, 0) + (-2 * r * seed - 2) * psi))
y_evolution_analytic3 = shiftr * par(y, 0) + shiftz * par(y, 1) + y * shiftr / r + w / r * (par(shiftz, 0) - par(shiftr, 1))

y_evolution_analytic = y_evolution_analytic1 + y_evolution_analytic2 + y_evolution_analytic3

## Y EVOLUTION
print("Y EVOLUTION")
sym.pretty_print(y_evolution.expand() - y_evolution_analytic.expand())

## U EVOLUTION

u_evolution_analytic1 = psi**(-4) * sym.exp(-2 * r * seed) * ((4 * par(psi, 1) / psi + 2 * r * par(seed, 1)) * par(lapse, 1) - (2 * seed + 4 * par(psi, 0) / psi + 2 * r * par(seed, 0)) * par(lapse, 0) + par(par(lapse, 0), 0) - par(par(lapse, 1), 1))
u_evolution_analytic2 = -lapse * psi**(-6) * sym.exp(-2 * r * seed) / r * (-4 * r**2 * psi * par(seed, 1) * par(psi, 1) + 2 * r * psi**2 * par(seed, 0) + 2 * seed * psi**2 - 6 * r * par(psi, 1)**2 + 2 * r * psi * par(par(psi, 1), 1) + 6 * r * par(psi, 0)**2 - 2 * r * psi * par(par(psi, 0), 0) + (4 * r**2 * psi * par(seed, 0) + 4 * r * seed * psi) * par(psi, 0))
u_evolution_analytic3 = shiftr * par(u, 0) + shiftz * par(u, 1) + 2 * w * (par(shiftr, 1) - par(shiftz, 0))

u_evolution_analytic = u_evolution_analytic1 + u_evolution_analytic2 + u_evolution_analytic3

print("U EVOLUTION")
sym.pretty_print(u_evolution.expand() - u_evolution_analytic.expand())

print("W EVOLUTION")

w_evolution_analytic1 = psi**(-4) * sym.exp(-2 * r * seed) * ((seed + 2 * par(psi, 0) / psi + r * par(seed, 0)) * par(lapse, 1) - par(par(lapse, 0), 1) + (2 * par(psi, 1) / psi + r * par(seed, 1)) * par(lapse, 0))
w_evolution_analytic2 = lapse * psi**(-6) * sym.exp(-2 * r * seed) * (2 * r * psi * par(seed, 1) * par(psi, 0) + psi**2 * par(seed, 1) + (2  *r * psi * par(seed, 0) + 2 * seed * psi + 6 * par(psi, 0)) * par(psi, 1) - 2 * psi * par(par(psi, 0), 1))
w_evolution_analytic3 = shiftr * par(w, 0) + shiftz * par(w, 1) + u / 2 * (par(shiftz, 0) - par(shiftr, 1))

w_evolution_analytic = w_evolution_analytic1 + w_evolution_analytic2 + w_evolution_analytic3

sym.pretty_print(w_evolution.expand() - w_evolution_analytic.expand())

# assert((y_evolution_analytic.expand() - y_evolution.expand()) == 0)

# u_evolution_analytic1 = shiftr * par(u, 0) + shiftz * par(u, 1) + 2 * w * (par(shiftr, 1) - par(shiftz, 0)) + kappa * lapse * (source[0, 0] - source[1, 1])
# u_evolution_analytic2 = sym.exp(-2*r*seed - 2*psi) * (2 * (r * par(seed, 1) + par(psi, 1)) * par(lapse, 1) - 2*(r * par(seed, 0) + seed + par(psi, 0)) * par(lapse, 0) + par(par(lapse, 0), 0) - par(par(lapse, 1), 1))
# u_evolution_analytic3 = lapse * sym.exp(-2*r*seed - 2*psi) * ((2 * r * par(seed, 1) + par(psi, 1)) * par(psi, 1) - (2 * r * par(seed, 0) + 2 * seed + par(psi, 0)) * par(psi, 0) + par(par(psi, 0), 0) - par(par(psi, 1), 1) - 2 * par(r * seed, 0) / r)

# u_evolution_analytic = u_evolution_analytic1 + u_evolution_analytic2 + u_evolution_analytic3

# assert((u_evolution.expand() - u_evolution_analytic.expand()) == 0)

# w_evolution_analytic1 = shiftr * par(w, 0) + shiftz * par(w, 1) + u / 2 * (par(shiftz, 0) - par(shiftr, 1)) - kappa * lapse * source[0, 1]
# w_evolution_analytic2 = sym.exp(-2*r*seed - 2*psi) * (par(lapse, 0) * (r * par(seed, 1) + par(psi, 1)) + par(lapse, 1) * (r * par(seed, 0) + seed + par(psi, 0)) - par(par(lapse, 0), 1))
# w_evolution_analytic3 = lapse * sym.exp(-2*r*seed - 2*psi) * (par(psi, 0) * (r * par(seed, 1) + par(psi, 1)) + par(psi, 1) * (r * par(seed, 0) + seed) + par(seed, 1) - par(par(psi, 0), 1))

# w_evolution_analytic = w_evolution_analytic1 + w_evolution_analytic2 + w_evolution_analytic3

# assert((w_evolution.expand() - w_evolution_analytic.expand()) == 0)

print("Finished")
# sym.pretty_print(diff)

# print("Analytic")
# sym.pretty_print(sym.simplify(y_evolution_analytic))
# print("Program")
# sym.pretty_print(sym.simplify(y_evolution))

# lapse_gauge = covariant(lapse, 0, 0) + covariant(lapse, 1, 1) + (par(lapse, 0) * par(lam, 0) + par(lapse, 1) * par(lam, 1)) / (A * lam) - lapse * (ext_combination + exttrace**2 + kappa * (rhoh + tau + strace) / 2)

# hamiltonian_constraint = (rtrace - ext_combination - exttrace**2) / 2 - (covariant(lam, 0, 0) + covariant(lam, 1, 1)) / lam - kappa * rhoh

# sym.pretty_print(sym.simplify(u_evolution))
# print("EXT")
# sym.pretty_print(sym.simplify(-(ext_combination + exttrace**2) / 2))
# print("Lam Covariant / lam")
# sym.pretty_print(sym.simplify(-(covariant(lam, 0, 0) + covariant(lam, 1, 1)) / lam))
# print("Source")
# sym.pretty_print(sym.simplify(-kappa * rhoh))




# sym.pretty_print(sym.simplify(covariant(lam, 0, 0)))


# print("Covariant")
# # sym.pretty_print(covariant(lam, 0, 1))
# print("Evolution")
# sym.pretty_print(sym.simplify(hamiltonian_constraint))

