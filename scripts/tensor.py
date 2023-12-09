import numpy as np
import sympy as sym

r, z = sym.symbols('r z')

psi = sym.Function('psi')(r, z)
seed = sym.Function('s')(r, z)

A = sym.exp(2 * psi) * sym.exp(2 * r * seed)
lam = r * sym.exp(psi)

# A = sym.Function('A')(r, z)
# lam = sym.Function('lambda')(r, z)

lapse = sym.Function('alpha')(r, z)
shiftr = sym.Function('beta_r')(r, z)
shiftz = sym.Function('beta_z')(r, z)

h = np.array([[A, 0], [0, A]])
hinv = np.array([[1./A, 0], [0, 1./A]])

def par(f, i):
    if i == 0:
        return sym.diff(f, r)
    else:
        return sym.diff(f, z)

def connection(a, b, c):
    result = 0

    for d in range(0, 2):
        scale = hinv[a, d] / 2
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


# Extrinsic curvature

u = sym.Function('U')(r, z)
w = sym.Function('W')(r, z)
y = sym.Function('Y')(r, z)

ext = np.array([
    [(r * y - u) / 3, w], 
    [w, 2 * (u + r*y/2) / 3 ]])

exttrace = (2*r*y + u) / 3

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

y_normal = sym.simplify((2 * ext_normal(0, 0) + ext_normal(1, 1)) / r)
y_beta = shiftr * par(r*y, 0) / r + shiftz * par(y, 1) + w/r * (par(shiftz, 0) - par(shiftr, 1))
y_evolution = lapse * y_normal + y_beta

l_normal = - (covariant(lam, 0, 0) + covariant(lam, 1, 1) ) / lam - (par(lam, 0) * par(lapse, 0) +  par(lam, 1) * par(lapse, 1)) / (A * lam * lapse) - kappa / 2 * (rhoh - strace + tau)

y_normal_alt = sym.simplify((ext_normal(0, 0) - l_normal) / r)
y_beta_alt = y_beta
y_evolution_alt = sym.simplify(lapse * y_normal_alt) + y_beta

print("Covariant")
# sym.pretty_print(covariant(lam, 0, 1))
print("Evolution")
sym.pretty_print(y_evolution_alt)

