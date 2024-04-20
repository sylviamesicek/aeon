/*************************************
This code was generated automatically
using Sagemath and SymPy
**************************************/

// Subexpressions
double x0 = pow(grr, -2);
double x1 = pow(gzz, 2);
double x2 = grr_z*gzz;
double x3 = grr*(grr_z - grz_r);
double x4 = grr*gzz;

// Final Equations
double hamiltonian = x0*(-24*grr*s_r*x1 + 2*gzz_z*x3 + 4*x1*(-grr_rr + krr**2) + x2*(grr_z + 2*grz_r) + 4*x4*(-grr_zz - gzz_rr + 2*krr*kzz))/(4*x1);

double momentum_r = 0;

double momentum_z = x0*(krr*x2 + kzz*x3 + 2*x4*(-krr_z + krz_r))/gzz;

