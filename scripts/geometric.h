/*************************************
This code was generated automatically
using Sagemath and SymPy
**************************************/

// Subexpressions
double x0 = pow(grz, 2);
double x1 = 2*x0;
double x2 = grr*gzz;
double x3 = grr*gzz_z;
double x4 = grr_r*gzz;
double x5 = 2*x2;
double x6 = 2*grz_r;
double x7 = 4*grz_rz;
double x8 = 2*grz_z;
double x9 = grr_z*grz;
double x10 = (1.0/4.0)*(grr*pow(gzz_r, 2) + grr_r*grz*gzz_z + pow(grr_z, 2)*gzz + grr_z*x3 + grr_zz*x1 - grr_zz*x5 + 4*grz*grz_r*grz_z - grz*gzz_r*x6 + gzz_r*x4 - gzz_r*x9 + gzz_rr*x1 - gzz_rr*x5 - x0*x7 + x2*x7 - x3*x6 - x4*x8 - x8*x9)/(pow(grr, 2)*pow(gzz, 2) + pow(grz, 4) - x1*x2);

// Final Equations
double ricci_rr = grr*x10;

double ricci_rz = grz*x10;

double ricci_zz = gzz*x10;

