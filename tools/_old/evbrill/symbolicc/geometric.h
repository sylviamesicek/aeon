/*************************************
This code was generated automatically
using Sagemath and SymPy
**************************************/

// Subexpressions
double x0 = pow(gzz, 2);
double x1 = grr*gzz;
double x2 = pow(grz, 2);
double x3 = 2*x2;
double x4 = 1.0/(pow(grr, 2)*x0 + pow(grz, 4) - x1*x3);
double x5 = grr*gzz_z;
double x6 = grz*gzz_z;
double x7 = grr_r*gzz;
double x8 = 2*x1;
double x9 = 4*grz_rz;
double x10 = grr_z*grz;
double x11 = grz*gzz_r;
double x12 = grr_z*x11;
double x13 = grz*grz_r;
double x14 = 2*x13;
double x15 = (1.0/4.0)*x4*(-2*grr*grz_r*gzz_z + grr*pow(gzz_r, 2) - 2*grr_r*grz_z*gzz + grr_r*x6 + pow(grr_z, 2)*gzz + grr_z*x5 + grr_zz*x3 - grr_zz*x8 - 2*grz_z*x10 + 4*grz_z*x13 - gzz_r*x14 + gzz_r*x7 + gzz_rr*x3 - gzz_rr*x8 + x1*x9 - x12 - x2*x9);
double x16 = 1.0/(x1 - x2);
double x17 = grr_z*gzz;
double x18 = (1.0/2.0)*x16;
double x19 = x18*(-x11 + x17);
double x20 = grr_r*grz;
double x21 = grr*gzz_r;
double x22 = x18*(-x10 + x21);
double x23 = grz*grz_z;
double x24 = pow(grz, 3);
double x25 = gzz_r*x2;
double x26 = pow(grz_r, 2);

// Final Equations
double ricci_rr = grr*x15;

double ricci_rz = grz*x15;

double ricci_zz = gzz*x15;

double gamma_rrr = x16*((1.0/2.0)*x10 - x13 + (1.0/2.0)*x7);

double gamma_rrz = x19;

double gamma_rzr = x19;

double gamma_rzz = x16*(grz_z*gzz - 1.0/2.0*gzz*gzz_r - 1.0/2.0*x6);

double gamma_zrr = x16*(-1.0/2.0*grr*grr_z + grr*grz_r - 1.0/2.0*x20);

double gamma_zrz = x22;

double gamma_zzr = x22;

double gamma_zzz = x16*((1.0/2.0)*x11 - x23 + (1.0/2.0)*x5);

double gamma_rrr_r = (1.0/2.0)*x4*(grr*grr_rr*x0 + grr*grr_rz*grz*gzz + grr*grr_z*grz_r*gzz + 2*grr*grz*grz_r*gzz_r - grr*x12 - pow(grr_r, 2)*x0 + 4*grr_r*grz*grz_r*gzz - grr_r*x25 - grr_rr*gzz*x2 - grr_rz*x24 + grr_z*grz_r*x2 - 2*grz*grz_rr*x1 + 2*grz_rr*x24 - x17*x20 - x26*x3 - x26*x8);

double g_inv_rr_r = x4*(-grr_r*x0 + 2*grz*grz_r*gzz - x25);

double g_det_r = -x14 + x21 + x7;

double g_det_z = x17 - 2*x23 + x5;

