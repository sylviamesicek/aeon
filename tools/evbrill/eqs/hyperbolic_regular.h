/*************************************
This code was generated automatically
using Sagemath and SymPy
**************************************/

// Subexpressions
double x0 = 2*lapse;
double x1 = krr*x0;
double x2 = 2*shiftz_z;
double x3 = kzz*x0;
double x4 = gzz*lapse;
double x5 = grr_z*x4;
double x6 = pow(gzz, 2);
double x7 = lapse*x6;
double x8 = 2*grz_r;
double x9 = gzz_z*lapse;
double x10 = krr*theta;
double x11 = grr*x6;
double x12 = grr_z*zz;
double x13 = grz_r*zz;
double x14 = (1.0/2.0)*grr;
double x15 = 1.0/grr;
double x16 = 1.0/x6;
double x17 = x15*x16;
double x18 = grr_z - grz_r;
double x19 = grr*x18;
double x20 = grr*gzz*x0;
double x21 = pow(grr, 2);
double x22 = x0*zz;
double x23 = gzz*x21;
double x24 = 1.0/x21;
double x25 = 1.0/gzz;
double x26 = x24*x25;
double x27 = x21*zz;
double x28 = kzz*lapse;
double x29 = 2*krr;
double x30 = grr*x4;
double x31 = pow(lapse, 2);

// Final Equations
double grr_t = 2*grr*shiftr_r + grr_z*shiftz - x1;

double gzz_t = gzz*x2 + gzz_z*shiftz - x3;

double grz_t = 0;

double krr_t = x17*((1.0/4.0)*grr*x9*(grr_z - x8) - grr_rr*x7 + (1.0/2.0)*grz_r*x5 + gzz*x14*(-grr_z*lapse_z - grr_zz*lapse + grz_rz*x0 - gzz_rr*lapse + kzz*x1 - 4*lapse*x13 + lapse_z*x8 + x0*x12) + x11*(2*krr*shiftr_r + krr_z*shiftz - 6*lapse*s_r + 2*lapse*zr_r - lapse_rr - x0*x10));

double kzz_t = (1.0/2.0)*x26*(pow(grr_z, 2)*x4 + x19*x9 + x20*(-grr_zz + grz_rz - gzz_rr + 2*krr*kzz) + x21*(gzz_z*lapse_z - gzz_z*x22 - pow(kzz, 2)*x0) + 2*x23*(kzz*x2 + kzz_z*shiftz + 2*lapse*zz_z - lapse_zz - theta*x3));

double krz_t = 0;

double theta_t = x16*x24*(shiftz*theta_z*x21*x6 + x0*x11*(-3*s_r - x10 + zr_r) + x14*x18*x9 + x23*(lapse*zz_z - lapse_z*zz - theta*x28) - 1.0/2.0*x27*x9 + x30*(-grr_zz + grz_rz - gzz_rr + kzz*x29 + x12 - x13) + (1.0/4.0)*x5*(grr_z + x8) + x7*(-grr_rr + pow(krr, 2)));

double zr_t = 0;

double zz_t = x26*(krr*x5 + x19*x28 + x20*(-krr_z + krz_r) + x23*(lapse*theta_z - lapse_z*theta + shiftz*zz_z + shiftz_z*zz) - x27*x3);

double y_t = 0;

double s_t = 0;

double lapse_t = -kzz*x25*x31 + lapse_z*shiftz + 2*theta*x31 - x15*x29*x31;

double shiftr_t = 0;

double shiftz_t = x17*(-gzz*x18*x31 + gzz_z*x14*x31 + shiftz*shiftz_z*x11 + x30*(-lapse_z + x22));

