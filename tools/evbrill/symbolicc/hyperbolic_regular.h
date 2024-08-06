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
double x12 = 2*grz_rz;
double x13 = grr_z*zz;
double x14 = grz_r*zz;
double x15 = (1.0/2.0)*grr;
double x16 = 1.0/grr;
double x17 = 1.0/x6;
double x18 = x16*x17;
double x19 = grr_z - grz_r;
double x20 = grr*x19;
double x21 = pow(grr, 2);
double x22 = x0*zz;
double x23 = grr*gzz*x0;
double x24 = gzz*x21;
double x25 = 1.0/x21;
double x26 = 1.0/gzz;
double x27 = x25*x26;
double x28 = x21*zz;
double x29 = kzz*lapse;
double x30 = 2*krr;
double x31 = grr*x4;
double x32 = pow(lapse, 2);

// Final Equations
double grr_t = 2*grr*shiftr_r + grr_z*shiftz - x1;

double gzz_t = gzz*x2 + gzz_z*shiftz - x3;

double grz_t = 0;

double krr_t = x18*((1.0/4.0)*grr*x9*(grr_z - x8) - grr_rr*x7 + (1.0/2.0)*grz_r*x5 + gzz*x15*(-grr_z*lapse_z - grr_zz*lapse - gzz_rr*lapse + kzz*x1 + lapse*x12 - 4*lapse*x14 + lapse_z*x8 + x0*x13) + x11*(2*krr*shiftr_r + krr_z*shiftz - 6*lapse*s_r + 2*lapse*zr_r - lapse_rr - x0*x10));

double kzz_t = (1.0/2.0)*x27*(pow(grr_z, 2)*x4 + x20*x9 + x21*(gzz_z*lapse_z - gzz_z*x22 - pow(kzz, 2)*x0) + x23*(-grr_zz + 2*grz_rz - gzz_rr + 2*krr*kzz) + 2*x24*(kzz*x2 + kzz_z*shiftz + 2*lapse*zz_z - lapse_zz - theta*x3));

double krz_t = 0;

double theta_t = x17*x25*(shiftz*theta_z*x21*x6 + x0*x11*(-3*s_r - x10 + zr_r) + x15*x19*x9 + x24*(lapse*zz_z - lapse_z*zz - theta*x29) - 1.0/2.0*x28*x9 + x31*(-grr_zz - gzz_rr + kzz*x30 + x12 + x13 - x14) + (1.0/4.0)*x5*(grr_z + x8) + x7*(-grr_rr + pow(krr, 2)));

double zr_t = 0;

double zz_t = x27*(krr*x5 + x20*x29 + x23*(-krr_z + krz_r) + x24*(lapse*theta_z - lapse_z*theta + shiftz*zz_z + shiftz_z*zz) - x28*x3);

double y_t = 0;

double s_t = 0;

double lapse_t = -kzz*x26*x32 + lapse_z*shiftz + 2*theta*x32 - x16*x30*x32;

double shiftr_t = 0;

double shiftz_t = x18*(-gzz*x19*x32 + gzz_z*x15*x32 + shiftz*shiftz_z*x11 + x31*(-lapse_z + x22));

