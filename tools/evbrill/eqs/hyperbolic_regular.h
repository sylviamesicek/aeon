/*************************************
This code was generated automatically
using Sagemath and SymPy
**************************************/

// Subexpressions
double x0 = 2*shiftr_r;
double x1 = 2*lapse;
double x2 = krr*x1;
double x3 = 2*shiftz_z;
double x4 = kzz*x1;
double x5 = gzz*lapse;
double x6 = grr_z*x5;
double x7 = pow(gzz, 2);
double x8 = lapse*x7;
double x9 = 2*grz_r;
double x10 = gzz_z*lapse;
double x11 = krr*theta;
double x12 = grr*x7;
double x13 = grr_z*zz;
double x14 = grz_r*zz;
double x15 = (1.0/2.0)*grr;
double x16 = 1.0/grr;
double x17 = 1.0/x7;
double x18 = x16*x17;
double x19 = grr_z - grz_r;
double x20 = grr*x19;
double x21 = 2*krr;
double x22 = -grr_zz + grz_rz - gzz_rr + kzz*x21;
double x23 = grr*gzz*x1;
double x24 = pow(grr, 2);
double x25 = x1*zz;
double x26 = lapse*zz_z;
double x27 = gzz*x24;
double x28 = 1.0/x24;
double x29 = 1.0/gzz;
double x30 = x28*x29;
double x31 = x24*zz;
double x32 = kzz*lapse;
double x33 = grr*x5;
double x34 = pow(lapse, 2);

// Final Equations
double grr_t = grr*x0 + grr_z*shiftz - x2;

double gzz_t = gzz*x3 + gzz_z*shiftz - x4;

double grz_t = 0;

double krr_t = x18*((1.0/4.0)*grr*x10*(grr_z - x9) - grr_rr*x8 + (1.0/2.0)*grz_r*x6 + gzz*x15*(-grr_z*lapse_z - grr_zz*lapse + grz_rz*x1 - gzz_rr*lapse + kzz*x2 - 4*lapse*x14 + lapse_z*x9 + x1*x13) + x12*(krr*x0 + krr_z*shiftz - 6*lapse*s_r - lapse_rr - x1*x11 + x1*zr_r));

double kzz_t = (1.0/2.0)*x30*(pow(grr_z, 2)*x5 + x10*x20 + x22*x23 + x24*(gzz_z*lapse_z - gzz_z*x25 - pow(kzz, 2)*x1) + 2*x27*(kzz*x3 + kzz_z*shiftz - lapse_zz - theta*x4 + 2*x26));

double krz_t = 0;

double theta_t = x17*x28*(shiftz*theta_z*x24*x7 + x1*x12*(-3*s_r - x11 + zr_r) + x10*x15*x19 - 1.0/2.0*x10*x31 + x27*(-lapse_z*zz - theta*x32 + x26) + x33*(x13 - x14 + x22) + (1.0/4.0)*x6*(grr_z + x9) + x8*(-grr_rr + pow(krr, 2)));

double zr_t = 0;

double zz_t = x30*(krr*x6 + x20*x32 + x23*(-krr_z + krz_r) + x27*(lapse*theta_z - lapse_z*theta + shiftz*zz_z + shiftz_z*zz) - x31*x4);

double y_t = 0;

double s_t = 0;

double lapse_t = -kzz*x29*x34 + lapse_z*shiftz + 2*theta*x34 - x16*x21*x34;

double shiftr_t = 0;

double shiftz_t = x18*(gzz*x34*(-grr_z + grz_r) + gzz_z*x15*x34 + shiftz*shiftz_z*x12 + x33*(-lapse_z + x25));

