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
double x5 = pow(gzz, 2);
double x6 = 1.0/x5;
double x7 = gzz*lapse;
double x8 = grr_z*x7;
double x9 = lapse*x5;
double x10 = 2*grz_r;
double x11 = gzz_z*lapse;
double x12 = krr*theta;
double x13 = grr*x5;
double x14 = grr_z*zz;
double x15 = grz_r*zz;
double x16 = grr*gzz;
double x17 = grr*(grr_z - grz_r);
double x18 = x11*x17;
double x19 = -grr_zz - gzz_rr + 2*krr*kzz;
double x20 = x1*x16;
double x21 = pow(grr, 2);
double x22 = gzz_z*zz;
double x23 = lapse*zz_z;
double x24 = gzz*x21;
double x25 = 1.0/x21;
double x26 = x25/gzz;
double x27 = kzz*lapse;

// Final Equations
double grr_t = grr*x0 + grr_z*shiftz - x2;

double gzz_t = gzz*x3 + gzz_z*shiftz - x4;

double grz_t = 0;

double krr_t = x6*((1.0/4.0)*grr*x11*(grr_z - x10) - grr_rr*x9 + (1.0/2.0)*grz_r*x8 + x13*(krr*x0 + krr_z*shiftz - 6*lapse*s_r - lapse_rr - x1*x12 + x1*zr_r) + (1.0/2.0)*x16*(-grr_z*lapse_z - grr_zz*lapse - gzz_rr*lapse + kzz*x2 - 4*lapse*x15 + lapse_z*x10 + x1*x14))/grr;

double kzz_t = (1.0/2.0)*x26*(pow(grr_z, 2)*x7 + x18 + x19*x20 + x21*(gzz_z*lapse_z - pow(kzz, 2)*x1 - x1*x22) + 2*x24*(kzz*x3 + kzz_z*shiftz - lapse_zz - theta*x4 + 2*x23));

double krz_t = 0;

double theta_t = x25*x6*(grr*x7*(x14 - x15 + x19) - 1.0/2.0*lapse*x21*x22 + shiftz*theta_z*x21*x5 + x1*x13*(-3*s_r - x12 + zr_r) + (1.0/2.0)*x18 + x24*(-lapse_z*zz - theta*x27 + x23) + (1.0/4.0)*x8*(grr_z + x10) + x9*(-grr_rr + pow(krr, 2)));

double zr_t = 0;

double zz_t = x26*(krr*x8 + x17*x27 + x20*(-krr_z + krz_r) - x21*x4*zz + x24*(lapse*theta_z - lapse_z*theta + shiftz*zz_z + shiftz_z*zz));

double y_t = 0;

double s_t = 0;

