/*************************************
This code was generated automatically
using Sagemath and SymPy
**************************************/

// Subexpressions
double x0 = pow(grr, 3);
double x1 = rho*x0;
double x2 = pow(grz, 3);
double x3 = krr*x2;
double x4 = grr*krz;
double x5 = 2*x4;
double x6 = pow(krr, 2);
double x7 = pow(grr, 2);
double x8 = pow(gzz, 2);
double x9 = x7*x8;
double x10 = pow(grz, 2);
double x11 = grr*gzz;
double x12 = grr*rho;
double x13 = -x10;
double x14 = x11 + x13;
double x15 = rho*x7;
double x16 = x14*x15;
double x17 = rho*x11;
double x18 = 2*krr;
double x19 = x7*(pow(krz, 2) - kzz*x18);
double x20 = gzz*x7;
double x21 = 2*x20;
double x22 = 2*x11;
double x23 = -x11;
double x24 = x10 + x23;
double x25 = rho*x20;
double x26 = pow(rho, 2);
double x27 = x14*x26;
double x28 = pow(rho, 3)*x24;
double x29 = grz_r*rho;
double x30 = grr_z*rho;
double x31 = rho*s;
double x32 = grz*x31;
double x33 = gzz_z*x0;
double x34 = 2*grz;
double x35 = grz*x29;
double x36 = grz*x30;
double x37 = x11*x31;
double x38 = grz_z*x7;
double x39 = grr_r*rho;
double x40 = grz*x39;
double x41 = grr*x30;
double x42 = grr*x32;
double x43 = (1.0/2.0)*x11;
double x44 = 2*x10;
double x45 = x11 - x44;
double x46 = x31*x45;
double x47 = x23 + x44;
double x48 = grz*krr;
double x49 = grr*x47;
double x50 = -grr*x44 + x20;
double x51 = (1.0/2.0)*grz;
double x52 = x10*x39;
double x53 = (1.0/2.0)*grr;
double x54 = krz*x2;
double x55 = grr*kzz;
double x56 = krr*x10;
double x57 = kzz*x7;
double x58 = x56 - x57;
double x59 = x26*x7;
double x60 = grz*x15;
double x61 = rho*x53;
double x62 = grz*x20;
double x63 = rho*x33;
double x64 = rho*x38;
double x65 = 3*x11;
double x66 = 4*x10;
double x67 = pow(grz, 4);
double x68 = -x11*x44 + x67 + x9;
double x69 = 1/(rho*x68*x7);
double x70 = krr*x67;
double x71 = grz*krz;
double x72 = -gzz*krr + x71;
double x73 = x4 - x48;
double x74 = x1*x14;
double x75 = x55 - x71;
double x76 = grz*x16;
double x77 = x12*x68;
double x78 = x10*x4;
double x79 = x59*x68;
double x80 = grz*x4;
double x81 = x57 - x80;
double x82 = (1.0/2.0)*gzz_r*x15;
double x83 = x11*x73;
double x84 = -x3 + x78 - x83;
double x85 = 2*x9;
double x86 = -x70;
double x87 = x2*x4;
double x88 = x77*y;
double x89 = x87 - x88;
double x90 = grr*x26;
double x91 = x18*x9;
double x92 = x10*x57 + x70;
double x93 = grz*(x55 + x71);
double x94 = -gzz*x93 + kzz*x2 + x4*x8;
double x95 = -x87 + x88;

// Final Equations
double hamiltonian = x69*(grr_r**2*gzz*rho*(x13 + x22)/4 + grr_r*x43*(-x46 + x47) - grr_rr*x14*x17/2 + grr_rz*grz*x12*x14 + grr_z**2*x12*(x10 + x11)/4 + grr_z*x51*(-x31*x49 + x39*(x10 - x22) + x50) - grr_zz*x16 - grz_r*x43*(grr*x34 + x40 - x41 + 2*x42) + grz_rz*x16 + gzz_r**2*x1/4 - gzz_r*x53*(grr*x35 + grr*x36 + grr*x46 + x50 - x52) - gzz_rr*x16/2 + rho*x10*x19 + rho*x3*x5 + rho*x6*x9 + s**2*x24*x25 - s*x14*x21 + s_r**2*x20*x28 + s_r*x61*(grz_z*rho*x21 + gzz*x39*x47 + gzz_r*rho*x49 - gzz_z*x60 + 8*x10*x11 - x22*x35 + 4*x24*x37 - x36*x47 - 8*x9) - s_rr*x20*x27 + s_rz*x27*x34*x7 + s_z**2*x0*x28 - s_z*x61*(-4*grr*grz*s_r*x27 + 8*grr*x2 + gzz_r*x60 - 4*x14*x42 - x21*x29 + x34*x64 - x40*x45 + x41*(x65 - x66) - 8*x62 - x63) - s_zz*x0*x27 - x17*(x10*x6 + x19 + x48*x5) - x33*(grz + x29 - x30 + x32)/2 + x38*(x11 + x35 - x36 + x37) - x59*y*(-grr*krr*x8 + gzz*(x34*x4 + x58) + x10*x55 - 2*x54));

double momentum_r = x69*(-grr*x70 + grz_z*x1*x72 - krr_r*x77 - krr_z*x76 + krz_r*x76 + krz_z*x74 - kzz_r*x74 + s_r*x90*(-grz*x83 + x86 + x89) - s_z*x59*x84 - x15*y*(-x11*x66 + x31*x68 + 2*x67 + x85) - x35*x7*x75 + x39*(-grz*x11*(3*x48 + x5) + x89 + x91 + x92)/2 + x41*(krz*x21 + x3 - 3*x78)/2 + x42*x84 + x54*x7 - x62*x73 - x63*x73/2 - x79*y_r + x82*(krr*x11 - x56 + x81));

double momentum_z = x69*(grz*x64*x72 - gzz_z*x15*x51*x73 - krr_z*x12*(-x10*x65 + x67 + x85) + krz*x0*x8 + krz_r*x14*x25 + krz_z*x76 - kzz_r*x76 + s_r*x59*x94 - s_z*x90*(krr*x9 - x11*(krr*x44 + x81) + x92 + x95) + x2*x57 - x20*x29*x75 - x20*x93 - x30*(x22*(x58 + x80) + x44*x57 + x86 - x91 + x95)/2 + x31*x7*x94 + x52*x53*(grz*kzz - gzz*krz) - x79*y_z + x82*(grz*x55 + krz*x11 - krz*x44));

