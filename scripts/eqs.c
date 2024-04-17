#include "stdint.h"
#include "math.h"

#define FIELD_FIRST(NAME) double NAME, NAME##_r, NAME##_z
#define FIELD_SECOND(NAME) double NAME, NAME##_r, NAME##_z, NAME##_rr, NAME##_rz, NAME##_zz

typedef struct HyperbolicVars
{
    FIELD_SECOND(grr);
    FIELD_SECOND(gzz);
    FIELD_SECOND(grz);
    FIELD_SECOND(s);

    FIELD_FIRST(Krr);
    FIELD_FIRST(Kzz);
    FIELD_FIRST(Krz);
    FIELD_FIRST(Y);

    FIELD_SECOND(lapse);
    FIELD_FIRST(shiftr);
    FIELD_FIRST(shiftz);

    FIELD_FIRST(theta);
    FIELD_FIRST(Zr);
    FIELD_FIRST(Zz);
} HyperbolicVars;

typedef struct HyperbolicDerivs
{
    double grr_t, gzz_t, grz_t;
    double Krr_t, Kzz_t, Krz_t;
    double theta_t, Zr_t, Zz_t;
} HyperbolicDerivs;

#define VARS_FIRST(NAME) double NAME = vars.##NAME##, NAME##_r = vars.##NAME##_r, NAME##_z = vars.##NAME##_z
#define VARS_SECOND(NAME) double NAME = vars.##NAME##, NAME##_r = vars.##NAME##_r, NAME##_z = vars.##NAME##_z, NAME##_rr = vars.##NAME##_rr, NAME##_rz = vars.##NAME##_rz, NAME##_zz = vars.##NAME##_zz

#define DERIVS(NAME) derivs.##NAME##_t = NAME##_t

HyperbolicDerivs hyperbolic(HyperbolicVars vars)
{
    VARS_SECOND(grr);
    VARS_SECOND(gzz);
    VARS_SECOND(grz);
    VARS_SECOND(s);

    VARS_FIRST(Krr);
    VARS_FIRST(Kzz);
    VARS_FIRST(Krz);
    VARS_FIRST(Y);

    VARS_SECOND(lapse);
    VARS_FIRST(shiftr);
    VARS_FIRST(shiftz);

    VARS_FIRST(theta);
    VARS_FIRST(Zr);
    VARS_FIRST(Zz);

// Include generated code.
#include "hyperbolic.h"

    // Return struct containing derivatives of each variable.
    HyperbolicDerivs derivs;

    DERIVS(grr);
    DERIVS(gzz);
    DERIVS(grz);
    // DERIVS(s);
    DERIVS(Krr);
    DERIVS(Kzz);
    DERIVS(Krz);
    // DERIVS(Y);
    DERIVS(theta);
    DERIVS(Zr);
    DERIVS(Zz);

    return derivs;
}