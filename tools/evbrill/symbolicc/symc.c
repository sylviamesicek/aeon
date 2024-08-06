#include "stdint.h"
#include "math.h"

#define FIELD_FIRST(NAME) \
    double NAME;          \
    double NAME##_r;      \
    double NAME##_z;

#define FIELD_SECOND(NAME) \
    double NAME;           \
    double NAME##_r;       \
    double NAME##_z;       \
    double NAME##_rr;      \
    double NAME##_rz;      \
    double NAME##_zz;

/// @brief System of fields that evolve according to the z4 axisymmetric EFEs.
typedef struct HyperbolicSystem
{
    FIELD_SECOND(grr);
    FIELD_SECOND(grz);
    FIELD_SECOND(gzz);
    FIELD_SECOND(s);

    FIELD_FIRST(krr);
    FIELD_FIRST(krz);
    FIELD_FIRST(kzz);
    FIELD_FIRST(y);

    FIELD_SECOND(lapse);
    FIELD_FIRST(shiftr);
    FIELD_FIRST(shiftz);

    FIELD_FIRST(theta);
    FIELD_FIRST(zr);
    FIELD_FIRST(zz);
} HyperbolicSystem;

typedef struct HyperbolicDerivs
{
    double grr_t, grz_t, gzz_t, s_t;
    double krr_t, krz_t, kzz_t, y_t;
    double lapse_t, shiftr_t, shiftz_t;
    double theta_t, zr_t, zz_t;
} HyperbolicDerivs;

#define VARS_FIRST(NAME) double NAME = vars.NAME, NAME##_r = vars.NAME##_r, NAME##_z = vars.NAME##_z
#define VARS_SECOND(NAME) double NAME = vars.NAME, NAME##_r = vars.NAME##_r, NAME##_z = vars.NAME##_z, NAME##_rr = vars.NAME##_rr, NAME##_rz = vars.NAME##_rz, NAME##_zz = vars.NAME##_zz

#define DERIVS(NAME) derivs.NAME##_t = NAME##_t

/// @brief Compute temporal derivatives for each tensor field on interior of domain.
/// @param vars Current values and spatial derivatives of each tensor field.
/// @param rho Position along rho axis.
/// @param z Position along z axis.
/// @return Struct containing derivatives.
HyperbolicDerivs hyperbolic_sys(HyperbolicSystem vars, double rho, double z)
{
    VARS_SECOND(grr);
    VARS_SECOND(gzz);
    VARS_SECOND(grz);
    VARS_SECOND(s);

    VARS_FIRST(krr);
    VARS_FIRST(kzz);
    VARS_FIRST(krz);
    VARS_FIRST(y);

    VARS_SECOND(lapse);
    VARS_FIRST(shiftr);
    VARS_FIRST(shiftz);

    VARS_FIRST(theta);
    VARS_FIRST(zr);
    VARS_FIRST(zz);

// Include generated code.
#include "hyperbolic.h"

    // Return struct containing derivatives of each variable.
    HyperbolicDerivs derivs;

    DERIVS(grr);
    DERIVS(gzz);
    DERIVS(grz);
    DERIVS(s);
    DERIVS(krr);
    DERIVS(kzz);
    DERIVS(krz);
    DERIVS(y);
    DERIVS(theta);
    DERIVS(zr);
    DERIVS(zz);
    DERIVS(lapse);
    DERIVS(shiftr);
    DERIVS(shiftz);

    return derivs;
}

/// @brief Compute temporal derivatives for each tensor field on axis.
/// @param vars Current values and spatial derivatives of each tensor field.
/// @param rho Position along rho axis.
/// @param z Position along z axis.
/// @return Struct containing derivatives.
HyperbolicDerivs hyperbolic_regular_sys(HyperbolicSystem vars, double rho, double z)
{
    VARS_SECOND(grr);
    VARS_SECOND(gzz);
    VARS_SECOND(grz);
    VARS_SECOND(s);

    VARS_FIRST(krr);
    VARS_FIRST(kzz);
    VARS_FIRST(krz);
    VARS_FIRST(y);

    VARS_SECOND(lapse);
    VARS_FIRST(shiftr);
    VARS_FIRST(shiftz);

    VARS_FIRST(theta);
    VARS_FIRST(zr);
    VARS_FIRST(zz);

// Include generated code.
#include "hyperbolic_regular.h"

    // Return struct containing derivatives of each variable.
    HyperbolicDerivs derivs;

    DERIVS(grr);
    DERIVS(gzz);
    DERIVS(grz);
    DERIVS(s);
    DERIVS(krr);
    DERIVS(kzz);
    DERIVS(krz);
    DERIVS(y);
    DERIVS(theta);
    DERIVS(zr);
    DERIVS(zz);
    DERIVS(lapse);
    DERIVS(shiftr);
    DERIVS(shiftz);

    return derivs;
}

typedef struct Geometric
{
    double ricci_rr, ricci_rz, ricci_zz;
} Geometric;

Geometric geometric_sys(HyperbolicSystem vars, double rho, double z) {
    VARS_SECOND(grr);
    VARS_SECOND(gzz);
    VARS_SECOND(grz);
    VARS_SECOND(s);

    VARS_FIRST(krr);
    VARS_FIRST(kzz);
    VARS_FIRST(krz);
    VARS_FIRST(y);

    VARS_SECOND(lapse);
    VARS_FIRST(shiftr);
    VARS_FIRST(shiftz);

    VARS_FIRST(theta);
    VARS_FIRST(zr);
    VARS_FIRST(zz);

#include "geometric.h"

    Geometric result;

    result.ricci_rr = ricci_rr;
    result.ricci_rz = ricci_rz;
    result.ricci_zz = ricci_zz;

    return result;
}