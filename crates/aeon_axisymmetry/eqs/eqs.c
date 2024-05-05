#include "stdint.h"
#include "math.h"

#define FIELD_FIRST(NAME) double NAME, NAME##_r, NAME##_z
#define FIELD_SECOND(NAME) double NAME, NAME##_r, NAME##_z, NAME##_rr, NAME##_rz, NAME##_zz

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
    double theta_t, zr_t, zz_t;
    double lapse_t, shiftr_t, shiftz_t;
} HyperbolicDerivs;

#define VARS_FIRST(NAME) double NAME = vars.NAME, NAME##_r = vars.NAME##_r, NAME##_z = vars.NAME##_z
#define VARS_SECOND(NAME) double NAME = vars.NAME, NAME##_r = vars.NAME##_r, NAME##_z = vars.NAME##_z, NAME##_rr = vars.NAME##_rr, NAME##_rz = vars.NAME##_rz, NAME##_zz = vars.NAME##_zz

#define DERIVS(NAME) derivs.NAME##_t = NAME##_t

/// @brief Compute temporal derivatives for each tensor field on interior of domain.
/// @param vars Current values and spatial derivatives of each tensor field.
/// @param rho Position along rho axis.
/// @param z Position along z axis.
/// @return Struct containing derivatives.
HyperbolicDerivs hyperbolic(HyperbolicSystem vars, double rho, double z)
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
HyperbolicDerivs hyperbolic_regular(HyperbolicSystem vars, double rho, double z)
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

/// @brief Steady state system solved for initial data.
typedef struct InitialSystem
{
    FIELD_SECOND(psi);
    FIELD_SECOND(s);

} InitialSystem;

/// @brief Derivatives for initial data solver.
typedef struct InitialDerivs
{
    double psi_t;
} InitialDerivs;

InitialDerivs initial(InitialSystem vars, double rho, double z)
{
    VARS_SECOND(psi);
    VARS_SECOND(s);

#include "initial.h"

    InitialDerivs derivs;
    derivs.psi_t = op;

    return derivs;
}

InitialDerivs initial_regular(InitialSystem vars, double rho, double z)
{
    VARS_SECOND(psi);
    VARS_SECOND(s);

#include "initial.h"

    InitialDerivs derivs;
    derivs.psi_t = op;

    return derivs;
}

// typedef struct Constraints
// {
//     double hamiltonian;
//     double momentum_r, momentum_z;
// } Constraints;

// Constraints constraints(System vars, double rho, double z)
// {
//     VARS_SECOND(grr);
//     VARS_SECOND(gzz);
//     VARS_SECOND(grz);
//     VARS_SECOND(s);

//     VARS_FIRST(krr);
//     VARS_FIRST(kzz);
//     VARS_FIRST(krz);
//     VARS_FIRST(y);

//     VARS_SECOND(lapse);
//     VARS_FIRST(shiftr);
//     VARS_FIRST(shiftz);

//     VARS_FIRST(theta);
//     VARS_FIRST(zr);
//     VARS_FIRST(zz);

// // Include generated code.
// #include "constraints.h"

//     Constraints cons;

//     cons.hamiltonian = hamiltonian;
//     cons.momentum_r = momentum_r;
//     cons.momentum_z = momentum_z;

//     return cons;
// }

// Constraints constraints_regular(System vars, double rho, double z)
// {
//     VARS_SECOND(grr);
//     VARS_SECOND(gzz);
//     VARS_SECOND(grz);
//     VARS_SECOND(s);

//     VARS_FIRST(krr);
//     VARS_FIRST(kzz);
//     VARS_FIRST(krz);
//     VARS_FIRST(y);

//     VARS_SECOND(lapse);
//     VARS_FIRST(shiftr);
//     VARS_FIRST(shiftz);

//     VARS_FIRST(theta);
//     VARS_FIRST(zr);
//     VARS_FIRST(zz);

// // Include generated code.
// #include "constraints_regular.h"

//     Constraints cons;

//     cons.hamiltonian = hamiltonian;
//     cons.momentum_r = momentum_r;
//     cons.momentum_z = momentum_z;

//     return cons;
// }