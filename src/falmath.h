#ifndef FALM_FALMATH_H
#define FALM_FALMATH_H

#include <math.h>
#include <string>
#include "devdefine.h"

namespace Falm {

static const Real Pi = M_PI;

__host__ __device__ static inline Real square(Real a) {
    return a * a;
}

__host__ __device__ static inline Real cubic(Real a) {
    return a * a * a;
}

__host__ __device__ static inline Real floormod(Real a, Real b) {
    return a - floor(a / b) * b;
}

__host__ __device__ static inline Real truncmod(Real a, Real b) {
    return a - trunc(a / b) * b;
}

template<typename T>
__host__ __device__ static inline Int sign(T a) {
    return (a>0) - (a<0);
}

enum class EulerAngle {Empty, Roll, Pitch, Yaw};

static std::string get_euler_angle_name(EulerAngle ea) {
    if (ea == EulerAngle::Roll) {
        return "Roll";
    } else if (ea == EulerAngle::Pitch) {
        return "Pitch";
    } else if (ea == EulerAngle::Yaw) {
        return "Yaw";
    } else {
        return "Not defined";
    }
}

__host__ __device__ static inline Real rad2deg(Real rad) {
    return rad*180./Pi;
}

__host__ __device__ static inline Real deg2rad(Real deg) {
    return deg*Pi/180.;
}

__host__ __device__ static Real3 one_angle_frame_rotation(const Real3 &x, const Real3 &angle, EulerAngle angle_type) {
    const Real &X = x[0], &Y = x[1], &Z = x[2];
    if (angle_type == EulerAngle::Roll) {
        const Real COS = cos(angle[0]), SIN = sin(angle[0]);
        return Real3 {{
              X,
              COS*Y + SIN*Z,
            - SIN*Y + COS*Z
        }};
    } else if (angle_type == EulerAngle::Pitch) {
        const Real COS = cos(angle[1]), SIN = sin(angle[1]);
        return Real3 {{
              COS*X - SIN*Z,
              Y,
              SIN*X + COS*Z
        }};
    } else if (angle_type == EulerAngle::Yaw) {
        const Real COS = cos(angle[2]), SIN = sin(angle[2]);
        return Real3 {{
              COS*X + SIN*Y,
            - SIN*X + COS*Y,
              Z
        }};
    } else {
        return x;
    }
}

__host__ __device__ static Real3 one_angle_frame_rotation_dt(const Real3 &x, const Real3 &v, const Real3 &angle, const Real3 &omega, EulerAngle angle_type) {
    const Real &X = x[0], &Y = x[1], &Z = x[2];
    const Real &U = v[0], &V = v[1], &W = v[2];
    if (angle_type == EulerAngle::Roll) {
        const Real COS = cos(angle[0]), SIN = sin(angle[0]);
        const Real &OMEGA = omega[0];
        Real3 dRdtx {{
            0.,
            OMEGA*(- SIN*Y + COS*Z),
            OMEGA*(- COS*Y - SIN*Z)
        }};
        Real3 Rdxdt {{
              U,
              COS*V + SIN*W,
            - SIN*V + COS*W
        }};
        return dRdtx + Rdxdt;
    } else if (angle_type == EulerAngle::Pitch) {
        const Real COS = cos(angle[1]), SIN = sin(angle[1]);
        const Real &OMEGA = omega[1];
        Real3 dRdtx {{
            OMEGA*(- SIN*X - COS*Z),
            0.,
            OMEGA*(  COS*X - SIN*Z)
        }};
        Real3 Rdxdt {{
              COS*U - SIN*W,
              V,
              SIN*U + COS*W
        }};
        return dRdtx + Rdxdt;
    } else if (angle_type == EulerAngle::Yaw) {
        const Real COS = cos(angle[2]), SIN = sin(angle[2]);
        const Real &OMEGA = omega[2];
        Real3 dRdtx {{
             OMEGA*(-SIN*X + COS*Y),
             OMEGA*(-COS*X - SIN*Y),
             0.
        }};
        Real3 Rdxdt {{
              COS*U + SIN*V,
            - SIN*U + COS*V,
              W
        }};
        return dRdtx + Rdxdt;
    } else {
        return v;
    }
}

}

#endif
