#ifndef FALM_FALMATH_H
#define FALM_FALMATH_H

#include <math.h>
#include <string>
#include "devdefine.h"

namespace Falm {

static const REAL Pi = M_PI;

__host__ __device__ static inline REAL square(REAL a) {
    return a * a;
}

__host__ __device__ static inline REAL cubic(REAL a) {
    return a * a * a;
}

__host__ __device__ static inline REAL floormod(REAL a, REAL b) {
    return a - floor(a / b) * b;
}

__host__ __device__ static inline REAL truncmod(REAL a, REAL b) {
    return a - trunc(a / b) * b;
}

template<typename T>
__host__ __device__ static inline INT sign(T a) {
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

__host__ __device__ static inline REAL rad2deg(REAL rad) {
    return rad*180./Pi;
}

__host__ __device__ static inline REAL deg2rad(REAL deg) {
    return deg*Pi/180.;
}

__host__ __device__ static REAL3 one_angle_frame_rotation(const REAL3 &x, const REAL3 &angle, EulerAngle angle_type) {
    const REAL &X = x[0], &Y = x[1], &Z = x[2];
    if (angle_type == EulerAngle::Roll) {
        const REAL COS = cos(angle[0]), SIN = sin(angle[0]);
        return REAL3 {{
              X,
              COS*Y + SIN*Z,
            - SIN*Y + COS*Z
        }};
    } else if (angle_type == EulerAngle::Pitch) {
        const REAL COS = cos(angle[1]), SIN = sin(angle[1]);
        return REAL3 {{
              COS*X - SIN*Z,
              Y,
              SIN*X + COS*Z
        }};
    } else if (angle_type == EulerAngle::Yaw) {
        const REAL COS = cos(angle[2]), SIN = sin(angle[2]);
        return REAL3 {{
              COS*X + SIN*Y,
            - SIN*X + COS*Y,
              Z
        }};
    } else {
        return x;
    }
}

__host__ __device__ static REAL3 one_angle_frame_rotation_dt(const REAL3 &x, const REAL3 &v, const REAL3 &angle, const REAL3 &omega, EulerAngle angle_type) {
    const REAL &X = x[0], &Y = x[1], &Z = x[2];
    const REAL &U = v[0], &V = v[1], &W = v[2];
    if (angle_type == EulerAngle::Roll) {
        const REAL COS = cos(angle[0]), SIN = sin(angle[0]);
        const REAL &OMEGA = omega[0];
        REAL3 dRdtx {{
            0.,
            OMEGA*(- SIN*Y + COS*Z),
            OMEGA*(- COS*Y - SIN*Z)
        }};
        REAL3 Rdxdt {{
              U,
              COS*V + SIN*W,
            - SIN*V + COS*W
        }};
        return dRdtx + Rdxdt;
    } else if (angle_type == EulerAngle::Pitch) {
        const REAL COS = cos(angle[1]), SIN = sin(angle[1]);
        const REAL &OMEGA = omega[1];
        REAL3 dRdtx {{
            OMEGA*(- SIN*X - COS*Z),
            0.,
            OMEGA*(  COS*X - SIN*Z)
        }};
        REAL3 Rdxdt {{
              COS*U - SIN*W,
              V,
              SIN*U + COS*W
        }};
        return dRdtx + Rdxdt;
    } else if (angle_type == EulerAngle::Yaw) {
        const REAL COS = cos(angle[2]), SIN = sin(angle[2]);
        const REAL &OMEGA = omega[2];
        REAL3 dRdtx {{
             OMEGA*(-SIN*X + COS*Y),
             OMEGA*(-COS*X - SIN*Y),
             0.
        }};
        REAL3 Rdxdt {{
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
