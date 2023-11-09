#ifndef FALM_VECTYPES_H
#define FALM_VECTYPES_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace Falm {

// template<typename T>
// struct VECTOR3 {
//     T x, y, z;

//     VECTOR3<T> operator+(const VECTOR3<T> &rhs) const {
//         return VECTOR3<T>{
//             x + rhs[0],
//             y + rhs[1],
//             z + rhs[2]
//         };
//     }

//     VECTOR3<T> operator-(const VECTOR3<T> &rhs) const {
//         return VECTOR3<T>{
//             x - rhs[0],
//             y - rhs[1],
//             z - rhs[2]
//         };
//     }

//     VECTOR3<T> operator+=(const VECTOR3<T> &rhs) {
//         x += rhs[0];
//         y += rhs[1];
//         z += rhs[2];
//         return *this;
//     }

//     VECTOR3<T> operator-=(const VECTOR3<T> &rhs) {
//         x -= rhs[0];
//         y -= rhs[1];
//         z -= rhs[2];
//         return *this;
//     }
// };

// template<typename T>
// struct VECTOR2 {
//     T x, y;

//     VECTOR2<T> operator+(const VECTOR2<T> &rhs) const {
//         return VECTOR2<T>{
//             x + rhs[0],
//             y + rhs[1],
//         };
//     }

//     VECTOR2<T> operator-(const VECTOR2<T> &rhs) const {
//         return VECTOR2<T>{
//             x - rhs[0],
//             y - rhs[1],
//         };
//     }

//     VECTOR2<T> &operator+=(const VECTOR2<T> &rhs) {
//         x += rhs[0];
//         y += rhs[1];
//         return *this;
//     }

//     VECTOR2<T> &operator-=(const VECTOR2<T> &rhs) {
//         x -= rhs[0];
//         y -= rhs[1];
//         return *this;
//     }
// };

template<typename T, size_t N>
struct VECTOR {
    T _m_vector[N];

    __host__ __device__ T &operator[](size_t i) {return _m_vector[i];}
    __host__ __device__ const T &operator[](size_t i) const {return _m_vector[i];}

    __host__ __device__ VECTOR operator+(const VECTOR &v) const {
        VECTOR vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _m_vector[i] + v[i];
        return vv;
    }

    __host__ __device__ VECTOR operator-(const VECTOR &v) const {
        VECTOR vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _m_vector[i] - v[i];
        return vv;
    }

    __host__ __device__ VECTOR operator+=(const VECTOR &v) {
        for (size_t i = 0; i < N; i ++)  _m_vector[i] += v[i];
        return *this;
    }

    __host__ __device__ VECTOR operator-=(const VECTOR &v) {
        for (size_t i = 0; i < N; i ++)  _m_vector[i] -= v[i];
        return *this;
    }

    __host__ __device__ bool operator==(const VECTOR &v) const {
        for (size_t i = 0; i < N; i ++) {
            if (_m_vector[i] != v[i]) {
                return false;
            }
        }
        return true;
    }

    __host__ __device__ bool operator!=(const VECTOR &v) const {
        for (size_t i = 0; i < N; i ++) {
            if (_m_vector[i] == v[i]) {
                return false;
            }
        }
        return true;
    }

    // __host__ __device__ VECTOR &operator=(const VECTOR &v) {
    //     for (size_t i = 0; i < N; i ++) _mv[i] = v[i];
    //     return *this;
    // }
};

}

#endif