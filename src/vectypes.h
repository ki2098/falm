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
struct Vector {
    T _m_vector[N];

    __host__ __device__ T &operator[](size_t i) {return _m_vector[i];}
    __host__ __device__ const T &operator[](size_t i) const {return _m_vector[i];}

    // @ vector

    __host__ __device__ Vector operator-() const {
        Vector vv;
        for (size_t i = 0; i < N; i ++) vv[i] = - _m_vector[i];
        return vv;
    }

    // vector @ vector

    __host__ __device__ Vector operator+(const Vector &v) const {
        Vector vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _m_vector[i] + v[i];
        return vv;
    }

    __host__ __device__ Vector operator-(const Vector &v) const {
        Vector vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _m_vector[i] - v[i];
        return vv;
    }

    __host__ __device__ Vector operator*(const Vector &v) const {
        Vector vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _m_vector[i] * v[i];
        return vv;
    }

    __host__ __device__ Vector operator/(const Vector &v) const {
        Vector vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _m_vector[i] / v[i];
        return vv;
    }

    // vector @= vector

    __host__ __device__ Vector &operator+=(const Vector &v) {
        for (size_t i = 0; i < N; i ++)  _m_vector[i] += v[i];
        return *this;
    }

    __host__ __device__ Vector &operator-=(const Vector &v) {
        for (size_t i = 0; i < N; i ++)  _m_vector[i] -= v[i];
        return *this;
    }

    __host__ __device__ Vector &operator*=(const Vector &v) {
        for (size_t i = 0; i < N; i ++)  _m_vector[i] *= v[i];
        return *this;
    }

    __host__ __device__ Vector &operator/=(const Vector &v) {
        for (size_t i = 0; i < N; i ++)  _m_vector[i] /= v[i];
        return *this;
    }

    // vector @ scalar

    __host__ __device__ Vector operator+(const T &s) const {
        Vector vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _m_vector[i] + s;
        return vv;
    }

    __host__ __device__ Vector operator-(const T &s) const {
        Vector vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _m_vector[i] - s;
        return vv;
    }

    __host__ __device__ Vector operator*(const T &s) const {
        Vector vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _m_vector[i] * s;
        return vv;
    }

    __host__ __device__ Vector operator/(const T &s) const {
        Vector vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _m_vector[i] / s;
        return vv;
    }

    // vector @= scalar

    __host__ __device__ Vector &operator+=(const T &s) {
        for (size_t i = 0; i < N; i ++)  _m_vector[i] += s;
        return *this;
    }

    __host__ __device__ Vector &operator-=(const T &s) {
        for (size_t i = 0; i < N; i ++)  _m_vector[i] -= s;
        return *this;
    }

    __host__ __device__ Vector &operator*=(const T &s) {
        for (size_t i = 0; i < N; i ++)  _m_vector[i] *= s;
        return *this;
    }

    __host__ __device__ Vector &operator/=(const T &s) {
        for (size_t i = 0; i < N; i ++)  _m_vector[i] /= s;
        return *this;
    }

    // vector ? vector

    __host__ __device__ bool operator==(const Vector &v) const {
        for (size_t i = 0; i < N; i ++) {
            if (_m_vector[i] != v[i]) {
                return false;
            }
        }
        return true;
    }

    __host__ __device__ bool operator!=(const Vector &v) const {
        for (size_t i = 0; i < N; i ++) {
            if (_m_vector[i] != v[i]) {
                return true;
            }
        }
        return false;
    }

    // vector ? scalar

    __host__ __device__ bool operator==(const T &s) const {
        for (size_t i = 0; i < N; i ++) {
            if (_m_vector[i] != s) {
                return false;
            }
        }
        return true;
    }

    __host__ __device__ bool operator!=(const T &s) const {
        for (size_t i = 0; i < N; i ++) {
            if (_m_vector[i] != s) {
                return true;
            }
        }
        return false;
    }

    // __host__ __device__ VECTOR &operator=(const VECTOR &v) {
    //     for (size_t i = 0; i < N; i ++) _mv[i] = v[i];
    //     return *this;
    // }
};

// scalar @ vector

template<typename T, size_t N>
__host__ __device__ Vector<T, N> operator+(const T &s, const Vector<T, N> &v) {
    Vector<T, N> vv;
    for (size_t i = 0; i < N; i ++) vv[i] = s + v[i];
    return vv;
}

template<typename T, size_t N>
__host__ __device__ Vector<T, N> operator-(const T &s, const Vector<T, N> &v) {
    Vector<T, N> vv;
    for (size_t i = 0; i < N; i ++) vv[i] = s - v[i];
    return vv;
}

template<typename T, size_t N>
__host__ __device__ Vector<T, N> operator*(const T &s, const Vector<T, N> &v) {
    Vector<T, N> vv;
    for (size_t i = 0; i < N; i ++) vv[i] = s * v[i];
    return vv;
}

template<typename T, size_t N>
__host__ __device__ Vector<T, N> operator/(const T &s, const Vector<T, N> &v) {
    Vector<T, N> vv;
    for (size_t i = 0; i < N; i ++) vv[i] = s / v[i];
    return vv;
}

// scalar ? vector

template<typename T, size_t N>
__host__ __device__ bool operator==(const T &s, const Vector<T, N> &v) {
    for (size_t i = 0; i < N; i ++) {
        if (s != v[i]) return false;
    }
    return true;
}

template<typename T, size_t N>
__host__ __device__ bool operator!=(const T &s, const Vector<T, N> &v) {
    for (size_t i = 0; i < N; i ++) {
        if (s != v[i]) return true;
    }
    return false;
}

}

#endif