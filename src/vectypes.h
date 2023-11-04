#ifndef FALM_VECTYPES_H
#define FALM_VECTYPES_H

namespace Falm {

template<typename T>
struct VECTOR3 {
    T x, y, z;

    VECTOR3<T> operator+(const VECTOR3<T> &rhs) const {
        return VECTOR3<T>{
            x + rhs.x,
            y + rhs.y,
            z + rhs.z
        };
    }

    VECTOR3<T> operator-(const VECTOR3<T> &rhs) const {
        return VECTOR3<T>{
            x - rhs.x,
            y - rhs.y,
            z - rhs.z
        };
    }

    VECTOR3<T> operator+=(const VECTOR3<T> &rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    VECTOR3<T> operator-=(const VECTOR3<T> &rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }
};

template<typename T>
struct VECTOR2 {
    T x, y;

    VECTOR2<T> operator+(const VECTOR2<T> &rhs) const {
        return VECTOR2<T>{
            x + rhs.x,
            y + rhs.y,
        };
    }

    VECTOR2<T> operator-(const VECTOR2<T> &rhs) const {
        return VECTOR2<T>{
            x - rhs.x,
            y - rhs.y,
        };
    }

    VECTOR2<T> &operator+=(const VECTOR2<T> &rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    VECTOR2<T> &operator-=(const VECTOR2<T> &rhs) {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }
};

template<typename T, size_t N>
struct VECTOR {
    T _mv[N];

    T &operator[](size_t i) {return _mv[i];}
    const T &operator[](size_t i) const {return _mv[i];}

    VECTOR operator+(const VECTOR &v) const {
        VECTOR vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _mv[i] + v[i];
    }

    VECTOR operator-(const VECTOR &v) const {
        VECTOR vv;
        for (size_t i = 0; i < N; i ++) vv[i] = _mv[i] - v[i];
    }

    VECTOR operator+=(const VECTOR &v) {
        for (size_t i = 0; i < N; i ++)  _mv[i] += v[i];
        return *this;
    }

    VECTOR operator-=(const VECTOR &v) {
        for (size_t i = 0; i < N; i ++)  _mv[i] -= v[i];
        return *this;
    }

    bool operator==(const VECTOR &v) const {
        for (size_t i = 0; i < N; i ++) {
            if (_mv[i] != v[i]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const VECTOR &v) const {
        for (size_t i = 0; i < N; i ++) {
            if _mv[i] == v[i] {
                return false;
            }
        }
        return true;
    }

    VECTOR &operator=(const VECTOR &v) {
        for (size_t i = 0; i < N; i ++) _mv[i] = v[i];
        return *this;
    }
};

}

#endif