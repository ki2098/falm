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

}

#endif