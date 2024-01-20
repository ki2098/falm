#ifndef FALM_POSTFALM_FVEC_HPP
#define FALM_POSTFALM_FVEC_HPP

#include <stdlib.h>

template<typename T>
class farray {
    T *_ptr;
    size_t _size;
public:
    farray() : _ptr(nullptr), _size(0) {}

    farray(size_t n) : _size(n) {
        _ptr = new T[n];
    }

    ~farray() {
        delete[] _ptr;
    }

    T &operator[](size_t i) {
        return *(_ptr + i);
    }

    const T &operator[](size_t i) const {
        return *(_ptr + i);
    }

    T *ptr() {
        return _ptr;
    }

    const T *ptr() const {
        return _ptr;
    }

    size_t size() const {
        return _size;
    }
};

#endif