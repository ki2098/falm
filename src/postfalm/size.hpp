#ifndef FALM_POSTFALM_SIZE_HPP
#define FALM_POSTFALM_SIZE_HPP

#include <stdlib.h>
#include <string>
#include <sstream>
#include "../nlohmann/json.hpp"

struct size2 {
    size_t _sz[2];
    size2(size_t _vx, size_t _vy) {
        _sz[0] = _vx;
        _sz[1] = _vy;
    }

    size_t &operator[](size_t i) {
        return _sz[i];
    }

    const size_t &operator[](size_t i) const {
        return _sz[i];
    }

    size_t product() const {
        return _sz[0] * _sz[1];
    }

    size_t idx(size_t i, size_t j) const {
        return i + j * _sz[0];
    }

};

struct size3 {
    size_t _sz[3];
    size3(size_t _vx, size_t _vy, size_t _vz) {
        _sz[0] = _vx;
        _sz[1] = _vy;
        _sz[2] = _vz;
    }

    size3(const nlohmann::json &jsz) {
        _sz[0] = jsz[0];
        _sz[1] = jsz[1];
        _sz[2] = jsz[2];
    }

    size3() {}

    size_t &operator[](size_t i) {
        return _sz[i];
    }

    const size_t &operator[](size_t i) const {
        return _sz[i];
    }

    size3 operator+(size_t _v) {
        size3 tmp;
        tmp[0] = _sz[0] + _v;
        tmp[1] = _sz[1] + _v;
        tmp[2] = _sz[2] + _v;
        return tmp;
    }

    std::string str() {
        std::stringstream tmp;
        tmp << "(" << _sz[0] << " " << _sz[1] << " " << _sz[2] << ")";
        return tmp.str();
    }

    size_t product() const {
        return _sz[0] * _sz[1] * _sz[2];
    }

    size_t idx(size_t i, size_t j, size_t k) const {
        return i + j * _sz[0] + k * _sz[0] * _sz[1];
    }
};

struct size4 {
    size_t _sz[4];

    size4(size_t _vx, size_t _vy, size_t _vz, size_t _vn) {
        _sz[0] = _vx;
        _sz[1] = _vy;
        _sz[2] = _vz;
        _sz[3] = _vn;
    }

    size4(const size3 &sz3, size_t n) {
        _sz[0] = sz3[0];
        _sz[1] = sz3[1];
        _sz[2] = sz3[2];
        _sz[3] = n;
    }

    size_t &operator[](size_t i) {
        return _sz[i];
    }

    const size_t &operator[](size_t i) const {
        return _sz[i];
    }

    size3 operator+(size_t _v) {
        size3 tmp;
        tmp[0] = _sz[0] + _v;
        tmp[1] = _sz[1] + _v;
        tmp[2] = _sz[2] + _v;
        return tmp;
    }

    size_t product() const {
        return _sz[0] * _sz[1] * _sz[2] * _sz[3];
    }

    size_t idx(size_t i, size_t j, size_t k, size_t n) const {
        return i + j * _sz[0] + k * _sz[0] * _sz[1] + n * _sz[0] * _sz[1] * _sz[2];
    }
};

#endif
