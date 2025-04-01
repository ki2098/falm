#ifndef FALM_FLAG_H
#define FALM_FLAG_H

#include "typedef.h"

namespace Falm {

class MCP {
public:
    static const Flag Empty   = 0;
    static const Flag Hst2Hst = 1;
    static const Flag Hst2Dev = 2;
    static const Flag Dev2Hst = 4;
    static const Flag Dev2Dev = 8;
};

class HDC {
public:
    static const Flag Empty  = 0;
    static const Flag Host   = 1;
    static const Flag Device = 2;
    static const Flag HstDev = Host | Device;
};

class Color {
public:
    static const Int Black = 0;
    static const Int Red   = 1;
};

}

#endif
