#ifndef FALM_FLAG_H
#define FALM_FLAG_H

#include "typedef.h"

namespace Falm {

class MCpType {
public:
    static const FLAG Empty   = 0;
    static const FLAG Hst2Hst = 1;
    static const FLAG Hst2Dev = 2;
    static const FLAG Dev2Hst = 4;
    static const FLAG Dev2Dev = 8;
};

class HDCType {
public:
    static const FLAG Empty  = 0;
    static const FLAG Host   = 1;
    static const FLAG Device = 2;
    static const FLAG HstDev = Host | Device;
};

class Color {
public:
    static const INT Black = 0;
    static const INT Red   = 1;
};

}

#endif
