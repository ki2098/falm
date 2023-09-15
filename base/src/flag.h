#ifndef FALM_FLAG_H
#define FALM_FLAG_H

namespace Falm {

class MCPTYPE {
public:
    static const unsigned int Empty   = 0;
    static const unsigned int Hst2Hst = 1;
    static const unsigned int Hst2Dev = 2;
    static const unsigned int Dev2Hst = 4;
    static const unsigned int Dev2Dev = 8;
};

class HDCTYPE {
public:
    static const unsigned int Empty  = 0;
    static const unsigned int Host   = 1;
    static const unsigned int Device = 2;
    static const unsigned int HstDev = Host | Device;
};

class COLOR {
public:
    static const unsigned int Black = 0;
    static const unsigned int Red   = 1;
};

}

#endif