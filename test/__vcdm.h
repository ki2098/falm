#ifndef __TEST___VCDM_H__
#define __TEST___VCDM_H__

#include "../src/vcdm/VCDM.h"

class __VCDM {
public:
    Vcdm::VCDM<double> vcdm;

    std::string makefilename(const std::string &prefix);
};

#endif