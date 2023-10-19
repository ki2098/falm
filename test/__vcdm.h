#ifndef __TEST___VCDM_H__
#define __TEST___VCDM_H__

#include "../vcdm/VCDM.h"

class __VCDM {
public:
    Vcdm::VCDM<double> vcdm;

    std::string makefilename(const std::string &prefix);
};

#endif