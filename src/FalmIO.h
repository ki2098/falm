#ifndef FALM_FALMIO_H
#define FALM_FALMIO_H

#include <fstream>
#include <stdio.h>
#include <string>
#include "typedef.h"
#include "matrix.h"

namespace Falm {

class FalmIO {
public:

static void readControlVolumeFile(std::string srcpath, Matrix<REAL> &x, Matrix<REAL> &y, Matrix<REAL> &z, Matrix<REAL> &hx, Matrix<REAL> &hy, Matrix<REAL> &hz, INT3 &idmax) {
    std::ifstream cvfs(srcpath);
    cvfs >> idmax[0] >> idmax[1] >> idmax[2];
    x.alloc(idmax[0], 1, HDC::Host);
    y.alloc(idmax[1], 1, HDC::Host);
    z.alloc(idmax[2], 1, HDC::Host);
    hx.alloc(idmax[0], 1, HDC::Host);
    hy.alloc(idmax[1], 1, HDC::Host);
    hz.alloc(idmax[2], 1, HDC::Host);
    for (int i = 0; i < idmax[0]; i ++) {
        cvfs >> x(i) >> hx(i);
    }
    for (int j = 0; j < idmax[1]; j ++) {
        cvfs >> y(j) >> hy(j);
    }
    for (int k = 0; k < idmax[2]; k ++) {
        cvfs >> z(k) >> hz(k);
    }
    cvfs.close();
}

};

}

#endif