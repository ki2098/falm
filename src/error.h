#ifndef FALM_ERROR_H
#define FALM_ERROR_H

#include <stdio.h>
#include "typedef.h"

namespace Falm {

class FalmErr {
public:
    static const int success      = 0;
    static const int mallocErr    = 1;
    static const int setUpFileErr = 2;
    static const int cpmBufReqErr = 3;
    static const int cpmBufReleaseErr = 4;
    static const int cpmNoVacantBuffer = 5;
    static const int cuErrMask    = 100;
};

static inline int FalmErrCheck(int errCode, const char *name, const char *file, int line) {
    if (errCode != FalmErr::success) {
        fprintf(stderr, "%s at %s %d failed with %d\n", name, file, line, errCode);
        // exit((int)errCode);
    }
    return errCode;
}

}

#define falmErrCheckMacro(expr) Falm::FalmErrCheck(expr, #expr, __FILE__, __LINE__)

#endif