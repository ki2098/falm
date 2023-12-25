#ifndef FALM_ERROR_H
#define FALM_ERROR_H

#include <stdio.h>
#include "typedef.h"

namespace Falm {

class FalmErr {
public:
    static const INT success      = 0;
    static const INT mallocErr    = 1;
    static const INT setUpFileErr = 2;
    static const INT cuErrMask    = 100;
};

static inline void FalmErrCheck(INT errCode, const char *name, const char *file, int line) {
    if (errCode != FalmErr::success) {
        fprintf(stderr, "%s at %s %d failed with %d\n", name, file, line, errCode);
        // exit((int)errCode);
    }
}

}

#define falmErrCheckMacro(expr) Falm::FalmErrCheck(expr, #expr, __FILE__, __LINE__)

#endif