#ifndef FALM_CPMIO_H
#define FALM_CPMIO_H

#include <string>
#include "CPMBase.h"

namespace Falm {

class CPMio {
public:
    CPMBase   *base;
    std::string dir = ".";

    CPMio(CPMBase *_base, const std::string &_dir) {
        base = _base;
        dir  = _dir;
    }

    void writeIndexFile() {
        std::string fname = dir + "/" + "index.txt";
        FILE *file = fopen(fname.c_str(), "w");
        fprintf(file, "%d\n", base->gc);
        fprintf(file, "%d %d %d\n", base->global.shape.x, base->global.shape.y, base->global.shape.z);
        fprintf(file, "%d\n", base->size);
        fprintf(file, "%d %d %d\n", base->shape.x, base->shape.y, base->shape.z);

        for (int i = 0; i < base->size; i ++) {
            fprintf(file, "%d\n", i);
        }

        fclose(file);
    }

};

}

#endif