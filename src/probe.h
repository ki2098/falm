#ifndef FALM_PROBE_H
#define FALM_PROBE_H

#include <fstream>
#include "FalmMeshBase.h"
#include "matrix.h"

namespace Falm {

class Probe {
    INT3 shape;
    Matrix<REAL> *src;
    FalmBaseMesh *mesh;
    std::ofstream dst;

    Probe(FalmBaseMesh *mesh, Matrix<REAL> *src, std::string path) : mesh(mesh), src(src) {
        shape[0] = mesh->x.shape[0];
        shape[1] = mesh->y.shape[0];
        shape[2] = mesh->z.shape[0];
        dst.open(path);
    }
};

}

#endif