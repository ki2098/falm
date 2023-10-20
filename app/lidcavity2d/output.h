#ifndef _LID_CAVITY2D_OUTPUT_H_
#define _LID_CAVITY2D_OUTPUT_H_

#include <stdio.h>
#include "../../src/matrix.h"
#include "../../src/region.h"
#include "../../src/flag.h"

namespace LidCavity2d {

static void outputGridInfo(
    Falm::Matrix<Falm::REAL> &x,
    Falm::Matrix<Falm::REAL> &h,
    Falm::Matrix<Falm::REAL> &kx,
    Falm::Matrix<Falm::REAL> &g,
    Falm::Matrix<Falm::REAL> &ja,
    Falm::Region             &pdm
) {
    x.sync(Falm::MCpType::Dev2Hst);
    h.sync(Falm::MCpType::Dev2Hst);
    kx.sync(Falm::MCpType::Dev2Hst);
    g.sync(Falm::MCpType::Dev2Hst);
    ja.sync(Falm::MCpType::Dev2Hst);
    FILE *file = fopen("grid.csv", "w");
    fprintf(file, "x,y,z,h1,h2,h3,k1,k2,k3,g1,g2,g3,ja\n");
    for (Falm::INT k = 0; k < pdm.shape.z; k ++) {
        for (Falm::INT j = 0; j < pdm.shape.y; j ++) {
            for (Falm::INT i = 0; i < pdm.shape.x; i ++) {
                Falm::INT idx = Falm::IDX(i, j, k, pdm.shape);
                Falm::REAL x0 =  x(idx, 0);
                Falm::REAL x1 =  x(idx, 1);
                Falm::REAL x2 =  x(idx, 2);
                Falm::REAL h0 =  h(idx, 0);
                Falm::REAL h1 =  h(idx, 1);
                Falm::REAL h2 =  h(idx, 2);
                Falm::REAL k0 = kx(idx, 0);
                Falm::REAL k1 = kx(idx, 1);
                Falm::REAL k2 = kx(idx, 2);
                Falm::REAL g0 =  g(idx, 0);
                Falm::REAL g1 =  g(idx, 1);
                Falm::REAL g2 =  g(idx, 2);
                Falm::REAL j0 = ja(idx);
                fprintf(file, "%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.10e\n", x0, x1, x2, h0, h1, h2, k0, k1, k2, g0, g1, g2, j0);
            }
        }
    }
}

}

#endif