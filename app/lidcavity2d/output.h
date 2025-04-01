#ifndef _LID_CAVITY2D_OUTPUT_H_
#define _LID_CAVITY2D_OUTPUT_H_

#include <stdio.h>
#include "../../src/matrix.h"
#include "../../src/region.h"
#include "../../src/flag.h"

namespace LidCavity2d {

static void outputGridInfo(
    Falm::Matrix<Falm::Real> &x,
    Falm::Matrix<Falm::Real> &h,
    Falm::Matrix<Falm::Real> &kx,
    Falm::Matrix<Falm::Real> &g,
    Falm::Matrix<Falm::Real> &ja,
    Falm::Region             &pdm
) {
    x.sync(Falm::MCP::Dev2Hst);
    h.sync(Falm::MCP::Dev2Hst);
    kx.sync(Falm::MCP::Dev2Hst);
    g.sync(Falm::MCP::Dev2Hst);
    ja.sync(Falm::MCP::Dev2Hst);
    FILE *file = fopen("grid.csv", "w");
    fprintf(file, "x,y,z,h1,h2,h3,k1,k2,k3,g1,g2,g3,ja\n");
    for (Falm::Int k = 0; k < pdm.shape[2]; k ++) {
        for (Falm::Int j = 0; j < pdm.shape[1]; j ++) {
            for (Falm::Int i = 0; i < pdm.shape[0]; i ++) {
                Falm::Int idx = Falm::IDX(i, j, k, pdm.shape);
                Falm::Real x0 =  x(idx, 0);
                Falm::Real x1 =  x(idx, 1);
                Falm::Real x2 =  x(idx, 2);
                Falm::Real h0 =  h(idx, 0);
                Falm::Real h1 =  h(idx, 1);
                Falm::Real h2 =  h(idx, 2);
                Falm::Real k0 = kx(idx, 0);
                Falm::Real k1 = kx(idx, 1);
                Falm::Real k2 = kx(idx, 2);
                Falm::Real g0 =  g(idx, 0);
                Falm::Real g1 =  g(idx, 1);
                Falm::Real g2 =  g(idx, 2);
                Falm::Real j0 = ja(idx);
                fprintf(file, "%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.10e\n", x0, x1, x2, h0, h1, h2, k0, k1, k2, g0, g1, g2, j0);
            }
        }
    }
}

}

#endif