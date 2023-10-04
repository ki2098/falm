#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "coordinate.h"
#include "output.h"
#include "poisson.h"
#include "boundaryCondition.h"
#include "../../src/FalmCFDL1.h"
#include "../../src/structEqL1.h"

#define L 1.0
#define N 128
#define T 60
#define dt 1e-3

using namespace std;
using namespace Falm;
using namespace LidCavity2d;

Matrix<REAL> x, h, kx, g, ja;
Matrix<REAL> u, ua, uc, uu, uua, p, nut, ff, rhs, res, diver, w;
Matrix<REAL> poisson_a;
Mapper pdm;
REAL maxdiag;

void output() {
    FILE *file = fopen("lid2d.csv", "w");
    fprintf(file, "x,y,z,u,v,w,p\n");
    x.sync(MCpType::Dev2Hst);
    u.sync(MCpType::Dev2Hst);
    p.sync(MCpType::Dev2Hst);
    uu.sync(MCpType::Dev2Hst);
    uua.sync(MCpType::Dev2Hst);
    for (INT k = Gd - 1; k < pdm.shape.z - Gd + 1; k ++) {
        for (INT j = Gd - 1; j < pdm.shape.y - Gd + 1; j ++) {
            for (INT i = Gd - 1; i < pdm.shape.x - Gd + 1; i ++) {
                INT idx = IDX(i, j, k, pdm.shape);
                fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", x(idx, 0), x(idx, 1), x(idx, 2), u(idx, 0), u(idx, 1), u(idx, 2), p(idx));
            }
        }
    }
}

void allocVars(Mapper &pdm) {
    u.alloc(pdm.shape, 3, HDCType::Device);
    ua.alloc(pdm.shape, 3, HDCType::Device);
    uc.alloc(pdm.shape, 3, HDCType::Device);
    uu.alloc(pdm.shape, 3, HDCType::Device);
    uua.alloc(pdm.shape, 3, HDCType::Device);
    p.alloc(pdm.shape, 1, HDCType::Device);
    nut.alloc(pdm.shape, 1, HDCType::Device);
    ff.alloc(pdm.shape, 3, HDCType::Device);
    rhs.alloc(pdm.shape, 1, HDCType::Device);
    res.alloc(pdm.shape, 1, HDCType::Device);
    diver.alloc(pdm.shape, 1, HDCType::Device);

}

void main_loop(L1CFD &cfdsolver, L1EqSolver &eqsolver, dim3 block_dim = dim3{8, 8, 1}) {
    cfdsolver.L1Dev_Cartesian3d_FSCalcPseudoU(u, uu, ua, nut, kx, g, ja, ff, pdm, block_dim);

    cfdsolver.L1Dev_Cartesian3d_UtoCU(ua, uc, kx, ja, pdm, block_dim);

    cfdsolver.L1Dev_Cartesian3d_InterpolateCU(uua, uc, pdm, block_dim);

    forceFaceVelocityZero(uua, pdm);

    cfdsolver.L1Dev_Cartesian3d_MACCalcPoissonRHS(uua, rhs, ja, pdm, block_dim, maxdiag);

    eqsolver.L1Dev_Struct3d7p_Solve(poisson_a, p, rhs, res, pdm, pdm, block_dim);

    pressureBC(p, pdm);

    copyZ5(p, pdm);

    cfdsolver.L1Dev_Cartesian3d_ProjectPGrid(u, ua, p, kx, pdm, block_dim);

    cfdsolver.L1Dev_Cartesian3d_ProjectPFace(uu, uua, p, g, pdm, block_dim);

    velocityBC(u, pdm);

    copyZ5(u, pdm);

    // forceFaceVelocityZero(uu, pdm);

    cfdsolver.L1Dev_Cartesian3d_SGS(u, nut, x, kx, ja, pdm, block_dim);

    copyZ5(nut, pdm);

    cfdsolver.L1Dev_Cartesian3d_Divergence(uu, diver, ja, pdm, block_dim);
}

int main() {
    setCoord(L, N, pdm, x, h, kx, g, ja);
    printf("%d %d %d\n", pdm.shape.x, pdm.shape.y, pdm.shape.z);

    poisson_a.alloc(pdm.shape, 7, HDCType::Device);

    maxdiag =  makePoissonMatrix(poisson_a, g, ja, pdm);

    printf("%lf\n", maxdiag);

    Matrix<REAL> &a = poisson_a;

    a.sync(MCpType::Dev2Hst);
    for (INT i = Gd; i < pdm.shape.x - Gd; i ++) {
        INT idx = IDX(i, Gd, Gd, pdm.shape);
        printf(
            "%.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
            a(idx, 0), a(idx, 1), a(idx, 2), a(idx, 3), a(idx, 4), a(idx, 5), a(idx, 6)
        );
    }

    L1CFD cfdsolver(1000, dt, AdvectionSchemeType::Upwind3, SGSType::Empty, 0.1);
    L1EqSolver eqsolver(LSType::PBiCGStab, 1000, 1e-9, 1.2, LSType::SOR, 5, 1.5);

    REAL __t = 0;
    INT  __it = 0;
    allocVars(pdm);
    velocityBC(u, pdm);
    pressureBC(p, pdm);
    Mapper inner(pdm, Gd);
    while (__t < T) {
        main_loop(cfdsolver, eqsolver);
        __t += dt;
        __it ++;
        REAL divernorm = sqrt(L1Dev_Norm2Sq(diver, pdm, dim3(8, 8, 1))) / inner.size;
        printf("\r%8d %12.5e, %12.5e, %3d, %12.5e", __it, __t, divernorm, eqsolver.it, eqsolver.err);
        fflush(stdout);
    }
    printf("\n");
    output();

    return 0;
}