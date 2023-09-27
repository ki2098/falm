#include <stdio.h>
#include "../src/structEqL1.h"
#include "../src/MVL1.h"

using namespace Falm;

#define Nx 100
#define Ny 1
#define Nz 1
#define Lx 1.0
#define TW 0.0
#define TE 100.0

int main() {
    Mapper global(
        INTx3{Nx + 2 * Gd, Ny + 2 * Gd, Nz + 2 * Gd},
        INTx3{0, 0, 0}
    );
    Mapper pdom(global.shape, global.offset);
    Matrix<double> a, t, b, r;
    a.alloc(pdom.shape, 7, HDCType::Host  , 0);
    t.alloc(pdom.shape, 1, HDCType::Device, 1);
    b.alloc(pdom.shape, 1, HDCType::Host  , 2);
    r.alloc(pdom.shape, 1, HDCType::Device, 3);
    const double dx = Lx / Nx;
    for (unsigned int i = Gd; i < Gd + Nx; i ++) {
        double ac, ae, aw, bc;
        if (i == Gd) {
            ae = 1.0 / (dx * dx);
            aw = 0.0;
            ac = - (ae + 2.0 / (dx * dx));
            bc = - (2 * TW) / (dx * dx);
        } else if (i == Gd + Nx - 1) {
            ae = 0.0;
            aw = 1.0 / (dx * dx);
            ac = - (aw + 2.0 / (dx * dx));
            bc = - (2 * TE) / (dx * dx);
        } else {
            ae = 1.0 / (dx * dx);
            aw = 1.0 / (dx * dx);
            ac = - (ae + aw);
            bc = 0.0;
        }
        unsigned int idx = IDX(i, Gd, Gd, pdom.shape);
        a(idx, 0) = ac;
        a(idx, 1) = ae;
        a(idx, 2) = aw;
        b(idx)    = bc;
    }
    a.sync(MCpType::Hst2Dev);
    b.sync(MCpType::Hst2Dev);
    dim3 block_dim(32, 1, 1);
    double max_diag = L1Dev_MaxDiag(a, pdom, block_dim);
    printf("%12lf\n", max_diag);
    L1Dev_ScaleMatrix(a, 1.0 / max_diag, block_dim);
    L1Dev_ScaleMatrix(b, 1.0 / max_diag, block_dim);
    a.sync(MCpType::Dev2Hst);
    b.sync(MCpType::Dev2Hst);

    for (unsigned int i = Gd; i < Gd + Nx; i ++) {
        unsigned int idx = IDX(i, Gd, Gd, pdom.shape);
        for (unsigned int j = 0; j < 7; j ++) {
            printf("%12lf ", a(idx, j));
        }
        printf("= %12lf\n", b(idx));
    }

    L1EqSolver solver(LSType::PBiCGStab, 10000, 1e-9, 1.5, LSType::Jacobi, 5, 1.0);
    solver.L1Dev_Struct3d7p_Solve(a, t, b, r, global, pdom, block_dim);
    t.sync(MCpType::Dev2Hst);
    r.sync(MCpType::Dev2Hst);
    for (unsigned int i = Gd; i < Gd + Nx; i ++) {
        unsigned int idx = IDX(i, Gd, Gd, pdom.shape);
        printf("%12.4lf %12.4lf\n", t(idx), r(idx));
    }
    printf("%d %.12lf\n", solver.it, solver.err);

    return 0;
}
