#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include "coordinate.h"
#include "output.h"
#include "poisson.h"
#include "boundaryCondition.h"
#include "../../src/FalmCFD.h"
#include "../../src/FalmEq.h"

#define L 1.0
#define N 128
#define T 100.0
#define DT 1e-3

const int monitor_i = int(N * 0.01);
const int monitor_j = int(N * 0.5);

using namespace std;
using namespace Falm;
using namespace LidCavity2d;

Matrix<Real> x, h, kx, g, ja;
Matrix<Real> u, ua, uc, uu, uua, p, nut, ff, rhs, res, diver, w;
Matrix<Real> poisson_a;
Real maxdiag;
CPM cpm;

void output(Int i) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Int    &gc  = cpm.gc;
    std::string filename = "data/lid2d.csv." + std::to_string(i);
    FILE *file = fopen(filename.c_str(), "w");
    fprintf(file, "x,y,z,u,v,w,p\n");
    x.sync(MCP::Dev2Hst);
    u.sync(MCP::Dev2Hst);
    p.sync(MCP::Dev2Hst);
    uu.sync(MCP::Dev2Hst);
    uua.sync(MCP::Dev2Hst);
    for (Int k = gc - 1; k < pdm.shape[2] - gc + 1; k ++) {
        for (Int j = gc - 1; j < pdm.shape[1] - gc + 1; j ++) {
            for (Int i = gc - 1; i < pdm.shape[0] - gc + 1; i ++) {
                Int idx = IDX(i, j, k, pdm.shape);
                fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", x(idx, 0), x(idx, 1), x(idx, 2), u(idx, 0), u(idx, 1), u(idx, 2), p(idx));
            }
        }
    }
    fclose(file);
}

void allocVars(Region &pdm) {
    u.alloc(pdm.shape, 3, HDC::Device);
    ua.alloc(pdm.shape, 3, HDC::Device);
    uc.alloc(pdm.shape, 3, HDC::Device);
    uu.alloc(pdm.shape, 3, HDC::Device);
    uua.alloc(pdm.shape, 3, HDC::Device);
    p.alloc(pdm.shape, 1, HDC::Device);
    nut.alloc(pdm.shape, 1, HDC::Device);
    ff.alloc(pdm.shape, 3, HDC::Device);
    rhs.alloc(pdm.shape, 1, HDC::Device);
    res.alloc(pdm.shape, 1, HDC::Device);
    diver.alloc(pdm.shape, 1, HDC::Device);

}

void main_loop(FalmCFD &cfdsolver, FalmEq &eqsolver, dim3 block_dim = dim3{8, 8, 1}) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Int    &gc  = cpm.gc;
    Region  map(pdm.shape, gc);
    Matrix<Real> un(u.shape[0], u.shape[1], HDC::Device, "un");
    un.copy(u, HDC::Device);

    FalmCFD rk2fs1(cfdsolver.Re, cfdsolver.dt * 0.5, cfdsolver.AdvScheme, cfdsolver.SGSModel, cfdsolver.CSmagorinsky);
    FalmCFD rk2fs2(cfdsolver.Re, cfdsolver.dt      , cfdsolver.AdvScheme, cfdsolver.SGSModel, cfdsolver.CSmagorinsky);

    // rk2fs1.L1Dev_Cartesian3d_FSCalcPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, pdm, block_dim);
    // rk2fs1.L1Dev_Cartesian3d_UtoCU(ua, uc, kx, ja, pdm, block_dim);
    // rk2fs1.L1Dev_Cartesian3d_InterpolateCU(uua, uc, pdm, block_dim);
    // forceFaceVelocityZero(uua, pdm);
    // rk2fs1.L1Dev_Cartesian3d_MACCalcPoissonRHS(uua, rhs, ja, pdm, block_dim, maxdiag);
    // eqsolver.L1Dev_Struct3d7p_Solve(poisson_a, p, rhs, res, pdm, pdm, block_dim);
    // pressureBC(p, pdm);
    // copyZ5(p, pdm);
    // rk2fs1.L1Dev_Cartesian3d_ProjectPGrid(u, ua, p, kx, pdm, block_dim);
    // rk2fs1.L1Dev_Cartesian3d_ProjectPFace(uu, uua, p, g, pdm, block_dim);
    // velocityBC(u, pdm);
    // copyZ5(u, pdm);
    // rk2fs1.L1Dev_Cartesian3d_SGS(u, nut, x, kx, ja, pdm, block_dim);
    // copyZ5(nut, pdm);

    rk2fs2.FSPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, cpm, block_dim);
    rk2fs2.UtoUU(ua, uua, kx, ja, cpm, block_dim);
    forceFaceVelocityZero(uua, cpm);
    rk2fs2.Divergence(uua, rhs, ja, cpm, block_dim);
    FalmMVDevCall::ScaleMatrix(rhs, 1.0 / (DT * maxdiag), block_dim);
    eqsolver.Solve(poisson_a, p, rhs, res, cpm, block_dim);
    pressureBC(p, cpm);
    copyZ5(p, cpm);
    rk2fs2.ProjectP(u, ua, uu, uua, p, kx, g, cpm, block_dim);
    velocityBC(u, cpm);
    copyZ5(u, cpm);
    rk2fs2.SGS(u, nut, x, kx, ja,  cpm, block_dim);
    copyZ5(nut, cpm);

    rk2fs2.Divergence(uu, diver, ja, cpm, block_dim);
    // cfdsolver.L1Dev_Cartesian3d_FSCalcPseudoU(u, u, uu, ua, nut, kx, g, ja, ff, pdm, block_dim);

    // cfdsolver.L1Dev_Cartesian3d_UtoCU(ua, uc, kx, ja, pdm, block_dim);

    // cfdsolver.L1Dev_Cartesian3d_InterpolateCU(uua, uc, pdm, block_dim);

    // forceFaceVelocityZero(uua, pdm);

    // cfdsolver.L1Dev_Cartesian3d_MACCalcPoissonRHS(uua, rhs, ja, pdm, block_dim, maxdiag);

    // eqsolver.L1Dev_Struct3d7p_Solve(poisson_a, p, rhs, res, pdm, pdm, block_dim);

    // pressureBC(p, pdm);

    // copyZ5(p, pdm);

    // cfdsolver.L1Dev_Cartesian3d_ProjectPGrid(u, ua, p, kx, pdm, block_dim);

    // cfdsolver.L1Dev_Cartesian3d_ProjectPFace(uu, uua, p, g, pdm, block_dim);

    // velocityBC(u, pdm);

    // copyZ5(u, pdm);

    // // forceFaceVelocityZero(uu, pdm);

    // cfdsolver.L1Dev_Cartesian3d_SGS(u, nut, x, kx, ja, pdm, block_dim);

    // copyZ5(nut, pdm);

    // cfdsolver.L1Dev_Cartesian3d_Divergence(uu, diver, ja, pdm, block_dim);
}

int main() {
    cpm.initPartition({N, N, 1}, GuideCell);
    Region &pdm = cpm.pdm_list[cpm.rank];
    Int    &gc  = cpm.gc;
    Region  map(pdm.shape, gc);
    setCoord(L, N, cpm, x, h, kx, g, ja);
    printf("%d %d %d\n", pdm.shape[0], pdm.shape[1], pdm.shape[2]);

    poisson_a.alloc(pdm.shape, 7, HDC::Device);

    maxdiag =  makePoissonMatrix(poisson_a, g, ja, cpm);

    printf("%lf\n", maxdiag);

    Matrix<Real> &a = poisson_a;

    // a.sync(MCpType::Dev2Hst);
    // for (INT i = gc; i < pdm.shape[0] - gc; i ++) {
    //     INT idx = IDX(i, gc, gc, pdm.shape);
    //     printf(
    //         "%.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
    //         a(idx, 0), a(idx, 1), a(idx, 2), a(idx, 3), a(idx, 4), a(idx, 5), a(idx, 6)
    //     );
    // }

    FalmCFD cfdsolver(3200, DT, AdvectionSchemeType::Upwind3, SGSType::Empty, 0.1);
    FalmEq eqsolver(LSType::PBiCGStab, 1000, 1e-8, 1.2, LSType::SOR, 5, 1.5);

    printf("running on %dx%d grid with Re=%lf until t=%lf\n", N, N, cfdsolver.Re, T);

    FILE *probe = fopen("data/probe.csv", "w");
    fprintf(probe, "t,TKE,u,v\n");

    Real __t = 0;
    Int  __it = 0;
    const Int __IT = int(T / DT);
    const Real output_interval = 1.0;
    allocVars(pdm);
    velocityBC(u, cpm);
    pressureBC(p, cpm);
    Region inner(pdm.shape, gc);
    output(__it / Int(output_interval / DT));
    while (__it < __IT) {
        main_loop(cfdsolver, eqsolver);
        Real tke = sqrt(FalmMVDevCall::EuclideanNormSq(u,  pdm, map, dim3(8, 8, 1))) / inner.size;
        __t += DT;
        __it ++;
        Real divernorm = sqrt(FalmMVDevCall::EuclideanNormSq(diver,  pdm, map, dim3(8, 8, 1))) / inner.size;

        Real probeu, probev;
        Int probeidx = IDX(monitor_i + gc, monitor_j + gc, gc, pdm.shape);
        falmMemcpy(&probeu, &u.dev(probeidx, 0), sizeof(Real), MCP::Dev2Hst);
        falmMemcpy(&probev, &u.dev(probeidx, 1), sizeof(Real), MCP::Dev2Hst);

        printf("\r%8d %12.5e, %12.5e, %3d, %12.5e, %12.5e, %12.5e, %12.5e", __it, __t, divernorm, eqsolver.it, eqsolver.err, tke, probeu, probev);

        fprintf(probe, "%12.5e,%12.10e,%12.10e,%12.10e\n", __t, tke, probeu, probev);
        fflush(stdout);
        if (__it % Int(output_interval / DT) == 0) {
            output(__it / Int(output_interval / DT));
        }
    }
    printf("\n");

    fclose(probe);

    return 0;
}