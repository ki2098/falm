#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include "coordinate.h"
#include "output.h"
#include "poisson.h"
#include "boundaryCondition.h"
#include "../../src/FalmCFDDevCall.h"
#include "../../src/FalmEqDevCall.h"

#define L 1.0
#define N 128
#define T 100.0
#define DT 1e-3

const int monitor_i = int(N * 0.01);
const int monitor_j = int(N * 0.5);

using namespace std;
using namespace Falm;
using namespace LidCavity2d;

Matrix<REAL> x, h, kx, g, ja;
Matrix<REAL> u, ua, uc, uu, uua, p, nut, ff, rhs, res, diver, w;
Matrix<REAL> poisson_a;
REAL maxdiag;
CPMBase cpm;

void output(INT i) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    INT    &gc  = cpm.gc;
    std::string filename = "data/lid2d.csv." + std::to_string(i);
    FILE *file = fopen(filename.c_str(), "w");
    fprintf(file, "x,y,z,u,v,w,p\n");
    x.sync(MCpType::Dev2Hst);
    u.sync(MCpType::Dev2Hst);
    p.sync(MCpType::Dev2Hst);
    uu.sync(MCpType::Dev2Hst);
    uua.sync(MCpType::Dev2Hst);
    for (INT k = gc - 1; k < pdm.shape.z - gc + 1; k ++) {
        for (INT j = gc - 1; j < pdm.shape.y - gc + 1; j ++) {
            for (INT i = gc - 1; i < pdm.shape.x - gc + 1; i ++) {
                INT idx = IDX(i, j, k, pdm.shape);
                fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", x(idx, 0), x(idx, 1), x(idx, 2), u(idx, 0), u(idx, 1), u(idx, 2), p(idx));
            }
        }
    }
    fclose(file);
}

void allocVars(Region &pdm) {
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

void pbicgstab(FalmEqDevCall &eq, Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim) {
    Region &global = cpm.global;
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region gmap(global.shape, cpm.gc);
    Region map(pdm.shape, cpm.gc);

    Matrix<REAL> rr(pdm.shape, 1, HDCType::Device, "PBiCGStab rr");
    Matrix<REAL>  p(pdm.shape, 1, HDCType::Device, "PBiCGStab  p");
    Matrix<REAL>  q(pdm.shape, 1, HDCType::Device, "PBiCGStab  q");
    Matrix<REAL>  s(pdm.shape, 1, HDCType::Device, "PBiCGStab  s");
    Matrix<REAL> pp(pdm.shape, 1, HDCType::Device, "PBiCGStab pp");
    Matrix<REAL> ss(pdm.shape, 1, HDCType::Device, "PBiCGStab ss");
    Matrix<REAL>  t(pdm.shape, 1, HDCType::Device, "PBiCGStab  t");
    REAL rho, rrho, alpha, beta, omega;

    eq.Res(a, x, b, r, pdm, map, block_dim);
    eq.err = sqrt(MVDevCall::EuclideanNormSq(r, pdm, map, block_dim)) / gmap.size;

    rr.cpy(r, HDCType::Device);
    rrho  = 1.0;
    alpha = 0.0;
    omega = 1.0;

    it = 0;
    do {
        // if (err < tol) {
        //     break;
        // }

        rho = MV::DotProduct(r, rr, cpm, block_dim);
        if (fabs(rho) < __FLT_MIN__) {
            err = rho;
            break;
        }

        if (it == 0) {
            p.cpy(r, HDCType::Device);
        } else {
            beta = (rho * alpha) / (rrho * omega);
            PBiCGStab1(p, q, r, beta, omega, pdm, map, block_dim);
        }
        pp.clear(HDCType::Device);
        Precondition(a, pp, p, cpm, block_dim);
        MV::MVMult(a, pp, q, cpm, block_dim);
        alpha = rho / MV::DotProduct(rr, q, cpm, block_dim);

        PBiCGStab2(s, q, r, alpha, pdm, map, block_dim);
        ss.clear(HDCType::Device);
        Precondition(a, ss, s, cpm, block_dim);
        MV::MVMult(a, ss, t, cpm, block_dim);
        omega = MV::DotProduct(t, s, cpm, block_dim) / MV::DotProduct(t, t, cpm, block_dim);

        PBiCGStab3(x, pp, ss, alpha, omega, pdm, map, block_dim);
        PBiCGStab4(r, s, t, omega, pdm, map, block_dim);

        rrho = rho;

        err = sqrt(MV::EuclideanNormSq(r, cpm, block_dim)) / gmap.size;
        it ++;
    } while (it < maxit && err > tol);
}

void main_loop(FalmCFDDevCall &cfdsolver, FalmEqDevCall &eqsolver, dim3 block_dim = dim3{8, 8, 1}) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    INT    &gc  = cpm.gc;
    Region  map(pdm.shape, gc);
    Matrix<REAL> un(u.shape.x, u.shape.y, HDCType::Device, "un");
    un.cpy(u, HDCType::Device);

    FalmCFDDevCall rk2fs1(cfdsolver.Re, cfdsolver.dt * 0.5, cfdsolver.AdvScheme, cfdsolver.SGSModel, cfdsolver.CSmagorinsky);
    FalmCFDDevCall rk2fs2(cfdsolver.Re, cfdsolver.dt      , cfdsolver.AdvScheme, cfdsolver.SGSModel, cfdsolver.CSmagorinsky);

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

    rk2fs2.FSPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, pdm, map, block_dim);
    rk2fs2.UtoCU(ua, uc, kx, ja, pdm, map, block_dim);
    rk2fs2.InterpolateCU(uua, uc, pdm, map, block_dim);
    forceFaceVelocityZero(uua, cpm);
    rk2fs2.Divergence(uua, rhs, ja, pdm, map, block_dim);
    MVDevCall::ScaleMatrix(rhs, 1.0 / (DT * maxdiag), block_dim);
    eqsolver.L1Dev_Struct3d7p_Solve(poisson_a, p, rhs, res, cpm, block_dim);
    pressureBC(p, cpm);
    copyZ5(p, cpm);
    rk2fs2.ProjectPGrid(u, ua, p, kx,  pdm, map, block_dim);
    rk2fs2.ProjectPFace(uu, uua, p, g,  pdm, map, block_dim);
    velocityBC(u, cpm);
    copyZ5(u, cpm);
    rk2fs2.SGS(u, nut, x, kx, ja,  pdm, map, block_dim);
    copyZ5(nut, cpm);

    rk2fs2.Divergence(uu, diver, ja,  pdm, map, block_dim);
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
    INT    &gc  = cpm.gc;
    Region  map(pdm.shape, gc);
    setCoord(L, N, cpm, x, h, kx, g, ja);
    printf("%d %d %d\n", pdm.shape.x, pdm.shape.y, pdm.shape.z);

    poisson_a.alloc(pdm.shape, 7, HDCType::Device);

    maxdiag =  makePoissonMatrix(poisson_a, g, ja, cpm);

    printf("%lf\n", maxdiag);

    Matrix<REAL> &a = poisson_a;

    // a.sync(MCpType::Dev2Hst);
    // for (INT i = gc; i < pdm.shape.x - gc; i ++) {
    //     INT idx = IDX(i, gc, gc, pdm.shape);
    //     printf(
    //         "%.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
    //         a(idx, 0), a(idx, 1), a(idx, 2), a(idx, 3), a(idx, 4), a(idx, 5), a(idx, 6)
    //     );
    // }

    FalmCFDDevCall cfdsolver(3200, DT, AdvectionSchemeType::Upwind3, SGSType::Empty, 0.1);
    FalmEqDevCall eqsolver(LSType::PBiCGStab, 1000, 1e-8, 1.2, LSType::SOR, 5, 1.5);

    printf("running on %dx%d grid with Re=%lf until t=%lf\n", N, N, cfdsolver.Re, T);

    FILE *probe = fopen("data/probe.csv", "w");
    fprintf(probe, "t,TKE,u,v\n");

    REAL __t = 0;
    INT  __it = 0;
    const INT __IT = int(T / DT);
    const REAL output_interval = 1.0;
    allocVars(pdm);
    velocityBC(u, cpm);
    pressureBC(p, cpm);
    Region inner(pdm.shape, gc);
    output(__it / INT(output_interval / DT));
    while (__it < __IT) {
        main_loop(cfdsolver, eqsolver);
        REAL tke = sqrt(MVDevCall::EuclideanNormSq(u,  pdm, map, dim3(8, 8, 1))) / inner.size;
        __t += DT;
        __it ++;
        REAL divernorm = sqrt(MVDevCall::EuclideanNormSq(diver,  pdm, map, dim3(8, 8, 1))) / inner.size;

        REAL probeu, probev;
        INT probeidx = IDX(monitor_i + gc, monitor_j + gc, gc, pdm.shape);
        falmMemcpy(&probeu, &u.dev(probeidx, 0), sizeof(REAL), MCpType::Dev2Hst);
        falmMemcpy(&probev, &u.dev(probeidx, 1), sizeof(REAL), MCpType::Dev2Hst);

        printf("\r%8d %12.5e, %12.5e, %3d, %12.5e, %12.5e, %12.5e, %12.5e", __it, __t, divernorm, eqsolver.it, eqsolver.err, tke, probeu, probev);

        fprintf(probe, "%12.5e,%12.10e,%12.10e,%12.10e\n", __t, tke, probeu, probev);
        fflush(stdout);
        if (__it % INT(output_interval / DT) == 0) {
            output(__it / INT(output_interval / DT));
        }
    }
    printf("\n");

    fclose(probe);

    return 0;
}