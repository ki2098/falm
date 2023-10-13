#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include "coordinate.h"
#include "poisson.h"
#include "boundary.h"
#include "partition.h"
#include "../../src/FalmCFDL2.h"
#include "../../src/structEqL2.h"

#define L 1.0
#define N 128
#define T 100.0
#define DT 1e-3

const int monitor_i = int(N * 0.01);
const int monitor_j = int(N * 0.5);

using namespace std;
using namespace Falm;
using namespace LidCavity2d2;

Matrix<REAL> x, h, kx, g, ja;
Matrix<REAL> u, ua, uu, uua, p, nut, ff, rhs, res, dvr, w;
Matrix<REAL> poisson_a;
Mapper pdm, global;
REAL maxdiag;
CPMBase cpm;

void field_output(INT i, int rank) {
    std::string filename = "data/cavity-rank" + std::to_string(rank) + ".csv" + std::to_string(i);
    FILE *file = fopen(filename.c_str(), "w");
    fprintf(file, "x,y,z,u,v,w,p\n");
    x.sync(MCpType::Dev2Hst);
    u.sync(MCpType::Dev2Hst);
    p.sync(MCpType::Dev2Hst);
    for (INT k = Gd; k < pdm.shape.z - Gd; k ++) {
        for (INT j = Gd; j < pdm.shape.y - Gd; j ++) {
            for (INT i = Gd; i < pdm.shape.x - Gd; i ++) {
                INT idx = IDX(i, j, k, pdm.shape);
                fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", x(idx, 0), x(idx, 1), x(idx, 2), u(idx, 0), u(idx, 1), u(idx, 2), p(idx));
            }
        }
    }
    fclose(file);
}

void allocVars(Mapper &pdm) {
    u.alloc(pdm.shape, 3, HDCType::Device);
    ua.alloc(pdm.shape, 3, HDCType::Device);
    uu.alloc(pdm.shape, 3, HDCType::Device);
    uua.alloc(pdm.shape, 3, HDCType::Device);
    p.alloc(pdm.shape, 1, HDCType::Device);
    nut.alloc(pdm.shape, 1, HDCType::Device);
    ff.alloc(pdm.shape, 3, HDCType::Device);
    rhs.alloc(pdm.shape, 1, HDCType::Device);
    res.alloc(pdm.shape, 1, HDCType::Device);
    dvr.alloc(pdm.shape, 1, HDCType::Device);
}

REAL main_loop(L2CFD &cfd, L2EqSolver &eqsolver, CPMBase cpm, dim3 block_dim, STREAM *stream) {
    cfd.L2Dev_Cartesian3d_FSCalcPseudoU(u, u, uu, ua, nut, kx, g, ja, ff, pdm, block_dim, cpm, stream);
    cfd.L2Dev_Cartesian3d_UtoUU(ua, uua, kx, ja, pdm, block_dim, cpm, stream);
    forceFaceVelocityZero(uua, global, pdm, cpm);
    cfd.L2Dev_Cartesian3d_MACCalcPoissonRHS(uua, rhs, ja, pdm, block_dim, maxdiag);
    
    eqsolver.L2Dev_Struct3d7p_Solve(poisson_a, p, rhs, res, global, pdm, block_dim, cpm, stream);
    pressureBC(p, global, pdm, cpm, stream);
    copyZ5(p, pdm);

    cfd.L2Dev_Cartesian3d_ProjectP(u, ua, uu, uua, p, kx, g, pdm, block_dim, cpm, stream);
    velocityBC(u, global, pdm, cpm, stream);
    copyZ5(u, pdm);

    cfd.L2Dev_Cartesian3d_SGS(u, nut, x, kx, ja, pdm, block_dim, cpm, stream);
    copyZ5(nut, pdm, stream);

    cfd.L2Dev_Cartesian3d_Divergence(uu, dvr, ja, pdm, block_dim);

    return L2Dev_EuclideanNormSq(dvr, pdm, block_dim, cpm);
}

int main(int argc, char **argv) {
    CPML2_Init(&argc, &argv);
    cpm.use_cuda_aware_mpi = true;
    CPML2_GetRank(MPI_COMM_WORLD, cpm.rank);
    CPML2_GetSize(MPI_COMM_WORLD, cpm.size);
    cpm.shape = INTx3{atoi(argv[1]), atoi(argv[2]), 1};
    cpm.initNeighbour();
    setPartition(L, N, global, pdm, cpm);

    Mapper ginner(global, Gd);

    printf("rank %d, global size = (%d %d %d), proc size = (%d %d %d), proc offset = (%d %d %d)\n", cpm.rank, global.shape.x, global.shape.y, global.shape.z, pdm.shape.x, pdm.shape.y, pdm.shape.z, pdm.offset.x, pdm.offset.y, pdm.offset.z);

    allocVars(pdm);
    setCoord(L, N, pdm, x, h, kx, g, ja);
    poisson_a.alloc(pdm.shape, 7, HDCType::Device);
    maxdiag = makePoissonMatrix(poisson_a, g, ja, global, pdm, cpm);

    printf("%lf\n", maxdiag);

    for (int i = 0; i < cpm.size; i ++) {
        if (i == cpm.rank) {
            Matrix<REAL> &a = poisson_a;
            a.sync(MCpType::Dev2Hst);
            printf("rank = %d\n", cpm.rank);
            for (INT i = Gd; i < pdm.shape.x - Gd; i ++) {
                INT idx = IDX(i, Gd, Gd, pdm.shape);
                printf(
                    "%.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
                    a(idx, 0), a(idx, 1), a(idx, 2), a(idx, 3), a(idx, 4), a(idx, 5), a(idx, 6)
                );
            }
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }
    
    L2CFD cfdsolver(3200, DT, AdvectionSchemeType::Upwind3, SGSType::Empty, 0.1);
    L2EqSolver eqsolver(LSType::PBiCGStab, 10000, 1e-8, 1.2, LSType::SOR, 5, 1.5);

    if (cpm.rank == 0) {
        printf("running on %dx%d grid with Re=%lf until t=%lf\n", N, N, cfdsolver.Re, T);
        fflush(stdout);
    }

    REAL __t = 0;
    INT  __it = 0;
    const INT __IT = int(T / DT);
    while (__it < __IT) {
        REAL dvr_norm = sqrt(main_loop(cfdsolver, eqsolver, cpm, dim3(8, 8, 1), nullptr)) / ginner.size;
        __t += DT;
        __it ++;
        if (cpm.rank == 0) {
            printf("\r%8d %12.5e, %12.5e, %3d, %12.5e", __it, __t, dvr_norm, eqsolver.it, eqsolver.err);
            fflush(stdout);
        }
    }

    field_output(__IT, cpm.rank);

    return CPML2_Finalize();
}