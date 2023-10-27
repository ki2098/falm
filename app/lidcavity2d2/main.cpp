#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <type_traits>
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
// Mapper pdm, global;
REAL maxdiag;
CPMBase cpm;

void field_output(INT i, int rank) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    std::string filename = "data/cavity-rank" + std::to_string(rank) + ".csv" + std::to_string(i);
    FILE *file = fopen(filename.c_str(), "w");
    fprintf(file, "x,y,z,u,v,w,p\n");
    x.sync(MCpType::Dev2Hst);
    u.sync(MCpType::Dev2Hst);
    p.sync(MCpType::Dev2Hst);
    for (INT k = 0; k < pdm.shape.z; k ++) {
        for (INT j = 0; j < pdm.shape.y; j ++) {
            for (INT i = 0; i < pdm.shape.x; i ++) {
                INT idx = IDX(i, j, k, pdm.shape);
                fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", x(idx, 0), x(idx, 1), x(idx, 2), u(idx, 0), u(idx, 1), u(idx, 2), p(idx));
            }
        }
    }
    fclose(file);
}

void plt3d_output(int step, int rank, REAL dt, Vcdm::VCDM<float> &vcdm) {
    Matrix<float> uvw(cpm.pdm_list[cpm.rank].shape, 3, HDCType::Host, "uvw");
    u.sync(MCpType::Dev2Hst);
    // p.sync(MCpType::Dev2Hst);
    // falmMemcpy(&uvw(0, 0), &u(0, 0), sizeof(REAL) * u.size, MCpType::Hst2Hst);
    // falmMemcpy(&uvwp(0, 3), &p(0)   , sizeof(REAL) * p.size, MCpType::Hst2Hst);
    for (INT i = 0; i < u.size; i ++) {
        uvw(i) = u(i);
    }
    vcdm.writeFileData(&uvw(0, 0), cpm.gc, 3, rank, step, Vcdm::IdxType::IJKN);
    dim3 bdim(8, 8, 1);
    double umax = L2Dev_MatColMax(u, 0, cpm, bdim);
    double vmax = L2Dev_MatColMax(u, 1, cpm, bdim);
    double wmax = L2Dev_MatColMax(u, 2, cpm, bdim);
    double _max = L2Dev_VecMax(u, cpm, bdim);
    double umin = L2Dev_MatColMin(u, 0, cpm, bdim);
    double vmin = L2Dev_MatColMin(u, 1, cpm, bdim);
    double wmin = L2Dev_MatColMin(u, 2, cpm, bdim);
    double _min = L2Dev_VecMin(u, cpm, bdim);
    Vcdm::VcdmSlice slice;
    slice.step = step;
    slice.time = step * dt;
    slice.avgStep = 1;
    slice.avgTime = dt;
    slice.avgMode = false;
    slice.vectorMax = _max;
    slice.vectorMin = _min;
    slice.varMax = {umax, vmax, wmax};
    slice.varMin = {umin, vmin, wmin};
    vcdm.timeSlice.push_back(slice);
}

template<typename Type>
void setVcdmAttributes(Vcdm::VCDM<Type> &vcdm) {
    vcdm.setPath("data", "field");
    setVcdm<Type>(cpm, vcdm, {L, L, L/N});
    vcdm.dfiFinfo.varList = {"u", "v", "w"};
}

void allocVars(Region &pdm) {
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

REAL main_loop(L2CFD &cfd, L2EqSolver &eqsolver, CPMBase &cpm, dim3 block_dim, STREAM *stream) {
    cfd.L2Dev_Cartesian3d_FSCalcPseudoU(u, u, uu, ua, nut, kx, g, ja, ff, cpm, block_dim, stream);
    cfd.L2Dev_Cartesian3d_UtoUU(ua, uua, kx, ja, cpm, block_dim, stream);
    forceFaceVelocityZero(uua, cpm);
    cfd.L2Dev_Cartesian3d_MACCalcPoissonRHS(uua, rhs, ja, cpm, block_dim, maxdiag);
    
    eqsolver.L2Dev_Struct3d7p_Solve(poisson_a, p, rhs, res, cpm, block_dim, stream);
    pressureBC(p, cpm, stream);
    copyZ5(p, cpm);

    cfd.L2Dev_Cartesian3d_ProjectP(u, ua, uu, uua, p, kx, g, cpm, block_dim, stream);
    velocityBC(u, cpm, stream);
    copyZ5(u, cpm);

    cfd.L2Dev_Cartesian3d_SGS(u, nut, x, kx, ja, cpm, block_dim, stream);
    copyZ5(nut, cpm);

    cfd.L2Dev_Cartesian3d_Divergence(uu, dvr, ja, cpm, block_dim);

    return L2Dev_EuclideanNormSq(dvr, cpm, block_dim);
}

int main(int argc, char **argv) {
    // std::is_trivially_copyable<Matrix<REAL>> tcp;
    printf("%d\n", std::is_trivially_copyable<Matrix<REAL>>::value);

    CPML2_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    cpm.use_cuda_aware_mpi = true;
    CPML2_GetRank(MPI_COMM_WORLD, mpi_rank);
    CPML2_GetSize(MPI_COMM_WORLD, mpi_size);
    cpm.initPartition(
        {N, N, 1},
        GuideCell,
        mpi_rank,
        mpi_size,
        {atoi(argv[1]), atoi(argv[2]), 1}
    );
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region &global = cpm.global;

    Region ginner(global.shape, cpm.gc);

    printf("rank %d, global size = (%d %d %d), proc size = (%d %d %d), proc offset = (%d %d %d)\n", cpm.rank, global.shape.x, global.shape.y, global.shape.z, pdm.shape.x, pdm.shape.y, pdm.shape.z, pdm.offset.x, pdm.offset.y, pdm.offset.z);

    allocVars(pdm);
    setCoord(L, N, pdm, cpm.gc, x, h, kx, g, ja);
    poisson_a.alloc(pdm.shape, 7, HDCType::Device);
    maxdiag = makePoissonMatrix(poisson_a, g, ja, cpm);

    printf("%lf\n", maxdiag);

    for (int i = 0; i < cpm.size; i ++) {
        if (i == cpm.rank) {
            Matrix<REAL> &a = poisson_a;
            a.sync(MCpType::Dev2Hst);
            printf("rank = %d\n", cpm.rank);
            for (INT i = cpm.gc; i < pdm.shape.x - cpm.gc; i ++) {
                INT idx = IDX(i, cpm.gc, cpm.gc, pdm.shape);
                printf(
                    "%.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
                    a(idx, 0), a(idx, 1), a(idx, 2), a(idx, 3), a(idx, 4), a(idx, 5), a(idx, 6)
                );
            }
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }

    Vcdm::VCDM<float> vcdm;
    // setVcdm<REAL>(cpm, vcdm, Vcdm::doublex3{L, L, L/N});
    setVcdmAttributes(vcdm);
    if (cpm.rank == 0) {
        vcdm.writeIndexDfi();
        vcdm.writeProcDfi();
    }

    if (cpm.rank == 0) {
        Vcdm::double3 d3;
        Vcdm::int3    i3;
        printf("------------dfi info------------\n");
        printf("mpi (%d %d)\n", vcdm.dfiMPI.size, vcdm.dfiMPI.ngrp);
        d3 = vcdm.dfiDomain.globalOrigin;
        printf("gOrigin   (%e %e %e)\n", d3.x, d3.y, d3.z);
        d3 = vcdm.dfiDomain.globalRegion;
        printf("gRegion   (%e %e %e)\n", d3.x, d3.y, d3.z);
        i3 = vcdm.dfiDomain.globalVoxel;
        printf("gVoxel    (%d %d %d)\n", i3.x, i3.y, i3.z);
        i3 = vcdm.dfiDomain.globalDivision;
        printf("gDivision (%d %d %d)\n", i3.x, i3.y, i3.z);
        printf("\n");
        for (int i = 0; i < cpm.size; i ++) {
            Vcdm::VcdmRank &vproc = vcdm.dfiProc[i];
            printf("rank %d\n", vproc.rank);
            printf("host name %s\n", vproc.hostName.c_str());
            i3 = vproc.voxelSize;
            printf("voxel size (%d %d %d)\n", i3.x, i3.y, i3.z);
            i3 = vproc.headIdx;
            printf("head idx   (%d %d %d)\n", i3.x, i3.y, i3.z);
            i3 = vproc.tailIdx;
            printf("tail idx   (%d %d %d)\n", i3.x, i3.y, i3.z);
            printf("\n");
        }
        printf("------------dfi info------------\n");
        fflush(stdout);
    }
    CPML2_Barrier(MPI_COMM_WORLD);

    x.sync(MCpType::Dev2Hst);
    Matrix<float> xyz(x.shape.x, x.shape.y, HDCType::Host, "float x");
    for (INT i = 0; i < x.size; i ++) {
        xyz(i) = x(i);
    }
    vcdm.writeGridData(&xyz(0, 0), cpm.gc, cpm.rank, 0, Vcdm::IdxType::IJKN);
    
    L2CFD cfdsolver(3200, DT, AdvectionSchemeType::Upwind3, SGSType::Empty, 0.1);
    L2EqSolver eqsolver(LSType::PBiCGStab, 10000, 1e-8, 1.2, LSType::SOR, 5, 1.5);

    if (cpm.rank == 0) {
        printf("running on %dx%d grid with Re=%lf until t=%lf\n", N, N, cfdsolver.Re, T);
        fflush(stdout);
    }

    REAL __t = 0;
    INT  __it = 0;
    const INT __IT = int(T / DT);
    const INT __oIT = int(10.0/DT);
    
    plt3d_output(__it, cpm.rank, DT, vcdm);
    field_output(__it, cpm.rank);
    while (__it < __IT) {
        REAL dvr_norm = sqrt(main_loop(cfdsolver, eqsolver, cpm, dim3(8, 8, 1), nullptr)) / ginner.size;
        __t += DT;
        __it ++;
        if (cpm.rank == 0) {
            printf("\r%8d %12.5e, %12.5e, %3d, %12.5e", __it, __t, dvr_norm, eqsolver.it, eqsolver.err);
            fflush(stdout);
        }
        if (__it % __oIT == 0) {
            plt3d_output(__it, cpm.rank, DT, vcdm);
            field_output(__it, cpm.rank);
            if (cpm.rank == 0) {
                REAL probe_u;
                falmMemcpy(&probe_u, &u.dev(IDX(2, 0, 0, cpm.pdm_list[cpm.rank].shape), 0), sizeof(REAL), MCpType::Dev2Hst);
                printf("\n%lf\n", probe_u);
            }
        }
    }
    printf("\n");
    if (cpm.rank == 0) {
        vcdm.writeIndexDfi();
        vcdm.writeProcDfi();
    }
    // field_output(__IT, cpm.rank);

    return CPML2_Finalize();
}