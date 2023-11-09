#include "../../src/rmcp/alm.h"
#include "../../src/FalmCFD.h"
#include "../../src/FalmEq.h"
#include "../../src/falmath.h"
#include "boundary.h"

#define N 300
#define L 6.0
#define ORGN -3.0
#define DT 5e-3

using namespace Falm;
using namespace TURBINE1;

Matrix<REAL> x, h, kx, g, ja;
Matrix<REAL> u, uprevious, ua, uu, uua, p, nut, ff, rhs, res, dvr, w;
Matrix<REAL> poisson_a;
// Mapper pdm, global;
REAL maxdiag;
CPM cpm;
Vcdm::VCDM<float> vcdm;
STREAM facestream[6];

void plt3d_output(int step, int rank, REAL dt) {
    Matrix<float> uvwp(cpm.pdm_list[cpm.rank].shape, 4, HDCType::Host, "uvwp");
    u.sync(MCpType::Dev2Hst);
    p.sync(MCpType::Dev2Hst);
    for (INT i = 0; i < u.shape[0]; i ++) {
        uvwp(i, 0) = u(i, 0);
        uvwp(i, 1) = u(i, 1);
        uvwp(i, 2) = u(i, 2);
        uvwp(i, 3) = p(i);
    }
    vcdm.writeFileData(&uvwp(0), cpm.gc, 4, rank, step, Vcdm::IdxType::IJKN);
    dim3 bdim(8, 8, 8);
    double umax = FalmMV::MatColMax(u, 0, cpm, bdim);
    double vmax = FalmMV::MatColMax(u, 1, cpm, bdim);
    double wmax = FalmMV::MatColMax(u, 2, cpm, bdim);
    double pmax = FalmMV::MatColMax(p, 0, cpm, bdim);
    double umin = FalmMV::MatColMin(u, 0, cpm, bdim);
    double vmin = FalmMV::MatColMin(u, 1, cpm, bdim);
    double wmin = FalmMV::MatColMin(u, 2, cpm, bdim);
    double pmin = FalmMV::MatColMin(p, 0, cpm, bdim);
    Vcdm::VcdmSlice slice;
    slice.step = step;
    slice.time = step * dt;
    slice.avgStep = 1;
    slice.avgTime = dt;
    slice.avgMode = true;
    slice.varMax = {umax, vmax, wmax, pmax};
    slice.varMin = {umin, vmin, wmin, pmin};
    vcdm.timeSlice.push_back(slice);
}

REAL main_loop(FalmCFD &cfd, FalmEq &eq, RmcpAlm &alm, RmcpWindfarm &windfarm, INT step, REAL dt, dim3 block_dim, STREAM *stream) {
    uprevious.cpy(u, HDCType::Device);
    alm.SetALMFlag(x, step * dt, windfarm, cpm, block_dim);
    alm.ALM(u, x, ff, step * dt, windfarm, cpm, block_dim);

    cfd.FSPseudoU(uprevious, u, uu, ua, nut, kx, g, ja, ff, cpm, block_dim, stream);
    cfd.UtoUU(ua, uua, kx, ja, cpm, block_dim, stream);
    // forceFaceVelocityZero(uua, cpm);
    cfd.MACCalcPoissonRHS(uua, rhs, ja, cpm, block_dim, maxdiag);
    
    eq.Solve(poisson_a, p, rhs, res, cpm, block_dim, stream);
    pressure_bc(p, cpm, stream);
    // copyZ5(p, cpm);

    cfd.ProjectP(u, ua, uu, uua, p, kx, g, cpm, block_dim, stream);
    velocity_bc(u, uprevious, x, dt, cpm, stream);
    // copyZ5(u, cpm);

    cfd.SGS(u, nut, x, kx, ja, cpm, block_dim, stream);
    // copyZ5(nut, cpm);

    cfd.Divergence(uu, dvr, ja, cpm, block_dim);

    return FalmMV::EuclideanNormSq(dvr, cpm, block_dim);
}

int main(int argc, char **argv) {
    CPM_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    cpm.use_cuda_aware_mpi = true;
    CPM_GetRank(MPI_COMM_WORLD, mpi_rank);
    CPM_GetSize(MPI_COMM_WORLD, mpi_size);
    assert(mpi_size == 2);
    cpm.initPartition(
        {N, N, N},
        GuideCell,
        mpi_rank,
        mpi_size,
        {1, 1, 2}
    );
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region &global = cpm.global;
    Region ginner(global.shape, cpm.gc);
    RmcpAlm rmcp(cpm);
    FalmCFD cfdsolver(10000, DT, AdvectionSchemeType::Upwind3, SGSType::Empty, 0.1);
    FalmEq eqsolver(LSType::PBiCGStab, 1000, 1e-8, 1.2, LSType::SOR, 5, 1.5);
    for (INT fid = 0; fid < CPM::NFACE; fid ++) {
        cudaStreamCreate(&facestream[fid]);
    }
    
    setVcdm<float>(cpm, vcdm, {L, L, L}, {ORGN, ORGN, ORGN});
    vcdm.setPath("data", "flag");
    vcdm.dfiFinfo.varList = {"u", "v", "w", "p"};
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
        printf("gOrigin   (%e %e %e)\n", d3[0], d3[1], d3[2]);
        d3 = vcdm.dfiDomain.globalRegion;
        printf("gRegion   (%e %e %e)\n", d3[0], d3[1], d3[2]);
        i3 = vcdm.dfiDomain.globalVoxel;
        printf("gVoxel    (%d %d %d)\n", i3[0], i3[1], i3[2]);
        i3 = vcdm.dfiDomain.globalDivision;
        printf("gDivision (%d %d %d)\n", i3[0], i3[1], i3[2]);
        printf("\n");
        for (int i = 0; i < cpm.size; i ++) {
            Vcdm::VcdmRank &vproc = vcdm.dfiProc[i];
            printf("rank %d\n", vproc.rank);
            printf("host name %s\n", vproc.hostName.c_str());
            i3 = vproc.voxelSize;
            printf("voxel size (%d %d %d)\n", i3[0], i3[1], i3[2]);
            i3 = vproc.headIdx;
            printf("head idx   (%d %d %d)\n", i3[0], i3[1], i3[2]);
            i3 = vproc.tailIdx;
            printf("tail idx   (%d %d %d)\n", i3[0], i3[1], i3[2]);
            printf("\n");
        }
        printf("------------dfi info------------\n");
        fflush(stdout);
    }
    CPM_Barrier(MPI_COMM_WORLD);

    u.alloc(pdm.shape, 3, HDCType::Device);
    uprevious.alloc(pdm.shape, 3, HDCType::Device);
    ua.alloc(pdm.shape, 3, HDCType::Device);
    uu.alloc(pdm.shape, 3, HDCType::Device);
    uua.alloc(pdm.shape, 3, HDCType::Device);
    p.alloc(pdm.shape, 1, HDCType::Device);
    nut.alloc(pdm.shape, 1, HDCType::Device);
    ff.alloc(pdm.shape, 3, HDCType::Device);
    rhs.alloc(pdm.shape, 1, HDCType::Device);
    res.alloc(pdm.shape, 1, HDCType::Device);
    dvr.alloc(pdm.shape, 1, HDCType::Device);

    Matrix<float> xyz(pdm.shape, 3, HDCType::Host);

    const REAL pitch = L / N;
    const REAL volume = pitch * pitch * pitch;
    const REAL dkdx  = 1.0 / pitch;
    for (INT k = 0; k < pdm.shape[2]; k ++) {
    for (INT j = 0; j < pdm.shape[1]; j ++) {
    for (INT i = 0; i < pdm.shape[0]; i ++) {
        INT idx = IDX(i, j, k, pdm.shape);
        x(idx, 0) = ORGN + (i + pdm.offset[0] - cpm.gc + 0.5) * pitch;
        x(idx, 1) = ORGN + (j + pdm.offset[1] - cpm.gc + 0.5) * pitch;
        x(idx, 2) = ORGN + (k + pdm.offset[2] - cpm.gc + 0.5) * pitch;
        h(idx, 0) = h(idx, 1) = h(idx, 2) = pitch;
        kx(idx, 0) = kx(idx, 1) = kx(idx, 2) = dkdx;
        ja(idx) = volume;
        g(idx, 0) = g(idx, 1) = g(idx, 2) = volume * (dkdx * dkdx);

        xyz(idx, 0) = ORGN + (i + pdm.offset[0] - cpm.gc + 0.5) * (L / N);
        xyz(idx, 1) = ORGN + (j + pdm.offset[1] - cpm.gc + 0.5) * (L / N);
        xyz(idx, 2) = ORGN + (k + pdm.offset[2] - cpm.gc + 0.5) * (L / N);
    }}}
    x.sync(MCpType::Hst2Dev);
    h.sync(MCpType::Hst2Dev);
    kx.sync(MCpType::Hst2Dev);
    ja.sync(MCpType::Hst2Dev);
    g.sync(MCpType::Hst2Dev);
    vcdm.writeGridData(&xyz(0), cpm.gc, cpm.rank, 0, Vcdm::IdxType::IJKN);

    for (INT fid = 0; fid < CPM::NFACE; fid ++) {
        cudaStreamDestroy(facestream[fid]);
    }

    CPM_Finalize();
}