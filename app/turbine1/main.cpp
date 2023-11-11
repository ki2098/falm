#include <math.h>
#include "../../src/FalmCFD.h"
#include "../../src/FalmEq.h"
#include "../../src/rmcp/alm.h"
#include "../../src/falmath.h"
#include "bc.h"

using namespace Falm;

const dim3 block(8, 8, 8);
const REAL3 Lxyz{{24.0, 8.0, 8.0}};
const INT3  Nxyz{{480, 160, 160}};
const REAL3 origin{{-4,-4,-4}};

const REAL endtime = 100;
const REAL dt = 5e-3;

Matrix<REAL> x, h, kx, g, ja;
Matrix<REAL> u, uprev, uu, ua, uua, p, nut, ff ,rhs, res, dvr;
Matrix<REAL> poisson_a;
REAL maxdiag;
CPM cpm;
Vcdm::VCDM<float> vcdm;
STREAM facestream[CPM::NFACE];

void plt3d_output(int step, int rank, double dt) {
    Matrix<float> uvw(cpm.pdm_list[cpm.rank].shape, 6, HDCType::Host, "uvw");
    u.sync(MCpType::Dev2Hst);
    p.sync(MCpType::Dev2Hst);
    nut.sync(MCpType::Dev2Hst);
    dvr.sync(MCpType::Dev2Hst);
    for (INT i = 0; i < u.shape[0]; i ++) {
        uvw(i, 0) = u(i, 0);
        uvw(i, 1) = u(i, 1);
        uvw(i, 2) = u(i, 2);
        uvw(i, 3) = p(i);
        uvw(i, 4) = nut(i);
        uvw(i, 5) = dvr(i);
    }
    vcdm.writeFileData(&uvw(0, 0), cpm.gc, 6, rank, step, Vcdm::IdxType::IJKN);
    double umax = FalmMV::MatColMax(u, 0, cpm, block);
    double vmax = FalmMV::MatColMax(u, 1, cpm, block);
    double wmax = FalmMV::MatColMax(u, 2, cpm, block);
    double pmax = FalmMV::MatColMax(p, 0, cpm, block);
    double nmax = FalmMV::MatColMax(nut, 0, cpm, block);
    double dmax = FalmMV::MatColMax(dvr, 0, cpm, block);
    double umin = FalmMV::MatColMin(u, 0, cpm, block);
    double vmin = FalmMV::MatColMin(u, 1, cpm, block);
    double wmin = FalmMV::MatColMin(u, 2, cpm, block);
    double pmin = FalmMV::MatColMin(p, 0, cpm, block);
    double nmin = FalmMV::MatColMin(nut, 0, cpm, block);
    double dmin = FalmMV::MatColMin(dvr, 0, cpm, block);
    Vcdm::VcdmSlice slice;
    slice.step = step;
    slice.time = step * dt;
    slice.avgStep = 1;
    slice.avgTime = dt;
    slice.avgMode = true;
    // slice.vectorMax = _max;
    // slice.vectorMin = _min;
    slice.varMax = {umax, vmax, wmax, pmax, nmax, dmax};
    slice.varMin = {umin, vmin, wmin, pmin, nmin, dmin};
    vcdm.timeSlice.push_back(slice);
}

REAL main_loop(FalmCFD &cfd, FalmEq &eq, RmcpAlm &alm, RmcpTurbineArray &turbineArray, INT step, REAL dt, STREAM *s) {
    uprev.copy(u, HDCType::Device);
    alm.SetALMFlag(x, dt * step, turbineArray, cpm, block);
    alm.ALM(u, x, ff, dt * step, turbineArray, cpm, block);

    cfd.FSPseudoU(uprev, u, uu, u, nut, kx, g, ja, ff, cpm, block, s);
    cfd.UtoUU(u, uu, kx, ja, cpm, block, s);
    cfd.MACCalcPoissonRHS(uu, rhs, ja, cpm, block, maxdiag);
    
    eq.Solve(poisson_a, p, rhs, res, cpm, block, s);
    TURBINE1::pbc(p, cpm, s);

    cfd.ProjectP(u, u, uu, uu, p, kx, g, cpm, block, s);
    TURBINE1::ubc(u, uprev, x, dt, cpm, s);
    // printf("6\n"); fflush(stdout);

    cfd.SGS(u, nut, x, kx, ja, cpm, block, s);
    // printf("7\n"); fflush(stdout);

    cfd.Divergence(uu, dvr, ja, cpm, block);

    return FalmMV::EuclideanNormSq(dvr, cpm, block);
}

int main(int argc, char **argv) {
    CPM_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    cpm.use_cuda_aware_mpi = true;
    CPM_GetRank(MPI_COMM_WORLD, mpi_rank);
    CPM_GetSize(MPI_COMM_WORLD, mpi_size);
    cpm.initPartition(Nxyz - INT3{{1,1,1}}, GuideCell, mpi_rank, mpi_size, {{mpi_size, 1, 1}});
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region &global = cpm.global;
    Region ginner(global.shape, cpm.gc);
    printf("rank %d, global size = (%d %d %d), proc size = (%d %d %d), proc offset = (%d %d %d)\n", cpm.rank, global.shape[0], global.shape[1], global.shape[2], pdm.shape[0], pdm.shape[1], pdm.shape[2], pdm.offset[0], pdm.offset[1], pdm.offset[2]);
    fflush(stdout); CPM_Barrier(MPI_COMM_WORLD);
    vcdm.setPath("data", "lid3d");
    setVcdm(cpm, vcdm, {{Lxyz[0],Lxyz[1],Lxyz[2]}}, {{origin[0], origin[1], origin[2]}});
    vcdm.dfiFinfo.varList = {"u", "v", "w", "p", "nut", "divergence"};
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
        vcdm.writeIndexDfi();
        vcdm.writeProcDfi();
    }
    CPM_Barrier(MPI_COMM_WORLD);

    for (int fid = 0; fid < 6; fid ++) {
        cudaStreamCreate(&facestream[fid]);
    }

    u.alloc(pdm.shape, 3, HDCType::HstDev);
    uprev.alloc(pdm.shape, 3, HDCType::Device);
    ua.alloc(pdm.shape, 3, HDCType::Device);
    uu.alloc(pdm.shape, 3, HDCType::Device);
    uua.alloc(pdm.shape, 3, HDCType::Device);
    p.alloc(pdm.shape, 1, HDCType::Device);
    nut.alloc(pdm.shape, 1, HDCType::Device);
    ff.alloc(pdm.shape, 3, HDCType::Device);
    rhs.alloc(pdm.shape, 1, HDCType::Device);
    res.alloc(pdm.shape, 1, HDCType::Device);
    dvr.alloc(pdm.shape, 1, HDCType::Device);
    for (INT id = 0; id < u.shape[0]; id ++) {
        u(id, 0) = 1.0;
        u(id, 1) = u(id, 2) = 0.0;
    }
    u.sync(MCpType::Hst2Dev);

    Matrix<float> xyz(pdm.shape, 3, HDCType::Host);
    x.alloc(pdm.shape, 3, HDCType::Host);
    h.alloc(pdm.shape, 3, HDCType::Host);
    kx.alloc(pdm.shape, 3, HDCType::Host);
    ja.alloc(pdm.shape, 1, HDCType::Host);
    g.alloc(pdm.shape, 3, HDCType::Host);
    const REAL3 pitch = {{Lxyz[0]/Nxyz[0], Lxyz[1]/Nxyz[1], Lxyz[2]/Nxyz[2]}};
    const REAL volume = pitch[0] * pitch[1] * pitch[2];
    const REAL3 dkdx  = {{1.0/pitch[0], 1.0/pitch[1], 1.0/pitch[2]}};
    for (INT k = 0; k < pdm.shape[2]; k ++) {
    for (INT j = 0; j < pdm.shape[1]; j ++) {
    for (INT i = 0; i < pdm.shape[0]; i ++) {
        INT idx = IDX(i, j, k, pdm.shape);
        x(idx, 0) = origin[0] + (i + pdm.offset[0] - cpm.gc + 0.5) * pitch[0];
        x(idx, 1) = origin[1] + (j + pdm.offset[1] - cpm.gc + 0.5) * pitch[1];
        x(idx, 2) = origin[2] + (k + pdm.offset[2] - cpm.gc + 0.5) * pitch[2];
        h(idx, 0) = pitch[0];
        h(idx, 1) = pitch[1];
        h(idx, 2) = pitch[2];
        g(idx, 0) = volume * (dkdx[0] * dkdx[0]);
        g(idx, 1) = volume * (dkdx[1] * dkdx[1]);
        g(idx, 2) = volume * (dkdx[2] * dkdx[2]);
        kx(idx, 0) = dkdx[0];
        kx(idx, 1) = dkdx[1];
        kx(idx, 2) = dkdx[2];
        ja(idx) = volume;

        xyz(idx, 0) = origin[0] + (i + pdm.offset[0] - cpm.gc + 0.5) * pitch[0];
        xyz(idx, 1) = origin[1] + (j + pdm.offset[1] - cpm.gc + 0.5) * pitch[1];
        xyz(idx, 2) = origin[2] + (k + pdm.offset[2] - cpm.gc + 0.5) * pitch[2];
    }}}
    x.sync(MCpType::Hst2Dev);
    // h.sync(MCpType::Hst2Dev);
    kx.sync(MCpType::Hst2Dev);
    ja.sync(MCpType::Hst2Dev);
    g.sync(MCpType::Hst2Dev);
    vcdm.writeGridData(&xyz(0), cpm.gc, cpm.rank, 0, Vcdm::IdxType::IJKN);

    poisson_a.alloc(pdm.shape, 7, HDCType::Host, "poisson matrix", StencilMatrix::D3P7);
    for (INT k = cpm.gc; k < pdm.shape[2] - cpm.gc; k ++) {
    for (INT j = cpm.gc; j < pdm.shape[1] - cpm.gc; j ++) {
    for (INT i = cpm.gc; i < pdm.shape[0] - cpm.gc; i ++) {
        INT3 gijk = INT3{{i, j, k}} + pdm.offset;
        REAL ac, ae, aw, an, as, at, ab;
        ac = ae = aw = an = as = at = ab = 0.0;
        INT idxcc = IDX(i  , j  , k  , pdm.shape);
        INT idxe1 = IDX(i+1, j  , k  , pdm.shape);
        INT idxw1 = IDX(i-1, j  , k  , pdm.shape);
        INT idxn1 = IDX(i  , j+1, k  , pdm.shape);
        INT idxs1 = IDX(i  , j-1, k  , pdm.shape);
        INT idxt1 = IDX(i  , j  , k+1, pdm.shape);
        INT idxb1 = IDX(i  , j  , k-1, pdm.shape);
        REAL gxcc  =  g(idxcc, 0);
        REAL gxe1  =  g(idxe1, 0);
        REAL gxw1  =  g(idxw1, 0);
        REAL gycc  =  g(idxcc, 1);
        REAL gyn1  =  g(idxn1, 1);
        REAL gys1  =  g(idxs1, 1);
        REAL gzcc  =  g(idxcc, 2);
        REAL gzt1  =  g(idxt1, 2);
        REAL gzb1  =  g(idxb1, 2);
        REAL jacob = ja(idxcc);
        REAL coefficient;
        coefficient = 0.5 * (gxcc + gxe1) / jacob;
        if (gijk[0] < global.shape[0] - cpm.gc) {
            ac -= coefficient;
            ae  = coefficient;
        }
        coefficient = 0.5 * (gxcc + gxw1) / jacob;
        if (gijk[0] > cpm.gc) {
            ac -= coefficient;
            aw  = coefficient;
        }
        coefficient = 0.5 * (gycc + gyn1) / jacob;
        if (gijk[1] < global.shape[1] - cpm.gc - 1) {
            ac -= coefficient;
            an  = coefficient;
        }
        coefficient = 0.5 * (gycc + gys1) / jacob;
        if (gijk[1] > cpm.gc) {
            ac -= coefficient;
            as  = coefficient;
        }
        coefficient = 0.5 * (gzcc + gzt1) / jacob;
        if (gijk[2] < global.shape[2] - cpm.gc - 1) {
            ac -= coefficient;
            at  = coefficient;
        }
        coefficient = 0.5 * (gzcc + gzb1) / jacob;
        if (gijk[2] > cpm.gc) {
            ac -= coefficient;
            ab  = coefficient;
        }
        poisson_a(idxcc, 0) = ac;
        poisson_a(idxcc, 1) = aw;
        poisson_a(idxcc, 2) = ae;
        poisson_a(idxcc, 3) = as;
        poisson_a(idxcc, 4) = an;
        poisson_a(idxcc, 5) = ab;
        poisson_a(idxcc, 6) = at;
    }}}
    poisson_a.sync(MCpType::Hst2Dev);
    maxdiag = FalmMV::MaxDiag(poisson_a, cpm, {8, 8, 8});
    FalmMV::ScaleMatrix(poisson_a, 1.0 / maxdiag, {8, 8, 8});
    if (cpm.rank == 0) {
        printf("max diag = %lf\n", maxdiag);
        fflush(stdout);
    }

    RmcpTurbineArray turbineArray(1);
    RmcpTurbine turbine;
    turbine.pos = {{0, 0, 0}};
    turbine.rotpos = {{0, 0, 0}};
    turbine.R = 1;
    turbine.width = 0.2;
    turbine.thick = 0.1;
    turbine.pitch = Pi/6;
    turbine.tip = 4;
    turbine.hub = 0.1;
    turbine.chord_a = {{
          0.2876200,
        - 0.2795100,
          0.1998600,
        - 0.1753800,
          0.1064600,
        - 0.0025213
    }};
    turbine.angle_a = {{
          49.992000,
        - 70.551000,
          45.603000,
        - 40.018000,
          24.292000,
        -  0.575430
    }};
    turbineArray[0] = turbine;
    turbineArray.sync(MCpType::Hst2Dev);

    FalmCFD cfdsolver(10000, dt, AdvectionSchemeType::Upwind3, SGSType::Smagorinsky);
    FalmEq eqsolver(LSType::PBiCGStab, 1000, 1e-6, 1.2, LSType::SOR, 5, 1.5);
    RmcpAlm alm(cpm);
    if (cpm.rank == 0) {
        printf("running on %dx%dx%d grid with Re=%lf until t=%lf\n", Nxyz[0], Nxyz[1], Nxyz[1], cfdsolver.Re, endtime);
        fflush(stdout);
    }
    REAL __t = 0;
    INT  __it = 0;
    const INT __IT = int(endtime / dt);
    const INT __oIT = int(10.0 / dt);
    plt3d_output(__it, cpm.rank, dt);
    if (cpm.rank == 0) {
        printf("time advance start\n");
        fflush(stdout);
    }
    cfdsolver.UtoUU(u, uu, kx, ja, cpm, block, facestream);
    while (__it < __IT) {
        REAL dvr_norm = sqrt(main_loop(cfdsolver, eqsolver, alm, turbineArray, __it, dt, facestream)) / ginner.size;
        __t += dt;
        __it ++;
        if (cpm.rank == 0) {
            printf("\r%8d %12.5e, %12.5e, %3d, %12.5e", __it, __t, dvr_norm, eqsolver.it, eqsolver.err);
            fflush(stdout);
        }
        if (__it % __oIT == 0) {
            plt3d_output(__it, cpm.rank, dt);
        }
    }
    printf("\n");
    if (cpm.rank == 0) {
        vcdm.writeIndexDfi();
        vcdm.writeProcDfi();
    }


    for (INT fid = 0; fid < CPM::NFACE; fid ++) {
        cudaStreamDestroy(facestream[fid]);
    }
    CPM_Finalize();
    return 0;
}