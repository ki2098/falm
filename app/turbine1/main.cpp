#include <math.h>
#include <fstream>
#include "../../src/FalmCFD.h"
#include "../../src/FalmEq.h"
#include "../../src/rmcp/alm.h"
#include "../../src/falmath.h"
#include "bc.h"

using namespace Falm;

const dim3 block(8, 8, 8);
const REAL3 Lxyz{{24.0, 8.0, 8.0}};
const INT3  Nxyz{{750, 250, 250}};
const REAL3 origin{{-4,-4,-4}};

const REAL endtime = 100;
const REAL dt = 5e-3;

Matrix<REAL> gx, gy, gz, ghx, ghy, ghz;
Matrix<REAL> x, h, kx, g, ja;
Matrix<REAL> u, uprev, uu, ua, uua, p, nut, ff ,rhs, res, dvr, vrt;
Matrix<REAL> poisson_a;
REAL maxdiag;
CPM cpm;
Vcdm::VCDM<float> vcdm;
STREAM facestream[CPM::NFACE];

void read_grid() {
    const INT3 &shape = cpm.pdm_list[cpm.rank].shape;
    const INT3 &gshape = cpm.global.shape;
    std::ifstream xfile("x.txt");
    std::ifstream yfile("y.txt");
    std::ifstream zfile("z.txt");
    if (!xfile || !yfile || !zfile) {
        printf("Cannot open grid file\n");
    }
    std::string line;
    int nx, ny, nz;
    std::getline(xfile, line);
    nx = std::stoi(line);
    std::getline(yfile, line);
    ny = std::stoi(line);
    std::getline(zfile, line);
    nz = std::stoi(line);
    printf("%d %d %d\n", nx, ny, nz);
    if (nx != gshape[0] - 2 || ny != gshape[1] - 2 || nz != gshape[2] - 2) {
        printf("Wrong domain size\n");
    }
    gx.alloc(gshape[0], 1, HDCType::Host);
    gy.alloc(gshape[1], 1, HDCType::Host);
    gz.alloc(gshape[2], 1, HDCType::Host);
    ghx.alloc(gshape[0], 1, HDCType::Host);
    ghy.alloc(gshape[1], 1, HDCType::Host);
    ghz.alloc(gshape[2], 1, HDCType::Host);
    for (int id = 1; id < gshape[0] - 1; id ++) {
        std::getline(xfile, line);
        gx(id) = std::stod(line);
    }
    gx(0         ) = 3 * gx(1         ) - 3 * gx(2         ) +     gx(3         );
    gx(gshape[0]-1) =     gx(gshape[0]-4) - 3 * gx(gshape[0]-3) + 3 * gx(gshape[0]-2);
    for (int id = 1; id < gshape[0] - 1; id ++) {
        ghx(id) = 0.5 * (gx(id+1) - gx(id-1));
    }
    ghx(0         ) = 2 * ghx(1         ) - ghx(2         );
    ghx(gshape[0]-1) = 2 * ghx(gshape[0]-2) - ghx(gshape[0]-3);

    for (int id = 1; id < gshape[1] - 1; id ++) {
        std::getline(yfile, line);
        gy(id) = std::stod(line);
    }
    gy(0         ) = 3 * gy(1         ) - 3 * gy(2         ) +     gy(3         );
    gy(gshape[1]-1) =     gy(gshape[1]-4) - 3 * gy(gshape[1]-3) + 3 * gy(gshape[1]-2);
    for (int id = 1; id < gshape[1] - 1; id ++) {
        ghy(id) = 0.5 * (gy(id+1) - gy(id-1));
    }
    ghy(0         ) = 2 * ghy(1         ) - ghy(2         );
    ghy(gshape[1]-1) = 2 * ghy(gshape[1]-2) - ghy(gshape[1]-3);

    for (int id = 1; id < gshape[2] - 1; id ++) {
        std::getline(zfile, line);
        gz(id) = std::stod(line);
    }
    gz(0         ) = 3 * gz(1         ) - 3 * gz(2         ) +     gz(3         );
    gz(gshape[2]-1) =     gz(gshape[2]-4) - 3 * gz(gshape[2]-3) + 3 * gz(gshape[2]-2);
    for (int id = 1; id < gshape[2] - 1; id ++) {
        ghz(id) = 0.5 * (gz(id+1) - gz(id-1));
    }
    ghz(0         ) = 2 * ghz(1         ) - ghz(2         );
    ghz(gshape[2]-1) = 2 * ghz(gshape[2]-2) - ghz(gshape[2]-3);

    const INT3 &offset = cpm.pdm_list[cpm.rank].offset;
    for (INT k = 0; k < shape[2]; k ++) {
    for (INT j = 0; j < shape[1]; j ++) {
    for (INT i = 0; i < shape[0]; i ++) {
        REAL idx = IDX(i, j, k, shape);
        x(idx, 0) = gx(i + offset[0]);
        x(idx, 1) = gy(j + offset[1]);
        x(idx, 2) = gz(k + offset[2]);
        h(idx, 0) = ghx(i + offset[0]);
        h(idx, 1) = ghy(j + offset[1]);
        h(idx, 2) = ghz(k + offset[2]);
    }}}

    gx.release(HDCType::Host);
    gy.release(HDCType::Host);
    gz.release(HDCType::Host);
    ghx.release(HDCType::Host);
    ghy.release(HDCType::Host);
    ghz.release(HDCType::Host);
}

void plt3d_output(int step, int rank, double dt) {
    Matrix<float> uvw(cpm.pdm_list[cpm.rank].shape, 7, HDCType::Host, "uvw");
    u.sync(MCpType::Dev2Hst);
    p.sync(MCpType::Dev2Hst);
    vrt.sync(MCpType::Dev2Hst);
    for (INT i = 0; i < u.shape[0]; i ++) {
        uvw(i, 0) = u(i, 0);
        uvw(i, 1) = u(i, 1);
        uvw(i, 2) = u(i, 2);
        uvw(i, 3) = p(i);
        uvw(i, 4) = vrt(i, 0);
        uvw(i, 5) = vrt(i, 1);
        uvw(i, 6) = vrt(i, 2);
    }
    vcdm.writeFileData(&uvw(0, 0), cpm.gc, 7, rank, step, Vcdm::IdxType::IJKN);
    double umax = FalmMV::MatColMax(u, 0, cpm, block);
    double vmax = FalmMV::MatColMax(u, 1, cpm, block);
    double wmax = FalmMV::MatColMax(u, 2, cpm, block);
    double pmax = FalmMV::MatColMax(p, 0, cpm, block);
    double vxmax = FalmMV::MatColMax(vrt, 0, cpm, block);
    double vymax = FalmMV::MatColMax(vrt, 1, cpm, block);
    double vzmax = FalmMV::MatColMax(vrt, 2, cpm, block);
    double umin = FalmMV::MatColMin(u, 0, cpm, block);
    double vmin = FalmMV::MatColMin(u, 1, cpm, block);
    double wmin = FalmMV::MatColMin(u, 2, cpm, block);
    double pmin = FalmMV::MatColMin(p, 0, cpm, block);
    double vxmin = FalmMV::MatColMin(vrt, 0, cpm, block);
    double vymin = FalmMV::MatColMin(vrt, 1, cpm, block);
    double vzmin = FalmMV::MatColMin(vrt, 2, cpm, block);
    Vcdm::VcdmSlice slice;
    slice.step = step;
    slice.time = step * dt;
    slice.avgStep = 1;
    slice.avgTime = dt;
    slice.avgMode = true;
    // slice.vectorMax = _max;
    // slice.vectorMin = _min;
    slice.varMax = {umax, vmax, wmax, pmax, vxmax, vymax, vzmax};
    slice.varMin = {umin, vmin, wmin, pmin, vxmin, vymin, vzmin};
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

    cfd.FalmCFDDevCall::Vortcity(u, kx, vrt, cpm.pdm_list[cpm.rank], Region(cpm.pdm_list[cpm.rank].shape, cpm.gc), block);

    return FalmMV::EuclideanNormSq(dvr, cpm, block);
}

int main(int argc, char **argv) {
    assert(GuideCell == 2);
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
    vcdm.dfiFinfo.varList = {"u", "v", "w", "p", "vorticityX", "vorticityY", "vorticityZ"};
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
    vrt.alloc(pdm.shape, 3, HDCType::Device);
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
    read_grid();
    for (INT k = 0; k < pdm.shape[2]; k ++) {
    for (INT j = 0; j < pdm.shape[1]; j ++) {
    for (INT i = 0; i < pdm.shape[0]; i ++) {
        INT idx = IDX(i, j, k, pdm.shape);
        REAL3 pitch;
        pitch[0] = h(idx, 0);
        pitch[1] = h(idx, 1);
        pitch[2] = h(idx, 2);
        const REAL volume = pitch[0] * pitch[1] * pitch[2];
        const REAL3 dkdx  = {{1.0/pitch[0], 1.0/pitch[1], 1.0/pitch[2]}};
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
    FalmEq eqsolver(LSType::PBiCGStab, 1000, 1e-6, 1.2, LSType::SOR, 3, 1.5);
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
    double t_start = MPI_Wtime();
    while (__it < __IT) {
        REAL dvr_norm = sqrt(main_loop(cfdsolver, eqsolver, alm, turbineArray, __it, dt, facestream)) / ginner.size;
        __t += dt;
        __it ++;
        if (cpm.rank == 0) {
            printf("%8d %12.5e, %12.5e, %3d, %12.5e\n", __it, __t, dvr_norm, eqsolver.it, eqsolver.err);
            fflush(stdout);
        }
        if (__it % __oIT == 0) {
            plt3d_output(__it, cpm.rank, dt);
        }
    }
    double t_end = MPI_Wtime();
    printf("\n");
    if (cpm.rank == 0) {
        vcdm.writeIndexDfi();
        vcdm.writeProcDfi();
        printf("wall time = %lf\n", t_end - t_start);
    }


    for (INT fid = 0; fid < CPM::NFACE; fid ++) {
        cudaStreamDestroy(facestream[fid]);
    }
    CPM_Finalize();
    return 0;
}