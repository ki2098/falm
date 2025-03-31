#include <math.h>
#include <fstream>
#include "../../src/FalmCFD.h"
#include "../../src/FalmEq.h"
#include "../../src/rmcp/alm.h"
#include "../../src/falmath.h"
#include "bc.h"

using namespace Falm;

const dim3 block(8, 8, 8);
Real3 Lxyz;
Int3  Nxyz;
Real3 origin;

Real endtime;
Real dt;
Real write_interval;

Matrix<Real> gx, gy, gz, ghx, ghy, ghz;
Matrix<Real> x, h, kx, g, ja;
Matrix<Real> u, uprev, uu, ua, uua, p, nut, ff ,rhs, res, dvr, vrt;
Matrix<Real> poisson_a;
Real maxdiag;
CPM cpm;
Vcdm::VCDM<float> vcdm, pvcdm;
Stream facestream[CPM::NFACE];
std::string gridpath;
Stream *streams;

FalmEq eqparam;

void read_param() {
    std::ifstream xfile(gridpath + "/x.txt");
    std::ifstream yfile(gridpath + "/y.txt");
    std::ifstream zfile(gridpath + "/z.txt");
    if (!xfile || !yfile || !zfile) {
        printf("Cannot open grid file\n");
    }
    std::string line;
    int nx, ny, nz;
    std::getline(xfile, line);
    nx = std::stoi(line);
    Nxyz[0] = nx - 1;
    std::getline(yfile, line);
    ny = std::stoi(line);
    Nxyz[1] = ny - 1;
    std::getline(zfile, line);
    nz = std::stoi(line);
    Nxyz[2] = nz - 1;
    xfile.close();
    yfile.close();
    zfile.close();

    std::ifstream dfile(gridpath + "/domain.txt");
    std::getline(dfile, line);
    origin[0] = std::stod(line);
    std::getline(dfile, line);
    origin[1] = std::stod(line);
    std::getline(dfile, line);
    origin[2] = std::stod(line);
    std::getline(dfile, line);
    Lxyz[0] = std::stod(line);
    std::getline(dfile, line);
    Lxyz[1] = std::stod(line);
    std::getline(dfile, line);
    Lxyz[2] = std::stod(line);
    dfile.close();

    std::ifstream lsfile(gridpath + "/ls.txt");
    std::getline(lsfile, line);
    if (line == "PBiCGStab") {
        eqparam.type = LSType::PBiCGStab;
    } else if (line == "SOR") {
        eqparam.type = LSType::SOR;
    } else if (line == "Jacobi") {
        eqparam.type = LSType::Jacobi;
    }
    std::getline(lsfile, line);
    eqparam.maxit = std::stoi(line);
    std::getline(lsfile, line);
    eqparam.tol = std::stod(line);
    std::getline(lsfile, line);
    eqparam.relax_factor = std::stod(line);
    std::getline(lsfile, line);
    if (line == "SOR") {
        eqparam.pc_type = LSType::SOR;
    } else if (line == "Jacobi") {
        eqparam.pc_type = LSType::Jacobi;
    }
    std::getline(lsfile, line);
    eqparam.pc_maxit = std::stoi(line);
    std::getline(lsfile, line);
    eqparam.pc_relax_factor = std::stod(line);
    lsfile.close();

    std::ifstream runfile(gridpath + "/run.txt");
    std::getline(runfile, line);
    endtime = std::stod(line);
    std::getline(runfile, line);
    dt = std::stod(line);
    std::getline(runfile, line);
    if (line == "streams") {
        streams = facestream;
    } else {
        streams = nullptr;
    }
    std::getline(runfile, line);
    write_interval = std::stod(line);
    runfile.close();
}

void read_grid() {
    const Int3 &shape = cpm.pdm_list[cpm.rank].shape;
    const Int3 &gshape = cpm.global.shape;
    std::ifstream xfile(gridpath + "/x.txt");
    std::ifstream yfile(gridpath + "/y.txt");
    std::ifstream zfile(gridpath + "/z.txt");
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
    gx.alloc(gshape[0], 1, HDC::Host);
    gy.alloc(gshape[1], 1, HDC::Host);
    gz.alloc(gshape[2], 1, HDC::Host);
    ghx.alloc(gshape[0], 1, HDC::Host);
    ghy.alloc(gshape[1], 1, HDC::Host);
    ghz.alloc(gshape[2], 1, HDC::Host);
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

    const Int3 &offset = cpm.pdm_list[cpm.rank].offset;
    for (Int k = 0; k < shape[2]; k ++) {
    for (Int j = 0; j < shape[1]; j ++) {
    for (Int i = 0; i < shape[0]; i ++) {
        Real idx = IDX(i, j, k, shape);
        x(idx, 0) = gx(i + offset[0]);
        x(idx, 1) = gy(j + offset[1]);
        x(idx, 2) = gz(k + offset[2]);
        h(idx, 0) = ghx(i + offset[0]);
        h(idx, 1) = ghy(j + offset[1]);
        h(idx, 2) = ghz(k + offset[2]);
    }}}

    Matrix<float> fgx(gx.shape[0], gx.shape[1], HDC::Host, "x float");
    Matrix<float> fgy(gy.shape[0], gy.shape[1], HDC::Host, "y float");
    Matrix<float> fgz(gz.shape[0], gz.shape[1], HDC::Host, "z float");

    for (int i = 0; i < gx.size; i ++) fgx(i) = gx(i);
    for (int i = 0; i < gy.size; i ++) fgy(i) = gy(i);
    for (int i = 0; i < gz.size; i ++) fgz(i) = gz(i);

    vcdm.writeCrd(&fgx(cpm.gc), &fgy(cpm.gc), &fgz(cpm.gc));

    gx.release(HDC::Host);
    gy.release(HDC::Host);
    gz.release(HDC::Host);
    ghx.release(HDC::Host);
    ghy.release(HDC::Host);
    ghz.release(HDC::Host);
    xfile.close();
    yfile.close();
    zfile.close();
}

void data_output(int step, int rank, double dt) {
    // Matrix<float> uvw(cpm.pdm_list[cpm.rank].shape, 7, HDCType::Host, "uvw");
    // u.sync(MCpType::Dev2Hst);
    // p.sync(MCpType::Dev2Hst);
    // vrt.sync(MCpType::Dev2Hst);
    // for (INT i = 0; i < u.shape[0]; i ++) {
    //     uvw(i, 0) = u(i, 0);
    //     uvw(i, 1) = u(i, 1);
    //     uvw(i, 2) = u(i, 2);
    //     uvw(i, 3) = p(i);
    //     uvw(i, 4) = vrt(i, 0);
    //     uvw(i, 5) = vrt(i, 1);
    //     uvw(i, 6) = vrt(i, 2);
    // }
    // vcdm.writeFileData(&uvw(0, 0), cpm.gc, 7, rank, step, Vcdm::IdxType::IJKN);
    // double umax = FalmMV::MatColMax(u, 0, cpm, block);
    // double vmax = FalmMV::MatColMax(u, 1, cpm, block);
    // double wmax = FalmMV::MatColMax(u, 2, cpm, block);
    // double pmax = FalmMV::MatColMax(p, 0, cpm, block);
    // double vxmax = FalmMV::MatColMax(vrt, 0, cpm, block);
    // double vymax = FalmMV::MatColMax(vrt, 1, cpm, block);
    // double vzmax = FalmMV::MatColMax(vrt, 2, cpm, block);
    // double umin = FalmMV::MatColMin(u, 0, cpm, block);
    // double vmin = FalmMV::MatColMin(u, 1, cpm, block);
    // double wmin = FalmMV::MatColMin(u, 2, cpm, block);
    // double pmin = FalmMV::MatColMin(p, 0, cpm, block);
    // double vxmin = FalmMV::MatColMin(vrt, 0, cpm, block);
    // double vymin = FalmMV::MatColMin(vrt, 1, cpm, block);
    // double vzmin = FalmMV::MatColMin(vrt, 2, cpm, block);
    // Vcdm::VcdmSlice slice;
    // slice.step = step;
    // slice.time = step * dt;
    // slice.avgStep = 1;
    // slice.avgTime = dt;
    // slice.avgMode = true;
    // // slice.vectorMax = _max;
    // // slice.vectorMin = _min;
    // slice.varMax = {umax, vmax, wmax, pmax, vxmax, vymax, vzmax};
    // slice.varMin = {umin, vmin, wmin, pmin, vxmin, vymin, vzmin};
    // vcdm.timeSlice.push_back(slice);
    Matrix<float> uf(cpm.pdm_list[cpm.rank].shape, 3, HDC::Host, "uvw");
    Matrix<float> pf(cpm.pdm_list[cpm.rank].shape, 1, HDC::Host, "p");
    u.sync(McpType::Dev2Hst);
    p.sync(McpType::Dev2Hst);
    for (int i = 0; i < cpm.pdm_list[cpm.rank].size; i ++) {
        for (int n = 0; n < 3; n ++) uf(i, n) = u(i, n);
        pf(i) = p(i);
    }
    vcdm.writeFileData(&uf(0), cpm.gc, uf.shape[1], cpm.rank, step, step * dt, Vcdm::IdxType::IJKN);
    pvcdm.writeFileData(&pf(0), cpm.gc, pf.shape[1], cpm.rank, step, step * dt, Vcdm::IdxType::IJK);

    Real3 umax = {{
        FalmMV::MatColMax(u, 0, cpm, block),
        FalmMV::MatColMax(u, 1, cpm, block),
        FalmMV::MatColMax(u, 2, cpm, block)
    }};
    Real3 umin = {{
        FalmMV::MatColMin(u, 0, cpm, block),
        FalmMV::MatColMin(u, 1, cpm, block),
        FalmMV::MatColMin(u, 2, cpm, block)
    }};
    Real uvecmax = FalmMV::VecMax(u, cpm, block);
    Real uvecmin = FalmMV::VecMin(u, cpm, block);
    Vcdm::VcdmSlice slice;
    slice.step = step;
    slice.time = step * dt;
    slice.avgStep = 1;
    slice.avgTime = dt;
    slice.avgMode = true;
    slice.vectorMax = uvecmax;
    slice.vectorMin = uvecmin;
    slice.varMax = {umax[0], umax[1], umax[2]};
    slice.varMin = {umin[0], umin[1], umin[2]};
    vcdm.timeSlice.push_back(slice);
    Real pmax = FalmMV::MatColMax(p, 0, cpm, block);
    Real pmin = FalmMV::MatColMin(p, 0, cpm, block);
    slice.varMax = {pmax};
    slice.varMin = {pmin};
    pvcdm.timeSlice.push_back(slice);
}

Real main_loop(FalmCFD &cfd, FalmEq &eq, RmcpAlm &alm, RmcpTurbineArray &turbineArray, Int step, Real dt, Stream *s) {
    uprev.copy(u, HDC::Device);
    alm.SetALMFlag(x, dt * step, turbineArray, cpm, block);
    alm.ALM(u, x, ff, dt * step, turbineArray, cpm, block);

    cfd.FSPseudoU(uprev, u, uu, u, nut, kx, g, ja, ff, dt, cpm, block, s, 1);
    cfd.UtoUU(u, uu, kx, ja, cpm, block, s);
    cfd.MACCalcPoissonRHS(uu, rhs, ja, dt, cpm, block, maxdiag);
    
    eq.Solve(poisson_a, p, rhs, res, cpm, block, s);
    TURBINE1::pbc(p, cpm, s);

    cfd.ProjectP(u, u, uu, uu, p, kx, g, dt, cpm, block, s);
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
    gridpath = std::string(argv[1]);
    int mpi_rank, mpi_size;
    cpm.use_cuda_aware_mpi = false;
    CPM_GetRank(MPI_COMM_WORLD, mpi_rank);
    CPM_GetSize(MPI_COMM_WORLD, mpi_size);
    read_param();
    cpm.initPartition(Nxyz - Int3{{1,1,1}}, GuideCell, mpi_rank, mpi_size, {{mpi_size, 1, 1}});
    int ngpu;
    cudaGetDeviceCount(&ngpu);
    cudaSetDevice(cpm.rank % ngpu);
    printf("rank %d on gpu %d\n", cpm.rank, cpm.rank % ngpu); fflush(stdout);
    CPM_Barrier(MPI_COMM_WORLD);
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region &global = cpm.global;
    Region ginner(global.shape, cpm.gc);
    printf("rank %d, global size = (%d %d %d), proc size = (%d %d %d), proc offset = (%d %d %d)\n", cpm.rank, global.shape[0], global.shape[1], global.shape[2], pdm.shape[0], pdm.shape[1], pdm.shape[2], pdm.offset[0], pdm.offset[1], pdm.offset[2]);
    fflush(stdout); CPM_Barrier(MPI_COMM_WORLD);
    vcdm.setPath("data", "velocity");
    pvcdm.setPath("data", "pressure");
    setVcdm(cpm, vcdm, {{Lxyz[0],Lxyz[1],Lxyz[2]}}, {{origin[0], origin[1], origin[2]}});
    setVcdm(cpm, pvcdm, {{Lxyz[0],Lxyz[1],Lxyz[2]}}, {{origin[0], origin[1], origin[2]}});
    vcdm.dfiFinfo.varList = {"u", "v", "w"};
    pvcdm.dfiFinfo.varList = {"p"};
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
        pvcdm.writeIndexDfi();
        pvcdm.writeProcDfi();
    }
    CPM_Barrier(MPI_COMM_WORLD);

    for (int fid = 0; fid < 6; fid ++) {
        cudaStreamCreate(&facestream[fid]);
    }

    u.alloc(pdm.shape, 3, HDC::HstDev);
    uprev.alloc(pdm.shape, 3, HDC::Device);
    ua.alloc(pdm.shape, 3, HDC::Device);
    uu.alloc(pdm.shape, 3, HDC::Device);
    uua.alloc(pdm.shape, 3, HDC::Device);
    p.alloc(pdm.shape, 1, HDC::Device);
    nut.alloc(pdm.shape, 1, HDC::Device);
    ff.alloc(pdm.shape, 3, HDC::Device);
    rhs.alloc(pdm.shape, 1, HDC::Device);
    res.alloc(pdm.shape, 1, HDC::Device);
    dvr.alloc(pdm.shape, 1, HDC::Device);
    vrt.alloc(pdm.shape, 3, HDC::Device);
    for (Int id = 0; id < u.shape[0]; id ++) {
        u(id, 0) = 1.0;
        u(id, 1) = u(id, 2) = 0.0;
    }
    u.sync(McpType::Hst2Dev);

    Matrix<float> xyz(pdm.shape, 3, HDC::Host);
    x.alloc(pdm.shape, 3, HDC::Host);
    h.alloc(pdm.shape, 3, HDC::Host);
    kx.alloc(pdm.shape, 3, HDC::Host);
    ja.alloc(pdm.shape, 1, HDC::Host);
    g.alloc(pdm.shape, 3, HDC::Host);
    read_grid();
    for (Int k = 0; k < pdm.shape[2]; k ++) {
    for (Int j = 0; j < pdm.shape[1]; j ++) {
    for (Int i = 0; i < pdm.shape[0]; i ++) {
        Int idx = IDX(i, j, k, pdm.shape);
        Real3 pitch;
        // pitch[0] = Lxyz[0] / Nxyz[0];
        // pitch[1] = Lxyz[1] / Nxyz[1];
        // pitch[2] = Lxyz[2] / Nxyz[2];
        // h(idx, 0) = pitch[0];
        // h(idx, 1) = pitch[1];
        // h(idx, 2) = pitch[2];
        // x(idx, 0) = origin[0] + (i + pdm.offset[0] - cpm.gc + 1) * pitch[0];
        // x(idx, 1) = origin[1] + (j + pdm.offset[1] - cpm.gc + 1) * pitch[1];
        // x(idx, 2) = origin[2] + (k + pdm.offset[2] - cpm.gc + 1) * pitch[2];
        pitch[0] = h(idx, 0);
        pitch[1] = h(idx, 1);
        pitch[2] = h(idx, 2);
        const Real volume = pitch[0] * pitch[1] * pitch[2];
        const Real3 dkdx  = {{1.0/pitch[0], 1.0/pitch[1], 1.0/pitch[2]}};
        g(idx, 0) = volume * (dkdx[0] * dkdx[0]);
        g(idx, 1) = volume * (dkdx[1] * dkdx[1]);
        g(idx, 2) = volume * (dkdx[2] * dkdx[2]);
        kx(idx, 0) = dkdx[0];
        kx(idx, 1) = dkdx[1];
        kx(idx, 2) = dkdx[2];
        ja(idx) = volume;

        xyz(idx, 0) = x(idx, 0);
        xyz(idx, 1) = x(idx, 1);
        xyz(idx, 2) = x(idx, 2);
    }}}
    x.sync(McpType::Hst2Dev);
    // h.sync(MCpType::Hst2Dev);
    kx.sync(McpType::Hst2Dev);
    ja.sync(McpType::Hst2Dev);
    g.sync(McpType::Hst2Dev);
    // vcdm.writeGridData(&xyz(0), cpm.gc, cpm.rank, 0, Vcdm::IdxType::IJKN);
    xyz.release(HDC::Host);

    poisson_a.alloc(pdm.shape, 7, HDC::Host, "poisson matrix", StencilMatrix::D3P7);
    for (Int k = cpm.gc; k < pdm.shape[2] - cpm.gc; k ++) {
    for (Int j = cpm.gc; j < pdm.shape[1] - cpm.gc; j ++) {
    for (Int i = cpm.gc; i < pdm.shape[0] - cpm.gc; i ++) {
        Int3 gijk = Int3{{i, j, k}} + pdm.offset;
        Real ac, ae, aw, an, as, at, ab;
        ac = ae = aw = an = as = at = ab = 0.0;
        Int idxcc = IDX(i  , j  , k  , pdm.shape);
        Int idxe1 = IDX(i+1, j  , k  , pdm.shape);
        Int idxw1 = IDX(i-1, j  , k  , pdm.shape);
        Int idxn1 = IDX(i  , j+1, k  , pdm.shape);
        Int idxs1 = IDX(i  , j-1, k  , pdm.shape);
        Int idxt1 = IDX(i  , j  , k+1, pdm.shape);
        Int idxb1 = IDX(i  , j  , k-1, pdm.shape);
        Real gxcc  =  g(idxcc, 0);
        Real gxe1  =  g(idxe1, 0);
        Real gxw1  =  g(idxw1, 0);
        Real gycc  =  g(idxcc, 1);
        Real gyn1  =  g(idxn1, 1);
        Real gys1  =  g(idxs1, 1);
        Real gzcc  =  g(idxcc, 2);
        Real gzt1  =  g(idxt1, 2);
        Real gzb1  =  g(idxb1, 2);
        Real jacob = ja(idxcc);
        Real coefficient;
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
    poisson_a.sync(McpType::Hst2Dev);
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
    turbine.yaw = 0;
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
    turbineArray.sync(McpType::Hst2Dev);

    FalmCFD cfdsolver(10000, AdvectionSchemeType::Upwind3, SGSType::Smagorinsky);
    FalmEq eqsolver(eqparam.type, eqparam.maxit, eqparam.tol, eqparam.relax_factor, eqparam.pc_type, eqparam.pc_maxit, eqparam.pc_relax_factor);
    if (gridpath == "weak") {
        eqsolver.maxit = 1;
        eqsolver.tol = 0.0;
    }
    RmcpAlm alm(cpm);
    if (cpm.rank == 0) {
        printf("running on %dx%dx%d grid with Re=%lf until t=%lf by dt=%e using linear solver %d, streams=%p\n", Nxyz[0], Nxyz[1], Nxyz[1], cfdsolver.Re, endtime, dt, (int)eqsolver.type, streams);
        printf("output every %e undimensional time\n", write_interval);
        fflush(stdout);
    }
    Real __t = 0;
    Int  __it = 0;
    const Int __IT = int(endtime / dt);
    const Int __oIT = int(write_interval / dt);
    data_output(__it, cpm.rank, dt);
    if (cpm.rank == 0) {
        printf("time advance start\n");
        // size_t freebyte, totalbyte;
        // cudaMemGetInfo(&freebyte, &totalbyte);
        // printf("%8d %12.5e, %12.5e, %3d, %12.5e, %e %e\n", __it, __t, 0, eqsolver.it, eqsolver.err, freebyte / (1024. * 1024.), totalbyte / (1024. * 1024.));
        fflush(stdout);
    }
    double t_start = MPI_Wtime();
    
    cfdsolver.UtoUU(u, uu, kx, ja, cpm, block, streams);
    cfdsolver.SGS(u, nut, x, kx, ja, cpm, block, streams);
    while (__it < __IT) {
        Real dvr_norm = sqrt(main_loop(cfdsolver, eqsolver, alm, turbineArray, __it, dt, streams)) / ginner.size;
        __t += dt;
        __it ++;
        // size_t freebyte, totalbyte;
        // cudaMemGetInfo(&freebyte, &totalbyte);
        // printf("\nrank %d: free %lf, total %lf\n", cpm.rank, freebyte / (1024. * 1024.), totalbyte / (1024. * 1024.));
        // fflush(stdout);
        // CPM_Barrier(MPI_COMM_WORLD);
        // if (cpm.rank == 0) {
        //     printf("%8d %12.5e, %12.5e, %3d, %12.5e\n", __it, __t, dvr_norm, eqsolver.it, eqsolver.err);
        //     fflush(stdout);
        // }
        if (cpm.rank == 0) {
            // size_t freebyte = 0, totalbyte = 0;
            // cudaMemGetInfo(&freebyte, &totalbyte);
            printf("%8d %12.5e, %12.5e, %3d, %12.5e\n", __it, __t, dvr_norm, eqsolver.it, eqsolver.err);
            fflush(stdout);
        }
        if (__it % __oIT == 0) {
            data_output(__it, cpm.rank, dt);
        }
    }
    double t_end = MPI_Wtime();
    printf("\n");
    // plt3d_output(__it, cpm.rank, dt);
    if (cpm.rank == 0) {
        vcdm.writeIndexDfi();
        vcdm.writeProcDfi();
        pvcdm.writeIndexDfi();
        pvcdm.writeProcDfi();
        printf("wall time = %lf\n", t_end - t_start);
    }

End:
    for (Int fid = 0; fid < CPM::NFACE; fid ++) {
        cudaStreamDestroy(facestream[fid]);
    }
    CPM_Finalize();
    return 0;
}