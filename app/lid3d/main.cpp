#include <math.h>
#include <fstream>
#include "../../src/falm.h"
#include "bc.h"
#include "../../src/profiler.h"

using namespace Falm;

Cprof::cprof_Profiler profiler;

#define TERMINAL_OUTPUT_RANK 1

FalmCore falm;
Real maxdiag;
Matrix<Real> u_previous;

dim3 block{8, 8, 8};

Stream facestream[CPM::NFACE];
Stream *streams;

void make_poisson_coefficient_matrix() {
    CPM &cpm = falm.cpm;
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region &global = cpm.global;
    FalmBasicVar &fv = falm.fv;
    for (Int k = cpm.gc; k < pdm.shape[2] - cpm.gc; k ++) {
    for (Int j = cpm.gc; j < pdm.shape[1] - cpm.gc; j ++) {
    for (Int i = cpm.gc; i < pdm.shape[0] - cpm.gc; i ++) {
        Int3 gijk = pdm.offset + Int3{{i, j, k}};
        Real ac, ae, aw, an, as, at, ab;
        ac = ae = aw = an = as = at = ab = 0.0;
        Int idxcc = IDX(i  , j  , k  , pdm.shape);
        Int idxe1 = IDX(i+1, j  , k  , pdm.shape);
        Int idxw1 = IDX(i-1, j  , k  , pdm.shape);
        Int idxn1 = IDX(i  , j+1, k  , pdm.shape);
        Int idxs1 = IDX(i  , j-1, k  , pdm.shape);
        Int idxt1 = IDX(i  , j  , k+1, pdm.shape);
        Int idxb1 = IDX(i  , j  , k-1, pdm.shape);
        Real gxcc  =  fv.g(idxcc, 0);
        Real gxe1  =  fv.g(idxe1, 0);
        Real gxw1  =  fv.g(idxw1, 0);
        Real gycc  =  fv.g(idxcc, 1);
        Real gyn1  =  fv.g(idxn1, 1);
        Real gys1  =  fv.g(idxs1, 1);
        Real gzcc  =  fv.g(idxcc, 2);
        Real gzt1  =  fv.g(idxt1, 2);
        Real gzb1  =  fv.g(idxb1, 2);
        Real jacob = fv.ja(idxcc);
        Real coefficient;
        coefficient = 0.5 * (gxcc + gxe1) / jacob;
        if (gijk[0] < global.shape[0] - cpm.gc - 1) {
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
        fv.poi_a(idxcc, 0) = ac;
        fv.poi_a(idxcc, 1) = aw;
        fv.poi_a(idxcc, 2) = ae;
        fv.poi_a(idxcc, 3) = as;
        fv.poi_a(idxcc, 4) = an;
        fv.poi_a(idxcc, 5) = ab;
        fv.poi_a(idxcc, 6) = at;
    }}}
    fv.poi_a.sync(MCP::Hst2Dev);
    maxdiag = FalmMV::MaxDiag(fv.poi_a, falm.cpm, block);
    FalmMV::ScaleMatrix(fv.poi_a, 1.0 / maxdiag, block);
}

void init(int &argc, char **&argv) {
    falm.env_init(argc, argv);
    falm.parse_settings("setup.json");
    falm.computation_init({{falm.cpm.size, 1, 1,}}, GuideCell);
    falm.print_info(TERMINAL_OUTPUT_RANK);
    u_previous.alloc(falm.fv.u.shape[0], falm.fv.u.shape[1], HDC::Device, "previous velocity");

    for (int i = 0; i < 6; i ++) cudaStreamCreate(&facestream[i]);
    streams = facestream;
    // streams = nullptr;
    if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
        printf("using streams %p\n", streams);
    }

    Real u_inflow = falm.params["inflow"]["velocity"].get<Real>();
    Matrix<Real> &u = falm.fv.u;
    for (Int i = 0; i < u.shape[0]; i ++) {
        u(i, 0) = u_inflow;
        u(i, 1) = u(i, 2) = 0.0;
    }
    u.sync(MCP::Hst2Dev);
    FalmBasicVar &fv = falm.fv;
    falm.falmCfd.UtoUU(fv.u, fv.uu, fv.kx, fv.ja, falm.cpm, block, streams);
    falm.falmCfd.SGS(fv.u, fv.nut, fv.xyz, fv.kx, fv.ja, falm.cpm, block, streams);

    make_poisson_coefficient_matrix();
}

Real main_loop(RmcpAlm &alm, RmcpTurbineArray &turbineArray, Stream *s) {
    FalmBasicVar &fv = falm.fv;
    u_previous.copy(fv.u, HDC::Device);
    // profiler.startEvent("ALM");
    // alm.SetALMFlag(fv.xyz, falm.gettime(), turbineArray, falm.cpm, block);
    // alm.ALM(fv.u, fv.xyz, fv.ff, falm.gettime(), turbineArray, falm.cpm, block);
    // profiler.endEvent("ALM");

    FalmCFD &fcfd = falm.falmCfd;

    profiler.startEvent("U*");
    fcfd.FSPseudoU(u_previous, fv.u, fv.uu, fv.u, fv.nut, fv.kx, fv.g, fv.ja, fv.ff, falm.dt, falm.cpm, block, s, 1);
    profiler.endEvent("U*");

    profiler.startEvent("U* interpolation");
    fcfd.UtoUU(fv.u, fv.uu, fv.kx, fv.ja, falm.cpm, block, s);
    LID3D::uubc(fv.uu, falm.cpm, s);
    profiler.endEvent("U* interpolation");

    profiler.startEvent("div(U*)/dt");
    fcfd.MACCalcPoissonRHS(fv.uu, fv.poi_rhs, fv.ja, falm.dt, falm.cpm, block, maxdiag);
    profiler.endEvent("div(U*)/dt");

    FalmEq &feq = falm.falmEq;
    profiler.startEvent("div(grad p)=div(U*)/dt");
    feq.Solve(fv.poi_a, fv.p, fv.poi_rhs, fv.poi_res, falm.cpm, block, s);
    profiler.endEvent("div(grad p)=div(U*)/dt");

    profiler.startEvent("p BC");
    LID3D::pbc(fv.p, falm.cpm, s);
    profiler.endEvent("p BC");

    profiler.startEvent("U = U* - dt grad(p)");
    fcfd.ProjectP(fv.u, fv.u, fv.uu, fv.uu, fv.p, fv.kx, fv.g, falm.dt, falm.cpm, block, s);
    profiler.endEvent("U = U* - dt grad(p)");

    profiler.startEvent("U BC");
    LID3D::ubc(fv.u, falm.cpm, s);
    profiler.endEvent("U BC");

    profiler.startEvent("Nut");
    fcfd.SGS(fv.u, fv.nut, fv.xyz, fv.kx, fv.ja, falm.cpm, block, s);
    profiler.endEvent("Nut");

    profiler.startEvent("||div(U)||");
    fcfd.Divergence(fv.uu, fv.divergence, fv.ja, falm.cpm, block);
    Real divnorm = FalmMV::EuclideanNormSq(fv.divergence, falm.cpm, block);
    profiler.endEvent("||div(U)||");

    falm.TAvg();
    falm.outputUVWP();

    return divnorm;
}

void finalize() {
    for (int i = 0; i < 6; i ++) cudaStreamDestroy(facestream[i]);
    u_previous.release();
    falm.env_finalize();
}

int main(int argc, char **argv) {
    init(argc, argv);

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
    turbineArray.sync(MCP::Hst2Dev);

    RmcpAlm alm(falm.cpm);

    if (argc > 1) {
        std::string arg(argv[1]);
        if (arg == "info") {
            goto FIN;
        }
    }

    printf("\n");
    if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
        printf("maxdiag = %e\n", maxdiag);
    }
    profiler.startEvent("global loop");
    for (falm.it = 1; falm.it <= falm.maxIt; falm.it ++) {
        REAL divnorm = sqrt(main_loop(alm, turbineArray, streams)) / PRODUCT3(falm.cpm.pdm_list[falm.cpm.rank].shape - INT(2 * falm.cpm.gc));
        // falm.it ++;
        if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
            printf("%8d %12.5e, %12.5e, %3d, %12.5e\n", falm.it, falm.gettime(), divnorm, falm.falmEq.it, falm.falmEq.err);
            fflush(stdout);
        }
    }
    profiler.endEvent("global loop");
    printf("\n");
FIN:
    if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
        profiler.output();
        printf("\n");
        // pprofiler.output();
    }
    finalize();
    return 0;
}