#include <math.h>
#include <fstream>
#include "../../src/falm.h"
#include "bc.h"
#include "../../src/profiler.h"

using namespace Falm;

Cprof::cprof_Profiler profiler;

#define TERMINAL_OUTPUT_RANK 0

FalmCore falm;
REAL maxdiag, maxdiag2=0;
Matrix<REAL> u_previous;
// BladeHandler blades;
// Rmcp::RmcpAlm alm;
Alm::AlmHandler aalm;

dim3 block{8, 8, 8};

STREAM facestream[CPM::NFACE];
STREAM *streams;

void make_poisson_coefficient_matrix() {
    CPM &cpm = falm.cpm;
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region &global = cpm.global;
    FalmBasicVar &fv = falm.fv;
    for (INT k = cpm.gc; k < pdm.shape[2] - cpm.gc; k ++) {
    for (INT j = cpm.gc; j < pdm.shape[1] - cpm.gc; j ++) {
    for (INT i = cpm.gc; i < pdm.shape[0] - cpm.gc; i ++) {
        INT3 gijk = pdm.offset + INT3{{i, j, k}};
        REAL ac, ae, aw, an, as, at, ab;
        ac = ae = aw = an = as = at = ab = 0.0;
        INT idxcc = IDX(i  , j  , k  , pdm.shape);
        INT idxe1 = IDX(i+1, j  , k  , pdm.shape);
        INT idxw1 = IDX(i-1, j  , k  , pdm.shape);
        INT idxn1 = IDX(i  , j+1, k  , pdm.shape);
        INT idxs1 = IDX(i  , j-1, k  , pdm.shape);
        INT idxt1 = IDX(i  , j  , k+1, pdm.shape);
        INT idxb1 = IDX(i  , j  , k-1, pdm.shape);
        REAL gxcc  =  fv.g(idxcc, 0);
        REAL gxe1  =  fv.g(idxe1, 0);
        REAL gxw1  =  fv.g(idxw1, 0);
        REAL gycc  =  fv.g(idxcc, 1);
        REAL gyn1  =  fv.g(idxn1, 1);
        REAL gys1  =  fv.g(idxs1, 1);
        REAL gzcc  =  fv.g(idxcc, 2);
        REAL gzt1  =  fv.g(idxt1, 2);
        REAL gzb1  =  fv.g(idxb1, 2);
        REAL jacob = fv.ja(idxcc);
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
        fv.poi_a(idxcc, 0) = ac;
        fv.poi_a(idxcc, 1) = aw;
        fv.poi_a(idxcc, 2) = ae;
        fv.poi_a(idxcc, 3) = as;
        fv.poi_a(idxcc, 4) = an;
        fv.poi_a(idxcc, 5) = ab;
        fv.poi_a(idxcc, 6) = at;
        // printf("%e\n", jacob);
        // if (fabs(ac) > maxdiag2) {
        //     maxdiag2 = fabs(ac);
        // }
    }}}
    fv.poi_a.sync(MCP::Hst2Dev);
    maxdiag = FalmMV::MaxDiag(fv.poi_a, falm.cpm, block);
    FalmMV::ScaleMatrix(fv.poi_a, 1.0 / maxdiag, block);
}

void init(int &argc, char **&argv) {
    falm.env_init(argc, argv);
    falm.parse_settings("setup.json");
    falm.computation_init({{falm.cpm.size, 1, 1}}, GuideCell);
    falm.print_info(TERMINAL_OUTPUT_RANK);
    u_previous.alloc(falm.fv.u.shape[0], falm.fv.u.shape[1], HDC::Device, "previous velocity");

    for (int i = 0; i < 6; i ++) cudaStreamCreate(&facestream[i]);
    streams = facestream;
    // streams = nullptr;
    if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
        printf("using streams %p\n", streams);
    }

    REAL u_inflow = falm.params["inflow"]["velocity"].get<REAL>();
    Matrix<REAL> &u = falm.fv.u;
    for (INT i = 0; i < u.shape[0]; i ++) {
        u(i, 0) = u_inflow;
        u(i, 1) = u(i, 2) = 0.0;
    }
    u.sync(MCP::Hst2Dev);
    FalmBasicVar &fv = falm.fv;
    falm.falmCfd.UtoUU(fv.u, fv.uu, fv.kx, fv.ja, falm.cpm, block, streams);
    falm.falmCfd.SGS(fv.u, fv.nut, fv.xyz, fv.kx, fv.ja, falm.cpm, block, streams);

    make_poisson_coefficient_matrix();
    // alm.init(falm.cpm, falm.params["turbine"], falm.workdir);
    // alm.print_info(falm.cpm.rank == TERMINAL_OUTPUT_RANK);

    aalm.init(falm.workdir, falm.params["turbine"], falm.cpm);
    for (int rank = 0; rank < falm.cpm.size; rank ++) {
        if (rank == falm.cpm.rank) {
            printf("rank %d\n", rank);
            aalm.print_info(rank);
            fflush(stdout);
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }
}

REAL main_loop(Alm::AlmHandler &alm, STREAM *s) {
    FalmBasicVar &fv = falm.fv;
    u_previous.copy(fv.u, HDC::Device);
    profiler.startEvent("ALM");
    // alm.SetALMFlag(fv.xyz, falm.gettime(), falm.cpm, block);
    // alm.ALM(fv.u, fv.xyz, fv.ff, falm.gettime(), falm.cpm, block);
    alm.Alm(falm.baseMesh.x, falm.baseMesh.y, falm.baseMesh.z, fv.u, fv.ff, falm.gettime(), block);
    profiler.endEvent("ALM");

    FalmCFD &fcfd = falm.falmCfd;

    profiler.startEvent("U*");
    fcfd.FSPseudoU(u_previous, fv.u, fv.uu, fv.u, fv.nut, fv.kx, fv.g, fv.ja, fv.ff, falm.dt, falm.cpm, block, s, 1);
    profiler.endEvent("U*");

    profiler.startEvent("U* interpolation");
    fcfd.UtoUU(fv.u, fv.uu, fv.kx, fv.ja, falm.cpm, block, s);
    profiler.endEvent("U* interpolation");

    profiler.startEvent("div(U*)/dt");
    fcfd.MACCalcPoissonRHS(fv.uu, fv.poi_rhs, fv.ja, falm.dt, falm.cpm, block, maxdiag);
    profiler.endEvent("div(U*)/dt");

    FalmEq &feq = falm.falmEq;
    profiler.startEvent("div(grad p)=div(U*)/dt");
    feq.Solve(fv.poi_a, fv.p, fv.poi_rhs, fv.poi_res, falm.cpm, block, s);
    profiler.endEvent("div(grad p)=div(U*)/dt");

    profiler.startEvent("p BC");
    pbc(fv.p, falm.cpm, s);
    profiler.endEvent("p BC");

    profiler.startEvent("U = U* - dt grad(p)");
    fcfd.ProjectP(fv.u, fv.u, fv.uu, fv.uu, fv.p, fv.kx, fv.g, falm.dt, falm.cpm, block, s);
    profiler.endEvent("U = U* - dt grad(p)");

    profiler.startEvent("U BC");
    ubc(fv.u, u_previous, fv.xyz, falm.dt, falm.cpm, s);
    profiler.endEvent("U BC");

    profiler.startEvent("Nut");
    fcfd.SGS(fv.u, fv.nut, fv.xyz, fv.kx, fv.ja, falm.cpm, block, s);
    profiler.endEvent("Nut");

    profiler.startEvent("||div(U)||");
    fcfd.Divergence(fv.uu, fv.divergence, fv.ja, falm.cpm, block);
    REAL divnorm = FalmMV::EuclideanNormSq(fv.divergence, falm.cpm, block);
    profiler.endEvent("||div(U)||");

    falm.TAvg();
    falm.outputUVWP();

    return divnorm;
}

void finalize() {
    for (int i = 0; i < 6; i ++) cudaStreamDestroy(facestream[i]);
    u_previous.release();
    falm.env_finalize();
    // alm.finalize();
    aalm.finalize();
}

int main(int argc, char **argv) {
    init(argc, argv);

    if (argc > 1) {
        std::string arg(argv[1]);
        if (arg == "info") {
            goto FIN;
        }
    }

    printf("\n");
    if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
        printf("maxdiag = %e %e\n", maxdiag, maxdiag2);
    }
    profiler.startEvent("global loop");
    for (falm.it = 1; falm.it <= falm.maxIt; falm.it ++) {
        REAL divnorm = sqrt(main_loop(aalm, streams)) / PRODUCT3(falm.cpm.pdm_list[falm.cpm.rank].shape - INT(2 * falm.cpm.gc));
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

        // falm.fv.ff.sync(MCP::Dev2Hst);
        // falm.fv.xyz.sync(MCP::Dev2Hst);
        // // alm.alm_flag.sync(MCP::Dev2Hst);
        // FILE *csv = fopen("data/alm.csv", "w");
        // fprintf(csv, "x,y,z,fx,fy,fz\n");
        // const INT3 &shape = falm.cpm.pdm_list[falm.cpm.rank].shape;
        // for (INT k = 0; k < shape[2]; k ++) {
        // for (INT j = 0; j < shape[1]; j ++) {
        // for (INT i = 0; i < shape[0]; i ++) {
        //     INT idx = IDX(i, j, k, shape);
        //     Matrix<REAL> &ff = falm.fv.ff;
        //     Matrix<REAL> &x  = falm.fv.xyz;
        //     // Matrix<INT>  &flag = alm.alm_flag;
        //     fprintf(csv, "%lf,%lf,%lf,%e,%e,%e\n", x(idx,0), x(idx,1), x(idx,2), ff(idx,0), ff(idx,1), ff(idx,2));
        // }}}
        // fclose(csv);
    }
    /* std::string fname = "data/alm_rank" + std::to_string(falm.cpm.rank) + ".csv";
    falm.fv.ff.sync(MCP::Dev2Hst);
    falm.fv.xyz.sync(MCP::Dev2Hst);
    FILE *csv = fopen(fname.c_str(), "w");
    fprintf(csv, "x,y,z,fx,fy,fz\n");
    const INT3 &shape = falm.cpm.pdm_list[falm.cpm.rank].shape;
    for (INT k = 0; k < shape[2]; k ++) {
    for (INT j = 0; j < shape[1]; j ++) {
    for (INT i = 0; i < shape[0]; i ++) {
        INT idx = IDX(i, j, k, shape);
        Matrix<REAL> &ff = falm.fv.ff;
        Matrix<REAL> &x  = falm.fv.xyz;
        // Matrix<INT>  &flag = alm.alm_flag;
        fprintf(csv, "%lf,%lf,%lf,%e,%e,%e\n", x(idx,0), x(idx,1), x(idx,2), ff(idx,0), ff(idx,1), ff(idx,2));
    }}}
    fclose(csv); */
    finalize();
    return 0;
}