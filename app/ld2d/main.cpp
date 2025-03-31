#include "../../src/falm.h"
#include "bc.h"

using namespace std;
using namespace Falm;

FalmCore falm;
Real maxdiag;
Stream facestream[6];
Stream *streams;

dim3 block{8, 8, 1};

void make_poisson_coefficient_matrix() {
    CPM &cpm = falm.cpm;
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region &global = cpm.global;
    FalmBasicVar &fv = falm.fv;
    Int gc = cpm.gc;
    for (Int k = cpm.gc; k < pdm.shape[2] - cpm.gc; k ++) {
    for (Int j = cpm.gc; j < pdm.shape[1] - cpm.gc; j ++) {
    for (Int i = cpm.gc; i < pdm.shape[0] - cpm.gc; i ++) {
        Int3 gijk = pdm.offset + Int3{{i, j, k}};
        Real ac, ae, aw, an, as;
        ac = ae = aw = an = as = 0.0;
        Int idxcc = IDX(i  , j  , k  , pdm.shape);
        Int idxe1 = IDX(i+1, j  , k  , pdm.shape);
        Int idxw1 = IDX(i-1, j  , k  , pdm.shape);
        Int idxn1 = IDX(i  , j+1, k  , pdm.shape);
        Int idxs1 = IDX(i  , j-1, k  , pdm.shape);
        Real gxcc  =  fv.g(idxcc, 0);
        Real gxe1  =  fv.g(idxe1, 0);
        Real gxw1  =  fv.g(idxw1, 0);
        Real gycc  =  fv.g(idxcc, 1);
        Real gyn1  =  fv.g(idxn1, 1);
        Real gys1  =  fv.g(idxs1, 1);
        Real jacob =  fv.ja(idxcc);
        Real coefficient;
        coefficient = 0.5 * (gxcc + gxe1) / jacob;
        if (gijk[0] < global.shape[0] - gc - 1) {
            ac -= coefficient;
            ae  = coefficient;
        }
        coefficient = 0.5 * (gxcc + gxw1) / jacob;
        if (gijk[0] > gc) {
            ac -= coefficient;
            aw  = coefficient;
        }
        coefficient = 0.5 * (gycc + gyn1) / jacob;
        if (gijk[1] < global.shape[1] - gc - 1) {
            ac -= coefficient;
            an  = coefficient;
        }
        coefficient = 0.5 * (gycc + gys1) / jacob;
        if (gijk[1] > gc) {
            ac -= coefficient;
            as  = coefficient;
        }
        fv.poi_a(idxcc, 0) = ac;
        fv.poi_a(idxcc, 1) = aw;
        fv.poi_a(idxcc, 2) = ae;
        fv.poi_a(idxcc, 3) = as;
        fv.poi_a(idxcc, 4) = an;
    }}}
    fv.poi_a.sync(MCP::Hst2Dev);
    maxdiag = FalmMV::MaxDiag(fv.poi_a, falm.cpm, block);
    FalmMV::ScaleMatrix(fv.poi_a, 1.0 / maxdiag, block);
}

void init(int &argc, char **&argv) {
    falm.env_init(argc, argv);
    falm.parse_settings("setup.json");
    Int3 mpishape{{atoi(argv[1]), atoi(argv[2]), 1}};
    falm.computation_init(mpishape, GuideCell);
    // falm.cpm.use_cuda_aware_mpi = true;
    falm.print_info();

    for (int i = 0; i < 6; i ++) cudaStreamCreate(&facestream[i]);
    streams = facestream;
    make_poisson_coefficient_matrix();
}

void csv_output() {
    char tmp[100];
    sprintf(tmp, "%s_rank%d.csv%d", falm.outputPrefix.c_str(), falm.cpm.rank, falm.it);
    std::string filename = falm.wpath(std::string(tmp));
    FILE *csvfile = fopen(filename.c_str(), "w");
    fprintf(csvfile, "x,y,z,u,v,w,p\n");
    FalmBasicVar &fv = falm.fv;
    fv.u.sync(MCP::Dev2Hst);
    fv.p.sync(MCP::Dev2Hst);
    CPM &cpm = falm.cpm;
    Int gc = cpm.gc;
    Int3 shape = cpm.pdm_list[cpm.rank].shape;

    for (Int k = gc; k < shape[2] - gc; k ++) {
    for (Int j = gc; j < shape[1] - gc; j ++) {
    for (Int i = gc; i < shape[0] - gc; i ++) {
        Int idx = IDX(i, j, k, shape);
        Real x = fv.xyz(idx, 0);
        Real y = fv.xyz(idx, 1);
        Real z = fv.xyz(idx, 2);
        Real u = fv.u(idx, 0);
        Real v = fv.u(idx, 1);
        Real w = fv.u(idx, 2);
        Real p = fv.p(idx);
        fprintf(csvfile, "%e,%e,%e,%e,%e,%e,%e\n", x, y, z, u, v, w, p);
    }}}
    fclose(csvfile);
}

Real main_loop(Stream *s) {
    FalmBasicVar &fv = falm.fv;
    FalmCFD &fcfd = falm.falmCfd;
    FalmEq &feq = falm.falmEq;

    fcfd.FSPseudoU(fv.u, fv.u, fv.uu, fv.u, fv.nut, fv.kx, fv.g, fv.ja, fv.ff, falm.dt, falm.cpm, block, s, 1);
    fcfd.UtoUU(fv.u, fv.uu, fv.kx, fv.ja, falm.cpm, block, s);
    uubc(fv.uu, falm.cpm, s);
    fcfd.MACCalcPoissonRHS(fv.uu, fv.poi_rhs, fv.ja, falm.dt, falm.cpm, block, maxdiag);

    feq.Solve(fv.poi_a, fv.p, fv.poi_rhs, fv.poi_res, falm.cpm, block, s);
    pbc(fv.p, falm.cpm, s);
    copy_z5(fv.p, falm.cpm);

    fcfd.ProjectP(fv.u, fv.u, fv.uu, fv.uu, fv.p, fv.kx, fv.g, falm.dt, falm.cpm, block, s);
    ubc(fv.u, falm.cpm, s);
    copy_z5(fv.u, falm.cpm);

    fcfd.SGS(fv.u, fv.nut, fv.xyz, fv.kx, fv.ja, falm.cpm, block, s);
    copy_z5(fv.nut, falm.cpm);

    fcfd.Divergence(fv.uu, fv.divergence, fv.ja, falm.cpm, block);

    return FalmMV::EuclideanNormSq(fv.divergence, falm.cpm, block);
}

void finalize() {
    for (int i = 0; i < 6; i ++) cudaStreamDestroy(facestream[i]);
    falm.env_finalize();
}

int main(int argc, char **argv) {
    init(argc, argv);

    falm.it = 0;
    printf("\n");
    if (falm.cpm.rank == 0) {
        printf("maxdiag = %e\n", maxdiag);
    }
    while (falm.it < falm.maxIt) {
        Real dvgn = sqrt(main_loop(streams) / PRODUCT3(falm.cpm.pdm_list[falm.cpm.rank].shape - Int(2 * falm.cpm.gc)));
        falm.it ++;
        if (falm.cpm.rank == 0) {
            printf("\r%8d %12.5e, %12.5e, %3d, %12.5e", falm.it, falm.gettime(), dvgn, falm.falmEq.it, falm.falmEq.err);
            fflush(stdout);
        }
        if (falm.it % falm.outputIntervalIt == 0) {
            falm.outputUVWP();
        }
    }
    printf("\n");

    finalize();

    return 0;
}