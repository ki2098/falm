#include <math.h>
#include "../../src/falm.h"
#include "bc.h"
#include "../../src/profiler.h"

Cprof::cprof_Profiler profiler;

using namespace Falm;

const int TERMINAL_OUTPUT_RANK = 0;

FalmCore falm;
Real maxDiag;
Matrix<Real> uPrevious;
const dim3 blockSize{8, 8, 8};
Stream faceStreams[CPM::NFACE];
Stream *streams;
Real uInflow;

void makePoissonCoefficientMatrix() {
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
        // printf("%lf %lf %lf %lf %lf %lf %lf\n", ac, aw, ae, as, an, ab, at);
    }}}
    fv.poi_a.sync(MCP::Hst2Dev);
    maxDiag = FalmMV::MaxDiag(fv.poi_a, falm.cpm, blockSize);
    FalmMV::ScaleMatrix(fv.poi_a, 1.0 / maxDiag, blockSize);
}

void init(int &argc, char **&argv) {
    falm.env_init(argc, argv);
    falm.parse_settings("setup.json");
    falm.computation_init({{falm.cpm.size, 1, 1}}, GuideCell);
    falm.print_info(TERMINAL_OUTPUT_RANK);
    uPrevious.alloc(falm.fv.u.shape[0], falm.fv.u.shape[1], HDC::Device, "previous velocity");

    for (auto &stream : faceStreams) {
        cudaStreamCreate(&stream);
    }
    streams = faceStreams;

    for (int i = 0; i < CPM::NFACE; i ++) {
        if (faceStreams[i] == nullptr) {
            printf("stream %d is not properly created\n", i);
        }
    }

    if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
        printf("using streams %p\n", streams);
    }

    uInflow = falm.params["inflow"]["velocity"].get<Real>();
    if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
        printf("inflow velocity = %lf\n", uInflow);
    }
    Matrix<Real> &u = falm.fv.u;
    for (Int i = 0; i < u.shape[0]; i ++) {
        u(i, 0) = uInflow;
        u(i, 1) = 0.0;
        u(i, 2) = 0.0;
    }
    u.sync(MCP::Hst2Dev);
    FalmBasicVar &fv = falm.fv;
    falm.falmCfd.UtoUU(fv.u, fv.uu, fv.kx, fv.ja, falm.cpm, blockSize, streams);
    falm.falmCfd.SGS(fv.u, fv.nut, fv.xyz, fv.kx, fv.ja, falm.cpm, blockSize, streams);

    makePoissonCoefficientMatrix();

}

Real mainLoop() {
    FalmBasicVar &fv = falm.fv;
    uPrevious.copy(fv.u, HDC::Device);

    FalmCFD &cfd = falm.falmCfd;
    cfd.FSPseudoU(uPrevious, fv.u, fv.uu, fv.u, fv.nut, fv.kx, fv.g, fv.ja, fv.ff, falm.dt, falm.cpm, blockSize, streams);

    cfd.UtoUU(fv.u, fv.uu, fv.kx, fv.ja, falm.cpm, blockSize, streams);

    cfd.MACCalcPoissonRHS(fv.uu, fv.poi_rhs, fv.ja, falm.dt, falm.cpm, blockSize, maxDiag);

    FalmEq &eq = falm.falmEq;
    eq.Solve(fv.poi_a, fv.p, fv.poi_rhs, fv.poi_res, falm.cpm, blockSize, streams);

    pbc(fv.p, falm.cpm, streams);

    cfd.ProjectP(fv.u, fv.u, fv.uu, fv.uu, fv.p, fv.kx, fv.g, falm.dt, falm.cpm, blockSize, streams);

    ubc(fv.u, uPrevious, fv.xyz, uInflow, falm.dt, falm.cpm, streams);

    cfd.SGS(fv.u, fv.nut, fv.xyz, fv.kx, fv.ja, falm.cpm, blockSize, streams);

    cfd.Divergence(fv.uu, fv.divergence, fv.ja, falm.cpm, blockSize);
    Real divergence = FalmMV::EuclideanNormSq(fv.divergence, falm.cpm, blockSize);

    falm.TAvg();
    
    falm.outputUVWP();

    return divergence;
}

void finalize() {
    for (auto &stream : faceStreams) {
        cudaStreamDestroy(stream);
    }
    uPrevious.release();
    falm.env_finalize();
}

int main(int argc, char **argv) {
    init(argc, argv);

    printf("\n");
    if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
        printf("max diag = %e\n", maxDiag);
    }

    profiler.startEvent("global loop");
    for (falm.it = 1; falm.it <= falm.maxIt; falm.it ++) {
        Real divergence = sqrt(mainLoop()) / PRODUCT3(falm.cpm.pdm_list[falm.cpm.rank].shape - Int(2 * falm.cpm.gc));
        if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
            printf("%8d %12.5e, %12.5e, %3d, %12.5e\n", falm.it, falm.gettime(), divergence, falm.falmEq.it, falm.falmEq.err);
            fflush(stdout);
        }
    }
    profiler.endEvent("global loop");
    printf("\n");

    if (falm.cpm.rank == TERMINAL_OUTPUT_RANK) {
        profiler.output();
        printf("\n");
    }

    finalize();
    return 0;
}