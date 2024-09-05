#include <math.h>
#include <fstream>
#include "../../src/falm.h"
#include "bc.h"
#include "../../src/profiler.h"

using namespace Falm;

#define TERMINAL_OUTPUT_RANK 0

FalmCore falm;
Alm::AlmHandler alm;

void init(int &argc, char **&argv) {
    falm.env_init(argc, argv);
    falm.parse_settings("setup.json");
    falm.computation_init({{falm.cpm.size, 1, 1,}}, GuideCell);
    falm.print_info(TERMINAL_OUTPUT_RANK);

    alm.init(falm.workdir, falm.params["turbine"], falm.params["turbine"]["apFile"], falm.cpm);
    alm.print_info(TERMINAL_OUTPUT_RANK);

    REAL u_inflow = falm.params["inflow"]["velocity"].get<REAL>();
    Matrix<REAL> &u = falm.fv.u;
    for (INT i = 0; i < u.shape[0]; i ++) {
        u(i, 0) = u_inflow;
        u(i, 1) = u(i, 2) = 0.0;
    }
    u.sync(MCP::Hst2Dev);
}

void finalize() {
    falm.env_finalize();
    alm.finalize();
}

int main(int argc, char **argv) {
    init(argc, argv);
    CPM &cpm = falm.cpm;
    FalmBasicVar &fv = falm.fv;
    FalmBaseMesh &mesh = falm.baseMesh;
    const Region &pdm = cpm.pdm_list[cpm.rank];
    const INT3 &shape = pdm.shape;

    Matrix<REAL> phi(shape, 1, HDC::HstDev);

    alm.Alm(mesh.x, mesh.y, mesh.z, fv.u, fv.ff, 0);
    alm.DryDistribution(mesh.x, mesh.y, mesh.z, phi);

    phi.sync(MCP::Dev2Hst);
    fv.ff.sync(MCP::Dev2Hst);

    FILE *csv = fopen("data/drydist.csv", "w");
    fprintf(csv, "x,y,z,phi,f1,f2,f3\n");
    for (INT k = 0; k < shape[2]; k ++) {
    for (INT j = 0; j < shape[1]; j ++) {
    for (INT i = 0; i < shape[0]; i ++) {
        INT idx = IDX(i,j,k,shape);
        Matrix<REAL> &x = mesh.x;
        Matrix<REAL> &y = mesh.y;
        Matrix<REAL> &z = mesh.z;
        fprintf(csv, "%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", x(i), y(j), z(k), phi(idx), fv.ff(idx,0), fv.ff(idx,1), fv.ff(idx,2));
    }}}
    fclose(csv);

    phi.release();

    finalize();
}