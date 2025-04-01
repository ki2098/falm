#include <math.h>
#include <fstream>
#include "../../src/FalmCFD.h"
#include "../../src/FalmEq.h"
#include "../../src/rmcp/alm.h"
#include "../../src/falmath.h"

using namespace Falm;



int main(int argc, char **argv) {
    Matrix<double> gx[3];
    Matrix<float> u;
    Vcdm::VCDM<float> vcdm;
    Vcdm::VCDM<double> dvcdm;
    CPM cpm;
    std::string gridpath = ".";

    CPM_Init(&argc, &argv);

    std::ifstream xfile(gridpath + "/x.txt");
    std::ifstream yfile(gridpath + "/y.txt");
    std::ifstream zfile(gridpath + "/z.txt");

    Int3 nxyz;
    std::string line;
    std::getline(xfile, line);
    nxyz[0] = std::stoi(line);
    std::getline(yfile, line);
    nxyz[1] = std::stoi(line);
    std::getline(zfile, line);
    nxyz[2] = std::stoi(line);

    gx[0].alloc(nxyz[0], 1, HDC::Host);
    gx[1].alloc(nxyz[1], 1, HDC::Host);
    gx[2].alloc(nxyz[2], 1, HDC::Host);

    for (int i = 0; i < nxyz[0]; i ++) {
        std::getline(xfile, line);
        gx[0](i) = std::stod(line);
    }
    for (int j = 0; j < nxyz[1]; j ++) {
        std::getline(yfile, line);
        gx[1](j) = std::stod(line);
    }
    for (int k = 0; k < nxyz[2]; k ++) {
        std::getline(zfile, line);
        gx[2](k) = std::stod(line);
    }

    Vcdm::double3 pitch;
    for (int i = 0; i < 3; i ++) {
        pitch[i] = (gx[i](nxyz[i]-1) - gx[i](0)) / (nxyz[i] - 1);
    }
    Vcdm::double3 gregion;
    for (int i = 0; i < 3; i ++) {
        gregion[i] = pitch[i] * nxyz[i];
    }
    Vcdm::double3 gorigin;
    for (int i = 0; i < 3; i ++) {
        gorigin[i] = gx[i](0);
    }

    cpm.initPartition(nxyz, 0);
    vcdm.setPath(".", "uvw");
    setVcdm(cpm, vcdm, gregion, gorigin);
    vcdm.dfiFinfo.varList = {"u", "v", "w"};
    dvcdm.setPath(".", "uvw");
    setVcdm(cpm, dvcdm, gregion, gorigin);
    dvcdm.dfiFinfo.varList = {"u", "v", "w"};
    u.alloc(cpm.pdm_list[cpm.rank].shape, 3, HDC::Host, "uvw");

    Int3 shape = cpm.pdm_list[cpm.rank].shape;
    for (int i = 0; i < shape[0]; i ++) {
    for (int j = 0; j < shape[1]; j ++) {
    for (int k = 0; k < shape[2]; k ++) {
        Int idx = IDX(i, j, k, shape);
        Real _x = gx[0](i) * 2 * Pi;
        Real _y = gx[1](j) * 2 * Pi;
        Real _z = gx[2](k) * 2 * Pi;
        u(idx, 0) = cos(_x) * sin(_y) * sin(_z);
        u(idx, 1) = sin(_x) * cos(_y) * sin(_z);
        u(idx, 2) = sin(_x) * sin(_y) * cos(_z);
    }}}

    vcdm.writeSph(&u(0), 0, 3, 0, 0, 0, Vcdm::IdxType::IJKN);
    dvcdm.writeCrd(&gx[0](0), &gx[1](0), &gx[2](0), 0);

    xfile.close();
    yfile.close();
    zfile.close();
    
    CPM_Finalize();
}