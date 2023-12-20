#include <math.h>
#include <fstream>
#include "../../src/falm.h"

using namespace Falm;


json param;

INT3 idmax;
Matrix<REAL> xyz, kx, g, ja;
Matrix<REAL> x, y, z, hx, hy, hz;

FalmEq   eqsolver;
FalmCFD  cfdsolver;
FalmTime ftimer;

void initialize() {
    std::ifstream setup_file("setup.json");
    param = json::parse(setup_file);
    Mesher::build_mesh(".", param["mesh"]["controlVolumeCenter"].get<std::string>(), param["mesh"]["path"].get<std::string>(), "controlVolume.txt", GuideCell);

    std::ifstream cvfile("controlVolume.txt");
    cvfile >> idmax[0] >> idmax[1] >> idmax[2];
    x.alloc(idmax[0], 1, HDCType::Host);
    y.alloc(idmax[1], 1, HDCType::Host);
    z.alloc(idmax[2], 1, HDCType::Host);
    hx.alloc(idmax[0], 1, HDCType::Host);
    hy.alloc(idmax[1], 1, HDCType::Host);
    hz.alloc(idmax[2], 1, HDCType::Host);
    for (int i = 0; i < idmax[0]; i ++) {
        cvfile >> x(i) >> hx(i);
    }
    for (int j = 0; j < idmax[1]; j ++) {
        cvfile >> y(j) >> hy(j);
    }
    for (int k = 0; k < idmax[2]; k ++) {
        cvfile >> z(k) >> hz(k);
    }
    cvfile.close();

    xyz.alloc(idmax, 3, HDCType::Host, "control volume center");
    kx.alloc(idmax, 3, HDCType::Host, "d ksi / d x");
    g.alloc(idmax, 3, HDCType::Host, "metric tensor");
    ja.alloc(idmax, 1, HDCType::Host, "jacobian");
    for (int i = 0; i < idmax[0]; i ++) {
    for (int j = 0; j < idmax[1]; j ++) {
    for (int k = 0; k < idmax[2]; k ++) {
        INT idx = IDX(i, j, k, idmax);
        xyz(idx, 0)   = x(i);
        xyz(idx, 1)   = y(j);
        xyz(idx, 2)   = z(k);
        REAL3 pitch;
        pitch[0] = hx(i);
        pitch[1] = hy(j);
        pitch[2] = hz(k);
        REAL volume = PRODUCT3(pitch);
        REAL3 dkdx = {{1.0/pitch[0], 1.0/pitch[1], 1.0/pitch[2]}};
        for (int n = 0; n < 3; n ++) {
            g(idx, n) = volume * dkdx[n] * dkdx[n];
            kx(idx, n) = dkdx[n];
        }
        ja(idx) = volume;
    }}}
    xyz.sync(MCpType::Hst2Dev);
    kx.sync(MCpType::Hst2Dev);
    g.sync(MCpType::Hst2Dev);
    ja.sync(MCpType::Hst2Dev);

    ftimer.start_time         = 0;
    ftimer.end_time           = param["runtime"]["time"]["end"];
    ftimer.delta_time         = param["runtime"]["time"]["dt"];
    ftimer.timeavg_start_time = param["runtime"]["timeAvg"]["start"];
    ftimer.timeavg_end_time   = ftimer.end_time;
    ftimer.output_start_time  = ftimer.timeavg_start_time;
    ftimer.output_end_time    = ftimer.end_time;
    ftimer.output_interval    = param["runtime"]["output"]["interval"];

}

int main(int argc, char **argv) {
    
    
    


    
    

    

    return 0;
}