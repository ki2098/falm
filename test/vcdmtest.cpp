#include <cmath>
#include "../src/vcdm/VCDM.h"
#include "../src/typedef.h"
#include "__vcdm.h"

using namespace Vcdm;

int main() {
    VCDM<Falm::REAL> vcdm;

    __VCDM __vcdm;
    std::string str = __vcdm.makefilename("pref");
    printf("%s\n", str.c_str());

    vcdm.setPath("data", "velocity");

    vcdm.dfiDomain.globalOrigin   = {0.0, 0.0, 0.0};
    vcdm.dfiDomain.globalRegion   = {1.0, 1.0, 1.0};
    vcdm.dfiDomain.globalVoxel    = {128, 128, 128};
    vcdm.dfiDomain.globalDivision = {2  , 2  , 2  };

    vcdm.dfiMPI.size              = 8;

    intx3 &division = vcdm.dfiDomain.globalDivision;
    for (int k = 0; k < division.z; k ++) {
        for (int j = 0; j < division.y; j ++) {
            for (int i = 0; i < division.x; i ++) {
                VcdmRank rank;
                rank.rank = i + j * division.x + k * division.x * division.y;
                rank.voxelSize = {64, 64, 64};
                rank.headIdx   = {i * 64 + 1 , j * 64 + 1 , k * 64 + 1 };
                rank.tailIdx   = {i * 64 + 64, j * 64 + 64, k * 64 + 64};
                vcdm.dfiProc.push_back(rank);
            }
        }
    }

    vcdm.writeProcDfi();

    vcdm.dfiFinfo.gc = 2;
    // vcdm.dfiFinfo.rankPrefix = "_id";
    vcdm.dfiFinfo.varList = {"u", "v", "w"};

    VcdmSlice slice;
    slice.step = 0;
    slice.time = 0.0;
    slice.avgStep = 0;
    slice.avgTime = 0;
    slice.avgMode = false;
    slice.vectorMax = sqrt(1 + 1 + 1);
    slice.vectorMin = 0.0;
    for (int i = 0; i < 3; i ++) {
        slice.varMax.push_back(1.0);
        slice.varMin.push_back(0.0);
    }
    vcdm.timeSlice.push_back(slice);
    slice.varMax.clear();
    slice.varMin.clear();

    slice.step = 10;
    slice.time = 100.0;
    slice.avgStep = 0;
    slice.avgTime = 0;
    slice.avgMode = false;
    slice.vectorMax = sqrt(4 + 4 + 4);
    slice.vectorMin = 0.0;
    for (int i = 0; i < 3; i ++) {
        slice.varMax.push_back(2.0);
        slice.varMin.push_back(0.0);
    }
    vcdm.timeSlice.push_back(slice);

    vcdm.writeIndexDfi();

    std::string fname = vcdm.makeFilename(
        vcdm.dfiFinfo.fnameFormat,
        vcdm.dfiFinfo.prefix,
        vcdm.dfiFinfo.rankPrefix,
        vcdm.makeFileDataExt(),
        0,
        200
    );
    printf("%s\n", fname.c_str());

    return 0;
}