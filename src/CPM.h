#ifndef FALM_CPM_H
#define FALM_CPM_H

#include "CPML2v2.h"
#include "./vcdm/VCDM.h"

namespace Falm {

template<typename T>
void setVcdm(CPMBase &cpm, Vcdm::VCDM<T> &vcdm, Vcdm::double3 gRegion, Vcdm::double3 gOrigin = {0, 0, 0}) {
    vcdm.dfiFinfo.gc = cpm.gc;
    vcdm.dfiDomain.globalOrigin = gOrigin;
    vcdm.dfiDomain.globalRegion = gRegion;
    vcdm.dfiDomain.globalVoxel  = Vcdm::int3{
        cpm.global.shape.x - cpm.gc*2,
        cpm.global.shape.y - cpm.gc*2,
        cpm.global.shape.z - cpm.gc*2
    };
    vcdm.dfiDomain.globalDivision = Vcdm::int3{
        cpm.shape.x, cpm.shape.y, cpm.shape.z
    };

    vcdm.dfiMPI.size = cpm.size;

    vcdm.dfiProc = std::vector<Vcdm::VcdmRank>(cpm.size, Vcdm::VcdmRank());
    for (INT k = 0; k < cpm.shape.z; k ++) {
    for (INT j = 0; j < cpm.shape.y; j ++) {
    for (INT i = 0; i < cpm.shape.x; i ++) {
        int rank = IDX(i, j, k, cpm.shape);
        Region &pdm = cpm.pdm_list[rank];
        Vcdm::VcdmRank &vproc = vcdm.dfiProc[rank];
        vproc.rank = rank;
        vproc.voxelSize = {
            pdm.shape.x - cpm.gc*2,
            pdm.shape.y - cpm.gc*2,
            pdm.shape.z - cpm.gc*2
        };
        vproc.headIdx = {
            pdm.offset.x + 1,
            pdm.offset.y + 1,
            pdm.offset.z + 1
        };
        vproc.tailIdx = {
            vproc.headIdx.x + vproc.voxelSize.x - 1,
            vproc.headIdx.y + vproc.voxelSize.y - 1,
            vproc.headIdx.z + vproc.voxelSize.z - 1
        };
    }}}
}

}

#endif