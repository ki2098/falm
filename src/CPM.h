#ifndef FALM_CPM_H
#define FALM_CPM_H

#include "CPMComm.h"
#include "./vcdm/VCDM.h"

namespace Falm {

template<typename T>
void setVcdm(CPM &cpm, Vcdm::VCDM<T> &vcdm, Vcdm::double3 gRegion, Vcdm::double3 gOrigin = {{0, 0, 0}}) {
    vcdm.dfiFinfo.gc = cpm.gc;
    vcdm.dfiDomain.globalOrigin = gOrigin;
    vcdm.dfiDomain.globalRegion = gRegion;
    vcdm.dfiDomain.globalVoxel  = Vcdm::int3{{
        cpm.global.shape[0] - cpm.gc*2,
        cpm.global.shape[1] - cpm.gc*2,
        cpm.global.shape[2] - cpm.gc*2
    }};
    vcdm.dfiDomain.globalDivision = Vcdm::int3{{
        cpm.shape[0], cpm.shape[1], cpm.shape[2]
    }};

    vcdm.dfiMPI.size = cpm.size;

    vcdm.dfiProc = std::vector<Vcdm::VcdmRank>(cpm.size, Vcdm::VcdmRank());
    for (Int k = 0; k < cpm.shape[2]; k ++) {
    for (Int j = 0; j < cpm.shape[1]; j ++) {
    for (Int i = 0; i < cpm.shape[0]; i ++) {
        int rank = IDX(i, j, k, cpm.shape);
        Region &pdm = cpm.pdm_list[rank];
        Vcdm::VcdmRank &vproc = vcdm.dfiProc[rank];
        vproc.rank = rank;
        vproc.voxelSize = {{
            pdm.shape[0] - cpm.gc*2,
            pdm.shape[1] - cpm.gc*2,
            pdm.shape[2] - cpm.gc*2
        }};
        vproc.headIdx = {{
            pdm.offset[0] + 1,
            pdm.offset[1] + 1,
            pdm.offset[2] + 1
        }};
        vproc.tailIdx = {{
            vproc.headIdx[0] + vproc.voxelSize[0] - 1,
            vproc.headIdx[1] + vproc.voxelSize[1] - 1,
            vproc.headIdx[2] + vproc.voxelSize[2] - 1
        }};
    }}}
}

}

#endif
