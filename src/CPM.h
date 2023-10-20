#ifndef FALM_CPM_H
#define FALM_CPM_H

#include "CPML2v2.h"
#include "./vcdm/VCDM.h"

namespace Falm {

template<typename T>
void setVcdm(CPMBase &cpm, Vcdm::VCDM<T> vcdm, Vcdm::doublex3 gRegion, Vcdm::doublex3 gOrigin = {0, 0, 0}) {
    vcdm.dfiDomain.globalOrigin = gOrigin;
    vcdm.dfiDomain.globalRegion = gRegion;
    vcdm.dfiDomain.globalVoxel  = Vcdm::intx3{
        cpm.global.x - cpm.gc*2,
        cpm.global.y - cpm.gc*2,
        cpm.global.z - cpm.gc*2
    };
    vcdm.dfiDomain.globalDivision = Vcdm::intx3{
        cpm.shape.x, cpm.shape.y, cpm.shape.z
    };

    vcdm.dfiProc = std::vector<Vcdm::VcdmRank>(cpm.size, Vcdm::VcdmRank());
    for (INT k = 0; k < cpm.shape.z; k ++) {
    for (INT j = 0; j < cpm.shape.y; j ++) {
    for (INT i = 0; i < cpm.shape.x; i ++) {
        int rank = IDX(i, j, k, cpm.shape);
        vcdm.dfiProc[rank].rank = rank;
        vcdm.dfiProc[rank].voxelSize = Vcdm::intx3{
            cpm.pdm_list[rank].shape.x - cpm.gc*2,
            cpm.pdm_list[rank].shape.y - cpm.gc*2,
            cpm.pdm_list[rank].shape.z - cpm.gc*2
        };
    }}}
}

}

#endif