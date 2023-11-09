#ifndef _LID_CAVITY2D2_PARTITION_H_
#define _LID_CAVITY2D2_PARTITION_H_

#include "../../src/CPM.h"

namespace LidCavity2d2 {

static Falm::INT dim_division(Falm::INT dim_size, Falm::INT mpi_size, Falm::INT mpi_rank) {
    Falm::INT p_dim_size = dim_size / mpi_size;
    if (mpi_rank < dim_size % mpi_size) {
        p_dim_size ++;
    }
    return p_dim_size;
}

static void setPartition(
    Falm::REAL          side_lenth,
    Falm::INT           side_n_cell,
    Falm::Region       &global,
    Falm::Region       &pdm,
    Falm::CPM      &cpm
) {
    global = Falm::Region(
        Falm::INT3{side_n_cell + (cpm.gc*2), side_n_cell + (cpm.gc*2), 1 + (cpm.gc*2)},
        Falm::INT3{0, 0, 0}
    );

    Falm::INT ox = 0;
    for (Falm::INT i = 0; i < cpm.idx[0]; i ++) {
        ox += dim_division(side_n_cell, cpm.shape[0], i);
    }
    Falm::INT oy = 0;
    for (Falm::INT j = 0; j < cpm.idx[1]; j ++) {
        oy += dim_division(side_n_cell, cpm.shape[1], j);
    }
    Falm::INT oz = 0;

    pdm = Falm::Region(
        Falm::INT3{
            dim_division(side_n_cell, cpm.shape[0], cpm.idx[0]) + (cpm.gc*2),
            dim_division(side_n_cell, cpm.shape[1], cpm.idx[1]) + (cpm.gc*2),
            1 + (cpm.gc*2)
        },
        Falm::INT3{ox, oy, oz}
    );
}

}

#endif