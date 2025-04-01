#ifndef _LID_CAVITY2D2_PARTITION_H_
#define _LID_CAVITY2D2_PARTITION_H_

#include "../../src/CPM.h"

namespace LidCavity2d2 {

static Falm::Int dim_division(Falm::Int dim_size, Falm::Int mpi_size, Falm::Int mpi_rank) {
    Falm::Int p_dim_size = dim_size / mpi_size;
    if (mpi_rank < dim_size % mpi_size) {
        p_dim_size ++;
    }
    return p_dim_size;
}

static void setPartition(
    Falm::Real          side_lenth,
    Falm::Int           side_n_cell,
    Falm::Region       &global,
    Falm::Region       &pdm,
    Falm::CPM      &cpm
) {
    global = Falm::Region(
        Falm::Int3{side_n_cell + (cpm.gc*2), side_n_cell + (cpm.gc*2), 1 + (cpm.gc*2)},
        Falm::Int3{0, 0, 0}
    );

    Falm::Int ox = 0;
    for (Falm::Int i = 0; i < cpm.idx[0]; i ++) {
        ox += dim_division(side_n_cell, cpm.shape[0], i);
    }
    Falm::Int oy = 0;
    for (Falm::Int j = 0; j < cpm.idx[1]; j ++) {
        oy += dim_division(side_n_cell, cpm.shape[1], j);
    }
    Falm::Int oz = 0;

    pdm = Falm::Region(
        Falm::Int3{
            dim_division(side_n_cell, cpm.shape[0], cpm.idx[0]) + (cpm.gc*2),
            dim_division(side_n_cell, cpm.shape[1], cpm.idx[1]) + (cpm.gc*2),
            1 + (cpm.gc*2)
        },
        Falm::Int3{ox, oy, oz}
    );
}

}

#endif