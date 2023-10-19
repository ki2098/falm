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
    Falm::Mapper       &global,
    Falm::Mapper       &pdm,
    Falm::CPMBase      &cpm
) {
    global = Falm::Mapper(
        Falm::INTx3{side_n_cell + (cpm.gc*2), side_n_cell + (cpm.gc*2), 1 + (cpm.gc*2)},
        Falm::INTx3{0, 0, 0}
    );

    Falm::INT ox = 0;
    for (Falm::INT i = 0; i < cpm.idx.x; i ++) {
        ox += dim_division(side_n_cell, cpm.shape.x, i);
    }
    Falm::INT oy = 0;
    for (Falm::INT j = 0; j < cpm.idx.y; j ++) {
        oy += dim_division(side_n_cell, cpm.shape.y, j);
    }
    Falm::INT oz = 0;

    pdm = Falm::Mapper(
        Falm::INTx3{
            dim_division(side_n_cell, cpm.shape.x, cpm.idx.x) + (cpm.gc*2),
            dim_division(side_n_cell, cpm.shape.y, cpm.idx.y) + (cpm.gc*2),
            1 + (cpm.gc*2)
        },
        Falm::INTx3{ox, oy, oz}
    );
}

}

#endif