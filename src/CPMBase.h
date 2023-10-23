#ifndef FALM_CPMBASE_H
#define FALM_CPMBASE_H

#include <vector>
#include <assert.h>
#include "region.h"

namespace Falm {

class CPMBase {
public:
    int neighbour[6];
    INTx3   shape;
    INTx3     idx;
    int      rank;
    int      size;
    bool use_cuda_aware_mpi;
    INT        gc;
    Region global;
    std::vector<Region> pdm_list;

    void initPartition(INTx3 gShape, INT guideCell, int mpi_rank = 0, int mpi_size = 1, INTx3 mpi_shape = {1,1,1}) {
        assert(mpi_size == PRODUCT3(mpi_shape));

        shape  = mpi_shape;
        rank   = mpi_rank;
        size   = mpi_size;
        initNeighbour();

        gc     = guideCell;
        global = Region(
            INTx3{gShape.x + gc*2, gShape.y + gc*2, gShape.z + gc*2},
            INTx3{0, 0, 0}
        );
        pdm_list = std::vector<Region>(size, Region());
        for (INT k = 0; k < shape.z; k ++) {
        for (INT j = 0; j < shape.y; j ++) {
        for (INT i = 0; i < shape.x; i ++) {
            INT ox = 0, oy = 0, oz = 0;
            for (INT __x = 0; __x < i; __x ++) {
                ox += dim_division(gShape.x, shape.x, __x);
            }
            for (INT __y = 0; __y < j; __y ++) {
                oy += dim_division(gShape.y, shape.y, __y);
            }
            for (INT __z = 0; __z < k; __z ++) {
                oz += dim_division(gShape.z, shape.z, __z);
            }
            pdm_list[IDX(i, j, k, shape)] = Region(
                INTx3{
                    dim_division(gShape.x, shape.x, i) + gc*2,
                    dim_division(gShape.y, shape.y, j) + gc*2,
                    dim_division(gShape.z, shape.z, k) + gc*2
                },
                INTx3{ox, oy, oz}
            );
        }}}

    }

    void set6Region(INTx3 &inner_shape, INTx3 &inner_offset, INTx3 *boundary_shape, INTx3 *boundary_offset, INT thick, const Region &map) {
        inner_shape = map.shape;
        inner_offset = map.offset;
        if (neighbour[0] >= 0) {
            boundary_shape[0]  = {thick, inner_shape.y, inner_shape.z};
            boundary_offset[0] = {inner_offset.x + inner_shape.x - thick, inner_offset.y, inner_offset.z};
            inner_shape.x -= thick;
        }
        if (neighbour[1] >= 0) {
            boundary_shape[1]  = {thick, inner_shape.y, inner_shape.z};
            boundary_offset[1] = {inner_offset.x, inner_offset.y, inner_offset.z};
            inner_shape.x  -= thick;
            inner_offset.x += thick; 
        }
        if (neighbour[2] >= 0) {
            boundary_shape[2]  = {inner_shape.x, thick, inner_shape.z};
            boundary_offset[2] = {inner_offset.x, inner_offset.y + inner_shape.y - thick, inner_offset.z};
            inner_shape.y -= thick;
        }
        if (neighbour[3] >= 0) {
            boundary_shape[3]  = {inner_shape.x, thick, inner_shape.z};
            boundary_offset[3] = {inner_offset.x, inner_offset.y, inner_offset.z};
            inner_shape.y  -= thick;
            inner_offset.y += thick;
        }
        if (neighbour[4] >= 0) {
            boundary_shape[4]  = {inner_shape.x, inner_shape.y, thick};
            boundary_offset[4] = {inner_offset.x, inner_offset.y, inner_offset.z + inner_shape.z - thick};
            inner_shape.z -= thick;
        }
        if (neighbour[5] >= 0) {
            boundary_shape[5]  = {inner_shape.x, inner_shape.y, thick};
            boundary_offset[5] = {inner_offset.x, inner_offset.y, inner_offset.z};
            inner_shape.z  -= thick;
            inner_offset.z += thick;
        }
    }

    // void setDefaultRegions(INTx3 &inner_shape, INTx3 &inner_offset, INTx3 *boundary_shape, INTx3 *boundary_offset, INT thick, Mapper &pdm) {
    //     Mapper __map(pdm, gc);
    //     setNonDefaultRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, thick, __map);
    // }

    bool validNeighbour(INT fid) {
        return (neighbour[fid] >= 0);
    }

protected:
    INT dim_division(INT dim_size, INT n_division, INT id) {
        INT p_dim_size = dim_size / n_division;
        if (id < dim_size % n_division) {
            p_dim_size ++;
        }
        return p_dim_size;
    }

    void initNeighbour() {
        int __rank = rank;
        INT i, j, k;
        k = __rank / (shape.x * shape.y);
        __rank = __rank % (shape.x * shape.y);
        j = __rank / shape.x;
        i = __rank % shape.x;
        idx.x = i;
        idx.y = j;
        idx.z = k;
        neighbour[0] = IDX(i + 1, j, k, shape);
        neighbour[1] = IDX(i - 1, j, k, shape);
        neighbour[2] = IDX(i, j + 1, k, shape);
        neighbour[3] = IDX(i, j - 1, k, shape);
        neighbour[4] = IDX(i, j, k + 1, shape);
        neighbour[5] = IDX(i, j, k - 1, shape);
        if (i == shape.x - 1) {
            neighbour[0] = - 1;
        }
        if (i == 0) {
            neighbour[1] = - 1;
        }
        if (j == shape.y - 1) {
            neighbour[2] = - 1;
        }
        if (j == 0) {
            neighbour[3] = - 1;
        }
        if (k == shape.z - 1) {
            neighbour[4] = - 1;
        }
        if (k == 0) {
            neighbour[5] = - 1;
        }
    }

};

}

#endif