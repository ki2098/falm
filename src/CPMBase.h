#ifndef FALM_CPMBASE_H
#define FALM_CPMBASE_H

#include <vector>
#include <assert.h>
#include "region.h"

namespace Falm {

class CPM {
public:
    static const INT XMINUS = 0;
    static const INT XPLUS  = 1;
    static const INT YMINUS = 2;
    static const INT YPLUS  = 3;
    static const INT ZMINUS = 4;
    static const INT ZPLUS  = 5;
    static const INT NFACE  = 6;

public:
    int neighbour[6];
    INT3   shape;
    INT3     idx;
    int      rank;
    int      size;
    bool use_cuda_aware_mpi;
    INT        gc;
    Region global;
    std::vector<Region> pdm_list;

    void initPartition(INT3 gShape, INT guideCell, int mpi_rank = 0, int mpi_size = 1, INT3 mpi_shape = {{1,1,1}}) {
        assert(mpi_size == PRODUCT3(mpi_shape));

        shape  = mpi_shape;
        rank   = mpi_rank;
        size   = mpi_size;
        initNeighbour();

        gc     = guideCell;
        global = Region(
            INT3{{gShape[0] + gc*2, gShape[1] + gc*2, gShape[2] + gc*2}},
            INT3{{0, 0, 0}}
        );
        pdm_list = std::vector<Region>(size, Region());
        for (INT k = 0; k < shape[2]; k ++) {
        for (INT j = 0; j < shape[1]; j ++) {
        for (INT i = 0; i < shape[0]; i ++) {
            INT ox = 0, oy = 0, oz = 0;
            for (INT __x = 0; __x < i; __x ++) {
                ox += dim_division(gShape[0], shape[0], __x);
            }
            for (INT __y = 0; __y < j; __y ++) {
                oy += dim_division(gShape[1], shape[1], __y);
            }
            for (INT __z = 0; __z < k; __z ++) {
                oz += dim_division(gShape[2], shape[2], __z);
            }
            pdm_list[IDX(i, j, k, shape)] = Region(
                INT3{{
                    dim_division(gShape[0], shape[0], i) + gc*2,
                    dim_division(gShape[1], shape[1], j) + gc*2,
                    dim_division(gShape[2], shape[2], k) + gc*2
                }},
                INT3{{ox, oy, oz}}
            );
        }}}

    }

    void set6Region(INT3 &inner_shape, INT3 &inner_offset, INT3 *boundary_shape, INT3 *boundary_offset, INT thick, const Region &map) const {
        inner_shape = map.shape;
        inner_offset = map.offset;
        if (neighbour[XPLUS] >= 0) {
            boundary_shape[XPLUS]  = {{thick, inner_shape[1], inner_shape[2]}};
            boundary_offset[XPLUS] = {{inner_offset[0] + inner_shape[0] - thick, inner_offset[1], inner_offset[2]}};
            inner_shape[0] -= thick;
        }
        if (neighbour[XMINUS] >= 0) {
            boundary_shape[XMINUS]  = {{thick, inner_shape[1], inner_shape[2]}};
            boundary_offset[XMINUS] = {{inner_offset[0], inner_offset[1], inner_offset[2]}};
            inner_shape[0]  -= thick;
            inner_offset[0] += thick; 
        }
        if (neighbour[YPLUS] >= 0) {
            boundary_shape[YPLUS]  = {{inner_shape[0], thick, inner_shape[2]}};
            boundary_offset[YPLUS] = {{inner_offset[0], inner_offset[1] + inner_shape[1] - thick, inner_offset[2]}};
            inner_shape[1] -= thick;
        }
        if (neighbour[YMINUS] >= 0) {
            boundary_shape[YMINUS]  = {{inner_shape[0], thick, inner_shape[2]}};
            boundary_offset[YMINUS] = {{inner_offset[0], inner_offset[1], inner_offset[2]}};
            inner_shape[1]  -= thick;
            inner_offset[1] += thick;
        }
        if (neighbour[ZPLUS] >= 0) {
            boundary_shape[ZPLUS]  = {{inner_shape[0], inner_shape[1], thick}};
            boundary_offset[ZPLUS] = {{inner_offset[0], inner_offset[1], inner_offset[2] + inner_shape[2] - thick}};
            inner_shape[2] -= thick;
        }
        if (neighbour[ZMINUS] >= 0) {
            boundary_shape[ZMINUS]  = {{inner_shape[0], inner_shape[1], thick}};
            boundary_offset[ZMINUS] = {{inner_offset[0], inner_offset[1], inner_offset[2]}};
            inner_shape[2]  -= thick;
            inner_offset[2] += thick;
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
        k = __rank / (shape[0] * shape[1]);
        __rank = __rank % (shape[0] * shape[1]);
        j = __rank / shape[0];
        i = __rank % shape[0];
        idx[0] = i;
        idx[1] = j;
        idx[2] = k;
        neighbour[XPLUS ] = IDX(i + 1, j, k, shape);
        neighbour[XMINUS] = IDX(i - 1, j, k, shape);
        neighbour[YPLUS ] = IDX(i, j + 1, k, shape);
        neighbour[YMINUS] = IDX(i, j - 1, k, shape);
        neighbour[ZPLUS ] = IDX(i, j, k + 1, shape);
        neighbour[ZMINUS] = IDX(i, j, k - 1, shape);
        if (i == shape[0] - 1) {
            neighbour[XPLUS] = - 1;
        }
        if (i == 0) {
            neighbour[XMINUS] = - 1;
        }
        if (j == shape[1] - 1) {
            neighbour[YPLUS] = - 1;
        }
        if (j == 0) {
            neighbour[YMINUS] = - 1;
        }
        if (k == shape[2] - 1) {
            neighbour[ZPLUS] = - 1;
        }
        if (k == 0) {
            neighbour[ZMINUS] = - 1;
        }
    }

};

}

#endif
