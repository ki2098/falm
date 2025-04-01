#ifndef FALM_CPMBASE_H
#define FALM_CPMBASE_H

#include <typeinfo>
#include <vector>
#include <assert.h>
#include "region.h"
#include "CPMBufMan.h"

namespace Falm {

class CPM {
public:
    static const Int XMINUS = 0;
    static const Int XPLUS  = 1;
    static const Int YMINUS = 2;
    static const Int YPLUS  = 3;
    static const Int ZMINUS = 4;
    static const Int ZPLUS  = 5;
    static const Int NFACE  = 6;

public:
    CpmBufMan bufman;
    int neighbour[6];
    Int3   shape;
    Int3     idx;
    int      rank;
    int      size;
    bool use_cuda_aware_mpi = false;
    Int        gc;
    Region global;
    std::vector<Region> pdm_list;
    std::vector<Int> pdshape[3], pdoffset[3];
    // STREAM *stream;

    void initPartition(Int3 gShape, Int guideCell, Int3 mpi_shape) {
        assert(size == PRODUCT3(mpi_shape));

        shape  = mpi_shape;

        initNeighbour();

        gc     = guideCell;
        Int gcgc = gc*2;
        global = Region(
            gShape,
            Int3{{0, 0, 0}}
        );
        for (int d = 0; d < 3; d ++) {
            pdshape[d] = std::vector<Int>(shape[d]);
            pdoffset[d] = std::vector<Int>(shape[d]);
        }
        for (int i = 0; i < shape[0]; i ++) {
            Int ox = 0;
            for (int __x = 0; __x < i; __x ++) {
                ox += dim_division(gShape[0] - gcgc, shape[0], __x);
            }
            pdoffset[0][i] = ox;
            pdshape[0][i] = dim_division(gShape[0] - gcgc, shape[0], i) + gcgc;
        }
        for (int j = 0; j < shape[1]; j ++) {
            Int oy = 0;
            for (int __y = 0; __y < j; __y ++) {
                oy += dim_division(gShape[1] - gcgc, shape[1], __y);
            }
            pdoffset[1][j] = oy;
            pdshape[1][j] = dim_division(gShape[1] - gcgc, shape[1], j) + gcgc;
        }
        for (int k = 0; k < shape[2]; k ++) {
            Int oz = 0;
            for (int __z = 0; __z < k; __z ++) {
                oz += dim_division(gShape[2] - gcgc, shape[2], __z);
            }
            pdoffset[2][k] = oz;
            pdshape[2][k] = dim_division(gShape[2] - gcgc, shape[2], k) + gcgc;
        }

        pdm_list = std::vector<Region>(size, Region());
        for (int k = 0; k < shape[2]; k ++) {
        for (int j = 0; j < shape[1]; j ++) {
        for (int i = 0; i < shape[0]; i ++) {
            pdm_list[IDX(i,j,k,shape)] = Region(
                Int3{{pdshape[0][i], pdshape[1][j], pdshape[2][k]}},
                Int3{{pdoffset[0][i], pdoffset[1][j], pdoffset[2][k]}}
            );
        }}}

        // pdm_list = std::vector<Region>(size, Region());
        // for (INT k = 0; k < shape[2]; k ++) {
        // for (INT j = 0; j < shape[1]; j ++) {
        // for (INT i = 0; i < shape[0]; i ++) {
        //     INT ox = 0, oy = 0, oz = 0;
        //     for (INT __x = 0; __x < i; __x ++) {
        //         ox += dim_division(gShape[0] - gcgc, shape[0], __x);
        //     }
        //     for (INT __y = 0; __y < j; __y ++) {
        //         oy += dim_division(gShape[1] - gcgc, shape[1], __y);
        //     }
        //     for (INT __z = 0; __z < k; __z ++) {
        //         oz += dim_division(gShape[2] - gcgc, shape[2], __z);
        //     }
        //     pdm_list[IDX(i, j, k, shape)] = Region(
        //         INT3{{
        //             dim_division(gShape[0] - gcgc, shape[0], i) + gc*2,
        //             dim_division(gShape[1] - gcgc, shape[1], j) + gc*2,
        //             dim_division(gShape[2] - gcgc, shape[2], k) + gc*2
        //         }},
        //         INT3{{ox, oy, oz}}
        //     );
        // }}}

    }

    void set6Region(Int3 &inner_shape, Int3 &inner_offset, Int3 *boundary_shape, Int3 *boundary_offset, Int thick, const Region &map) const {
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

    bool validNeighbour(Int fid) {
        return (neighbour[fid] >= 0);
    }

protected:
    Int dim_division(Int dim_size, Int n_division, Int id) {
        Int p_dim_size = dim_size / n_division;
        if (id < dim_size % n_division) {
            p_dim_size ++;
        }
        return p_dim_size;
    }

    void initNeighbour() {
        int __rank = rank;
        Int i, j, k;
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
