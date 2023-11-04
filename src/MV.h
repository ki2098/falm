#ifndef FALM_MVL2_H
#define FALM_MVL2_H

#include "MVDevCall.h"
#include "CPM.h"

namespace Falm {

class FalmMV : public FalmMVDevCall {
public:
static void MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    if (cpm.size == 1) {
        Region map(pdm.shape, cpm.gc);
        FalmMVDevCall::MV(a, x, ax, pdm, map, block_dim);
    } else {
        CPMComm<REAL> cpmop(&cpm);
        cpmop.IExchange6Face(x.dev.ptr, 1, 0, 0);
        INT3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
        cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));
    
        // Mapper inner_map(inner_shape, inner_offset);
        FalmMVDevCall::MV(a, x, ax, pdm, Region(inner_shape, inner_offset), block_dim);
    
        cpmop.Wait6Face();
        cpmop.PostExchange6Face();
    
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                // Mapper map(boundary_shape[fid], boundary_offset[fid]);
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                FalmMVDevCall::MV(a, x, ax, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
    }
    falmWaitStream();
}

static REAL DotProduct(Matrix<REAL> &a, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL r = FalmMVDevCall::DotProduct(a, b, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&r, 1, getMPIDtype<REAL>(), MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static REAL EuclideanNormSq(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL r = FalmMVDevCall::EuclideanNormSq(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&r, 1, getMPIDtype<REAL>(), MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static REAL MaxDiag(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL r = FalmMVDevCall::MatColAbsMax(a, 0, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&r, 1, getMPIDtype<REAL>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return r;
}

static REAL MatColMax(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL cmax = FalmMVDevCall::MatColMax(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&cmax, 1, getMPIDtype<REAL>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return cmax;
}

static REAL MatColMin(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL cmin = FalmMVDevCall::MatColMin(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&cmin, 1, getMPIDtype<REAL>(), MPI_MIN, MPI_COMM_WORLD);
    }
    return cmin;
}

static REAL MatColAbsMax(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL cmax = FalmMVDevCall::MatColAbsMax(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&cmax, 1, getMPIDtype<REAL>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return cmax;
}

static REAL MatColAbsMin(Matrix<REAL> &a, INT col, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL cmin = FalmMVDevCall::MatColAbsMin(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&cmin, 1, getMPIDtype<REAL>(), MPI_MIN, MPI_COMM_WORLD);
    }
    return cmin;
}

static REAL VecMax(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL vmax = FalmMVDevCall::VecMax(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&vmax, 1, getMPIDtype<REAL>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return vmax;
}

static REAL VecMin(Matrix<REAL> &a, CPMBase &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    REAL vmax = FalmMVDevCall::VecMin(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&vmax, 1, getMPIDtype<REAL>(), MPI_MIN, MPI_COMM_WORLD);
    }
    return vmax;
}

};

}

#endif
