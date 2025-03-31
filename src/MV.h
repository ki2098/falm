#ifndef FALM_MVL2_H
#define FALM_MVL2_H

#include "MVDevCall.h"
#include "CPM.h"

namespace Falm {

// using FalmMVDevCall=FalmMVDevCallv2;

class FalmMV : public FalmMVDevCall {
public:
static void MV(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &ax, CPM &cpm, dim3 block_dim, Stream *stream = nullptr) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    CPMComm<Real> cpmop(&cpm);
    cpmop.IExchange6Face(x.dev.ptr, 1, 0, 0);
    Int3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));
    
    FalmMVDevCall::MV(a, x, ax, pdm, Region(inner_shape, inner_offset), block_dim);
    
    cpmop.Wait6Face();
    cpmop.PostExchange6Face();
    
    for (Int fid = 0; fid < 6; fid ++) {
        if (cpm.neighbour[fid] >= 0) {
            dim3 __block(
                (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
            );
            // Mapper map(boundary_shape[fid], boundary_offset[fid]);
            Stream fstream = (stream)? stream[fid] : (Stream)0;
            FalmMVDevCall::MV(a, x, ax, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
        }
    }
    if (stream) {
        for (Int fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                falmWaitStream(stream[fid]);
            }
        }
    }
    falmWaitStream();
}

static Real DotProduct(Matrix<Real> &a, Matrix<Real> &b, CPM &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    Real r = FalmMVDevCall::DotProduct(a, b, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&r, 1, getMPIDtype<Real>(), MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static Real EuclideanNormSq(Matrix<Real> &a, CPM &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    Real r = FalmMVDevCall::EuclideanNormSq(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&r, 1, getMPIDtype<Real>(), MPI_SUM, MPI_COMM_WORLD);
    }
    return r;
}

static Real MaxDiag(Matrix<Real> &a, CPM &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    Real r = FalmMVDevCall::MatColAbsMax(a, 0, pdm, map, block_dim);
    // printf("%d %lf\n", cpm.rank, r);
    if (cpm.size > 1) {
        CPM_AllReduce(&r, 1, getMPIDtype<Real>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return r;
}

static Real MatColMax(Matrix<Real> &a, Int col, CPM &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    Real cmax = FalmMVDevCall::MatColMax(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&cmax, 1, getMPIDtype<Real>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return cmax;
}

static Real MatColMin(Matrix<Real> &a, Int col, CPM &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    Real cmin = FalmMVDevCall::MatColMin(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&cmin, 1, getMPIDtype<Real>(), MPI_MIN, MPI_COMM_WORLD);
    }
    return cmin;
}

static Real MatColAbsMax(Matrix<Real> &a, Int col, CPM &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    Real cmax = FalmMVDevCall::MatColAbsMax(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&cmax, 1, getMPIDtype<Real>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return cmax;
}

static Real MatColAbsMin(Matrix<Real> &a, Int col, CPM &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    Real cmin = FalmMVDevCall::MatColAbsMin(a, col, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&cmin, 1, getMPIDtype<Real>(), MPI_MIN, MPI_COMM_WORLD);
    }
    return cmin;
}

static Real VecMax(Matrix<Real> &a, CPM &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    Real vmax = FalmMVDevCall::VecMax(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&vmax, 1, getMPIDtype<Real>(), MPI_MAX, MPI_COMM_WORLD);
    }
    return vmax;
}

static Real VecMin(Matrix<Real> &a, CPM &cpm, dim3 block_dim) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region map(pdm.shape, cpm.gc);
    Real vmax = FalmMVDevCall::VecMin(a, pdm, map, block_dim);
    if (cpm.size > 1) {
        CPM_AllReduce(&vmax, 1, getMPIDtype<Real>(), MPI_MIN, MPI_COMM_WORLD);
    }
    return vmax;
}

};

}

#endif
