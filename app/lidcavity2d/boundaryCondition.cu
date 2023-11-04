#include "boundaryCondition.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

namespace LidCavity2d {

using namespace Falm;

__global__ void kernel_pressureBC_E(
    const MatrixFrame<REAL> *vp,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &p=*vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += pdm_shape[0] - gc;
        j += gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i-1, j, k, pdm_shape));
    }
}

__global__ void kernel_pressureBC_W(
    const MatrixFrame<REAL> *vp,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &p=*vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i+1, j, k, pdm_shape));
    }
}

__global__ void kernel_pressureBC_N(
    const MatrixFrame<REAL> *vp,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &p=*vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += pdm_shape[1] - gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j-1, k, pdm_shape));
    }
}

__global__ void kernel_pressureBC_S(
    const MatrixFrame<REAL> *vp,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &p=*vp;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j+1, k, pdm_shape));
    }
}

void pressureBC(
    Matrix<REAL> &p,
    CPMBase      &cpm,
    STREAM       *streamptr
) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    INT    &gc  = cpm.gc;
    dim3 grid_dim_ew(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    STREAM stream_e = (streamptr)? streamptr[0] : (STREAM)0;
    kernel_pressureBC_E<<<grid_dim_ew, block_dim_ew, 0, stream_e>>>(p.devptr, pdm.shape, gc);
    STREAM stream_w = (streamptr)? streamptr[1] : (STREAM)0;
    kernel_pressureBC_W<<<grid_dim_ew, block_dim_ew, 0, stream_w>>>(p.devptr, pdm.shape, gc);

    dim3 grid_dim_ns((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    STREAM stream_n = (streamptr)? streamptr[2] : (STREAM)0;
    kernel_pressureBC_N<<<grid_dim_ns, block_dim_ns, 0, stream_n>>>(p.devptr, pdm.shape, gc);
    STREAM stream_s = (streamptr)? streamptr[3] : (STREAM)0;
    kernel_pressureBC_S<<<grid_dim_ns, block_dim_ns, 0, stream_s>>>(p.devptr, pdm.shape, gc);

    if (streamptr) {
        for (INT fid = 0; fid < 4; fid ++) {
            falmWaitStream(streamptr[fid]);
        }
    } else {
        falmWaitStream();
    }
}

__global__ void kernel_velocityBC_E(
    const MatrixFrame<REAL> *vu,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &u=*vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += pdm_shape[0] - gc;
        j += gc;
        k += gc;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i  , j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i-1, j, k, pdm_shape), d);
            u(IDX(i+1, j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i-2, j, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_velocityBC_W(
    const MatrixFrame<REAL> *vu,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &u=*vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i  , j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i+1, j, k, pdm_shape), d);
            u(IDX(i-1, j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i+2, j, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_velocityBC_N(
    const MatrixFrame<REAL> *vu,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &u=*vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += pdm_shape[1] - gc;
        k += gc;
        REAL uboundary[] = {1.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i, j  , k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j-1, k, pdm_shape), d);
            u(IDX(i, j+1, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j-2, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_velocityBC_S(
    const MatrixFrame<REAL> *vu,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &u=*vu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i, j  , k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j+1, k, pdm_shape), d);
            u(IDX(i, j-1, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j+2, k, pdm_shape), d);
        }
    }
}

void velocityBC(
    Matrix<REAL> &u,
    CPMBase      &cpm,
    STREAM       *streamptr
) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    INT    &gc  = cpm.gc;
    dim3 grid_dim_ew(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    STREAM stream_e = (streamptr)? streamptr[0] : (STREAM)0;
    kernel_velocityBC_E<<<grid_dim_ew, block_dim_ew, 0, stream_e>>>(u.devptr, pdm.shape, gc);
    STREAM stream_w = (streamptr)? streamptr[1] : (STREAM)0;
    kernel_velocityBC_W<<<grid_dim_ew, block_dim_ew, 0, stream_w>>>(u.devptr, pdm.shape, gc);

    dim3 grid_dim_ns((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    STREAM stream_n = (streamptr)? streamptr[2] : (STREAM)0;
    kernel_velocityBC_N<<<grid_dim_ns, block_dim_ns, 0, stream_n>>>(u.devptr, pdm.shape, gc);
    STREAM stream_s = (streamptr)? streamptr[3] : (STREAM)0;
    kernel_velocityBC_S<<<grid_dim_ns, block_dim_ns, 0, stream_s>>>(u.devptr, pdm.shape, gc);

    if (streamptr) {
        for (INT fid = 0; fid < 4; fid ++) {
            falmWaitStream(streamptr[fid]);
        }
    } else {
        falmWaitStream();
    }
}

__global__ void kernel_forceFaceVelocityZero_E(
    const MatrixFrame<REAL> *vuu,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &uu=*vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += pdm_shape[0] - gc - 1;
        j += gc;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 0) = 0;
    }
}

__global__ void kernel_forceFaceVelocityZero_W(
    const MatrixFrame<REAL> *vuu,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &uu=*vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape[1] - (gc*2) && k < pdm_shape[2] - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 0) = 0;
    }
}

__global__ void kernel_forceFaceVelocityZero_N(
    const MatrixFrame<REAL> *vuu,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &uu=*vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += pdm_shape[1] - gc - 1;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 1) = 0;
    }
}

__global__ void kernel_forceFaceVelocityZero_S(
    const MatrixFrame<REAL> *vuu,
    INT3              pdm_shape,
    INT gc
) {
    const MatrixFrame<REAL> &uu=*vuu;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape[0] - (gc*2) && j < 1 && k < pdm_shape[2] - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 1) = 0;
    }
}

void forceFaceVelocityZero(
    Matrix<REAL> &uu,
    CPMBase      &cpm,
    STREAM       *streamptr
) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    INT    &gc  = cpm.gc;
    dim3 grid_dim_ew(1, (pdm.shape[1] - (gc*2) + 7) / 8, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    STREAM stream_e = (streamptr)? streamptr[0] : (STREAM)0;
    kernel_forceFaceVelocityZero_E<<<grid_dim_ew, block_dim_ew, 0, stream_e>>>(uu.devptr, pdm.shape, gc);
    STREAM stream_w = (streamptr)? streamptr[1] : (STREAM)0;
    kernel_forceFaceVelocityZero_W<<<grid_dim_ew, block_dim_ew, 0, stream_w>>>(uu.devptr, pdm.shape, gc);

    dim3 grid_dim_ns((pdm.shape[0] - (gc*2) + 7) / 8, 1, (pdm.shape[2] - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    STREAM stream_n = (streamptr)? streamptr[2] : (STREAM)0;
    kernel_forceFaceVelocityZero_N<<<grid_dim_ns, block_dim_ns, 0, stream_n>>>(uu.devptr, pdm.shape, gc);
    STREAM stream_s = (streamptr)? streamptr[3] : (STREAM)0;
    kernel_forceFaceVelocityZero_S<<<grid_dim_ns, block_dim_ns, 0, stream_s>>>(uu.devptr, pdm.shape, gc);

    if (streamptr) {
        for (INT fid = 0; fid < 4; fid ++) {
            falmWaitStream(streamptr[fid]);
        }
    } else {
        falmWaitStream();
    }
}

void copyZ5(
    Matrix<REAL> &field,
    CPMBase      &cpm,
    STREAM       *streamptr
) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    INT    &gc  = cpm.gc;
    INT idxcc = IDX(0, 0, gc  , pdm.shape);
    INT idxt1 = IDX(0, 0, gc+1, pdm.shape);
    INT idxt2 = IDX(0, 0, gc+2, pdm.shape);
    INT idxb1 = IDX(0, 0, gc-1, pdm.shape);
    INT idxb2 = IDX(0, 0, gc-2, pdm.shape);
    INT slice_size = pdm.shape[0] * pdm.shape[1];
    for (INT d = 0; d < field.shape[1]; d ++) {
        falmMemcpyAsync(&field.dev(idxt1, d), &field.dev(idxcc, d), sizeof(REAL) * slice_size, MCpType::Dev2Dev);
        falmMemcpyAsync(&field.dev(idxt2, d), &field.dev(idxcc, d), sizeof(REAL) * slice_size, MCpType::Dev2Dev);
        falmMemcpyAsync(&field.dev(idxb1, d), &field.dev(idxcc, d), sizeof(REAL) * slice_size, MCpType::Dev2Dev);
        falmMemcpyAsync(&field.dev(idxb2, d), &field.dev(idxcc, d), sizeof(REAL) * slice_size, MCpType::Dev2Dev);
    }
    falmWaitStream();
}

}