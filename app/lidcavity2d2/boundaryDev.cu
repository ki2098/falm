#include "boundaryDev.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

namespace LidCavity2d2 {

using namespace Falm;

__global__ void kernel_pressureBC_E(
    MatrixFrame<REAL> &p,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - (gc*2) && k < pdm_shape.z - (gc*2)) {
        i += pdm_shape.x - gc;
        j += gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i-1, j, k, pdm_shape));
    }
}

__global__ void kernel_pressureBC_W(
    MatrixFrame<REAL> &p,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - (gc*2) && k < pdm_shape.z - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i+1, j, k, pdm_shape));
    }
}

__global__ void kernel_pressureBC_N(
    MatrixFrame<REAL> &p,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - (gc*2) && j < 1 && k < pdm_shape.z - (gc*2)) {
        i += gc;
        j += pdm_shape.y - gc;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j-1, k, pdm_shape));
    }
}

__global__ void kernel_pressureBC_S(
    MatrixFrame<REAL> &p,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - (gc*2) && j < 1 && k < pdm_shape.z - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j+1, k, pdm_shape));
    }
}

void dev_pressureBC_E(
    Matrix<REAL> &p,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - (gc*2) + 7) / 8, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_pressureBC_E<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(p.devptr), pdm.shape, gc);
}

void dev_pressureBC_W(
    Matrix<REAL> &p,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - (gc*2) + 7) / 8, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_pressureBC_W<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(p.devptr), pdm.shape, gc);
}

void dev_pressureBC_N(
    Matrix<REAL> &p,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - (gc*2) + 7) / 8, 1, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_pressureBC_N<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(p.devptr), pdm.shape, gc);
}

void dev_pressureBC_S(
    Matrix<REAL> &p,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - (gc*2) + 7) / 8, 1, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_pressureBC_S<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(p.devptr), pdm.shape, gc);
}

__global__ void kernel_velocityBC_E(
    MatrixFrame<REAL> &u,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - (gc*2) && k < pdm_shape.z - (gc*2)) {
        i += pdm_shape.x - gc;
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
    MatrixFrame<REAL> &u,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - (gc*2) && k < pdm_shape.z - (gc*2)) {
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
    MatrixFrame<REAL> &u,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - (gc*2) && j < 1 && k < pdm_shape.z - (gc*2)) {
        i += gc;
        j += pdm_shape.y - gc;
        k += gc;
        REAL uboundary[] = {1.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i, j  , k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j-1, k, pdm_shape), d);
            u(IDX(i, j+1, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j-2, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_velocityBC_S(
    MatrixFrame<REAL> &u,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - (gc*2) && j < 1 && k < pdm_shape.z - (gc*2)) {
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

void dev_velocityBC_E(
    Matrix<REAL> &u,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - (gc*2) + 7) / 8, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_velocityBC_E<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(u.devptr), pdm.shape, gc);
}

void dev_velocityBC_W(
    Matrix<REAL> &u,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - (gc*2) + 7) / 8, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_velocityBC_W<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(u.devptr), pdm.shape, gc);
}

void dev_velocityBC_N(
    Matrix<REAL> &u,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - (gc*2) + 7) / 8, 1, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_velocityBC_N<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(u.devptr), pdm.shape, gc);
}

void dev_velocityBC_S(
    Matrix<REAL> &u,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - (gc*2) + 7) / 8, 1, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_velocityBC_S<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(u.devptr), pdm.shape, gc);
}

__global__ void kernel_forceFaceVelocityZero_E(
    MatrixFrame<REAL> &uu,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - (gc*2) && k < pdm_shape.z - (gc*2)) {
        i += pdm_shape.x - gc - 1;
        j += gc;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 0) = 0;
    }
}

__global__ void kernel_forceFaceVelocityZero_W(
    MatrixFrame<REAL> &uu,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - (gc*2) && k < pdm_shape.z - (gc*2)) {
        i += gc - 1;
        j += gc;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 0) = 0;
    }
}

__global__ void kernel_forceFaceVelocityZero_N(
    MatrixFrame<REAL> &uu,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - (gc*2) && j < 1 && k < pdm_shape.z - (gc*2)) {
        i += gc;
        j += pdm_shape.y - gc - 1;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 1) = 0;
    }
}

__global__ void kernel_forceFaceVelocityZero_S(
    MatrixFrame<REAL> &uu,
    INTx3              pdm_shape,
    INT                gc
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - (gc*2) && j < 1 && k < pdm_shape.z - (gc*2)) {
        i += gc;
        j += gc - 1;
        k += gc;
        uu(IDX(i, j, k, pdm_shape), 1) = 0;
    }
}

void dev_forceFaceVelocityZero_E(
    Matrix<REAL> &uu,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - (gc*2) + 7) / 8, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_forceFaceVelocityZero_E<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(uu.devptr), pdm.shape, gc);
}

void dev_forceFaceVelocityZero_W(
    Matrix<REAL> &uu,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - (gc*2) + 7) / 8, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_forceFaceVelocityZero_W<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(uu.devptr), pdm.shape, gc);
}

void dev_forceFaceVelocityZero_N(
    Matrix<REAL> &uu,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - (gc*2) + 7) / 8, 1, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_forceFaceVelocityZero_N<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(uu.devptr), pdm.shape, gc);
}

void dev_forceFaceVelocityZero_S(
    Matrix<REAL> &uu,
    Mapper       &pdm,
    INT           gc,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - (gc*2) + 7) / 8, 1, (pdm.shape.z - (gc*2) + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_forceFaceVelocityZero_S<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(uu.devptr), pdm.shape, gc);
}

}