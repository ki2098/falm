#include "boundaryDev.h"
#include "../../src/util.h"
#include "../../src/dev/devutil.cuh"

namespace LidCavity2d2 {

using namespace Falm;

__global__ void kernel_pressureBC_E(
    MatrixFrame<REAL> &p,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - Gdx2 && k < pdm_shape.z - Gdx2) {
        i += pdm_shape.x - Gd;
        j += Gd;
        k += Gd;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i-1, j, k, pdm_shape));
    }
}

__global__ void kernel_pressureBC_W(
    MatrixFrame<REAL> &p,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - Gdx2 && k < pdm_shape.z - Gdx2) {
        i += Gd - 1;
        j += Gd;
        k += Gd;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i+1, j, k, pdm_shape));
    }
}

__global__ void kernel_pressureBC_N(
    MatrixFrame<REAL> &p,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - Gdx2 && j < 1 && k < pdm_shape.z - Gdx2) {
        i += Gd;
        j += pdm_shape.y - Gd;
        k += Gd;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j-1, k, pdm_shape));
    }
}

__global__ void kernel_pressureBC_S(
    MatrixFrame<REAL> &p,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - Gdx2 && j < 1 && k < pdm_shape.z - Gdx2) {
        i += Gd;
        j += Gd - 1;
        k += Gd;
        p(IDX(i, j, k, pdm_shape)) = p(IDX(i, j+1, k, pdm_shape));
    }
}

void dev_pressureBC_E(
    Matrix<REAL> &p,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - Gdx2 + 7) / 8, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_pressureBC_E<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(p.devptr), pdm.shape);
}

void dev_pressureBC_W(
    Matrix<REAL> &p,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - Gdx2 + 7) / 8, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_pressureBC_W<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(p.devptr), pdm.shape);
}

void dev_pressureBC_N(
    Matrix<REAL> &p,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - Gdx2 + 7) / 8, 1, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_pressureBC_N<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(p.devptr), pdm.shape);
}

void dev_pressureBC_S(
    Matrix<REAL> &p,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - Gdx2 + 7) / 8, 1, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_pressureBC_S<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(p.devptr), pdm.shape);
}

__global__ void kernel_velocityBC_E(
    MatrixFrame<REAL> &u,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - Gdx2 && k < pdm_shape.z - Gdx2) {
        i += pdm_shape.x - Gd;
        j += Gd;
        k += Gd;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i  , j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i-1, j, k, pdm_shape), d);
            u(IDX(i+1, j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i-2, j, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_velocityBC_W(
    MatrixFrame<REAL> &u,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - Gdx2 && k < pdm_shape.z - Gdx2) {
        i += Gd - 1;
        j += Gd;
        k += Gd;
        REAL uboundary[] = {0.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i  , j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i+1, j, k, pdm_shape), d);
            u(IDX(i-1, j, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i+2, j, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_velocityBC_N(
    MatrixFrame<REAL> &u,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - Gdx2 && j < 1 && k < pdm_shape.z - Gdx2) {
        i += Gd;
        j += pdm_shape.y - Gd;
        k += Gd;
        REAL uboundary[] = {1.0, 0.0, 0.0};
        for (INT d = 0; d < 3; d ++) {
            u(IDX(i, j  , k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j-1, k, pdm_shape), d);
            u(IDX(i, j+1, k, pdm_shape), d) = 2 * uboundary[d] - u(IDX(i, j-2, k, pdm_shape), d);
        }
    }
}

__global__ void kernel_velocityBC_S(
    MatrixFrame<REAL> &u,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - Gdx2 && j < 1 && k < pdm_shape.z - Gdx2) {
        i += Gd;
        j += Gd - 1;
        k += Gd;
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
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - Gdx2 + 7) / 8, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_velocityBC_E<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(u.devptr), pdm.shape);
}

void dev_velocityBC_W(
    Matrix<REAL> &u,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - Gdx2 + 7) / 8, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_velocityBC_W<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(u.devptr), pdm.shape);
}

void dev_velocityBC_N(
    Matrix<REAL> &u,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - Gdx2 + 7) / 8, 1, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_velocityBC_N<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(u.devptr), pdm.shape);
}

void dev_velocityBC_S(
    Matrix<REAL> &u,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - Gdx2 + 7) / 8, 1, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_velocityBC_S<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(u.devptr), pdm.shape);
}

__global__ void kernel_forceFaceVelocityZero_E(
    MatrixFrame<REAL> &uu,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - Gdx2 && k < pdm_shape.z - Gdx2) {
        i += pdm_shape.x - Gd - 1;
        j += Gd;
        k += Gd;
        uu(IDX(i, j, k, pdm_shape), 0) = 0;
    }
}

__global__ void kernel_forceFaceVelocityZero_W(
    MatrixFrame<REAL> &uu,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < pdm_shape.y - Gdx2 && k < pdm_shape.z - Gdx2) {
        i += Gd - 1;
        j += Gd;
        k += Gd;
        uu(IDX(i, j, k, pdm_shape), 0) = 0;
    }
}

__global__ void kernel_forceFaceVelocityZero_N(
    MatrixFrame<REAL> &uu,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - Gdx2 && j < 1 && k < pdm_shape.z - Gdx2) {
        i += Gd;
        j += pdm_shape.y - Gd - 1;
        k += Gd;
        uu(IDX(i, j, k, pdm_shape), 1) = 0;
    }
}

__global__ void kernel_forceFaceVelocityZero_S(
    MatrixFrame<REAL> &uu,
    INTx3              pdm_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < pdm_shape.x - Gdx2 && j < 1 && k < pdm_shape.z - Gdx2) {
        i += Gd;
        j += Gd - 1;
        k += Gd;
        uu(IDX(i, j, k, pdm_shape), 1) = 0;
    }
}

void dev_forceFaceVelocityZero_E(
    Matrix<REAL> &uu,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - Gdx2 + 7) / 8, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_forceFaceVelocityZero_E<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(uu.devptr), pdm.shape);
}

void dev_forceFaceVelocityZero_W(
    Matrix<REAL> &uu,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ew(1, (pdm.shape.y - Gdx2 + 7) / 8, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ew(1, 8, 8);
    kernel_forceFaceVelocityZero_W<<<grid_dim_ew, block_dim_ew, 0, stream>>>(*(uu.devptr), pdm.shape);
}

void dev_forceFaceVelocityZero_N(
    Matrix<REAL> &uu,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - Gdx2 + 7) / 8, 1, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_forceFaceVelocityZero_N<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(uu.devptr), pdm.shape);
}

void dev_forceFaceVelocityZero_S(
    Matrix<REAL> &uu,
    Mapper       &pdm,
    STREAM        stream
) {
    dim3 grid_dim_ns((pdm.shape.x - Gdx2 + 7) / 8, 1, (pdm.shape.z - Gdx2 + 7) / 8);
    dim3 block_dim_ns(8, 1, 8);
    kernel_forceFaceVelocityZero_S<<<grid_dim_ns, block_dim_ns, 0, stream>>>(*(uu.devptr), pdm.shape);
}

}