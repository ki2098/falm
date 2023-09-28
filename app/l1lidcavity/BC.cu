#include "BC.h"
#include "../../src/dev/devutil.cuh"

using namespace Falm;

__global__ void kernel_pressureBCE(
    RealFrame &p,
    INTx3 proc_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < proc_shape.y - Gdx2 && k < proc_shape.z - Gdx2) {
        i += proc_shape.x - Gd;
        j += Gd;
        k += Gd;
        p(IDX(i, j, k, proc_shape)) = p(IDX(i-1, j, k, proc_shape));
    }
}

__global__ void kernel_pressureBCW(
    RealFrame &p,
    INTx3 proc_shape
) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < 1 && j < proc_shape.y - Gdx2 && k < proc_shape.z - Gdx2) {
        i += Gd - 1;
        j += Gd;
        k += Gd;
        p(IDX(i, j, k, proc_shape)) = p(IDX(i+1, j, k,proc_shape));
    }
}

__global__ void kernel_pressureBCN(

) {
    
}

void dev_pressureBC(RealField &p, Mapper &proc_domain) {

}