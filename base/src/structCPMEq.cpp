#include "structCPMEq.h"

namespace Falm {

void devL2_Struct3d7p_MV(Matrix<double> &a, Matrix<double> &x, Matrix<double> &ax, Mapper &pdom, Mapper &map, dim3 &block_dim, CPM &cpm) {
    CPMBuffer<double> *buffer;
    MPI_Request *req;
    cpm.CPML2dev_IExchange6Face(a.dev.ptr, pdom, 1, 0, buffer, HDCType::Device, req);
    uint3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper icmap;
    devL1_Struct3d7p_MV(a, x, ax, pdom, icmap, block_dim);

    cpm.CPML2_Wait6Face(req);
    cpm.CPML2dev_PostExchange6Face(a.dev.ptr, pdom, buffer, req);

    
}

}