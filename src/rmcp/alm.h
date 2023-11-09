#ifndef FALM_RMCP_ALM_H
#define FALM_RMCP_ALM_H

#include "almDevCall.h"
#include "../CPM.h"

namespace Falm {

class RmcpAlm : public RmcpAlmDevCall {
public:

    RmcpAlm(const CPM &cpm) : RmcpAlmDevCall(cpm.pdm_list[cpm.rank]) {}

    void CalcTorque(Matrix<REAL> &x, Matrix<REAL> &ff, RmcpWindfarm &wf, CPM &cpm, dim3 block_dim={8,8,8}) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region  map(pdm.shape, cpm.gc);
        RmcpAlmDevCall::CalcTorque(x, ff, wf, pdm, map, block_dim);
        for (INT __ti = 0; __ti < wf.nTurbine; __ti ++) {
            CPM_AllReduce(&wf.tptr[__ti].torque, 1, getMPIDtype<REAL>(), MPI_SUM, MPI_COMM_WORLD);
        }
    }

    void ALM(Matrix<REAL> &u, Matrix<REAL> &x, Matrix<REAL> &ff, REAL t, RmcpWindfarm &wf, CPM &cpm, dim3 block_dim={8,8,8}) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region  map(pdm.shape, cpm.gc);
        RmcpAlmDevCall::ALM(u, x, ff, t, wf, pdm, map, block_dim);
    }

    void SetALMFlag(Matrix<REAL> &x, REAL t, RmcpWindfarm &wf, CPM &cpm, dim3 block_dim={8,8,8}) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region  map(pdm.shape, cpm.gc);
        RmcpAlmDevCall::SetALMFlag(x, t, wf, pdm, map, block_dim);
    }

};

}

#endif