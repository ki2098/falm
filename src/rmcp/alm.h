#ifndef FALM_RMCP_ALM_H
#define FALM_RMCP_ALM_H

#include "../matrix.h"
#include "../CPM.h"
#include "turbine.h"

namespace Falm {

class RmcpAlm {
public:
    Matrix<INT> alm_flag;

    RmcpAlm(CPMBase &cpm) : alm_flag(cpm.pdm_list[cpm.rank].shape, 1, HDCType::Device, "ALM flag") {}

    void Rmcp_ALM(Matrix<REAL> &u, Matrix<REAL> &x, Matrix<REAL> &ff, REAL t, RmcpWindfarm &wf, CPMBase &cpm);

    void Rmcp_SetALMFlag(Matrix<REAL> &x, REAL t, RmcpWindfarm &wf, CPMBase &cpm, dim3 block_dim={8,8,8});

    void Rmcp_CalcTorque(Matrix<REAL> &ff, RmcpWindfarm &wf, CPMBase &cpm);

};

}

#endif