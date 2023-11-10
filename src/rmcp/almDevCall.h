#ifndef FALM_RMCP_ALMDEVCALL_H
#define FALM_RMCP_ALMDEVCALL_H

#include "../matrix.h"
#include "../region.h"
#include "turbine.h"

namespace Falm {

class RmcpAlmDevCall {
public:
    Matrix<INT> alm_flag;

    RmcpAlmDevCall(const Region &pdm) : alm_flag(pdm.shape, 1, HDCType::Device, "ALM flag") {}

    void ALM(Matrix<REAL> &u, Matrix<REAL> &x, Matrix<REAL> &ff, REAL t, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim={8,8,8});

    void SetALMFlag(Matrix<REAL> &x, REAL t, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim={8,8,8});

    void CalcTorque(Matrix<REAL> &x, Matrix<REAL> &ff, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim={8,8,8});

};

}

#endif