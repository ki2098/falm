#ifndef FALM_RMCP_ALMDEVCALL_H
#define FALM_RMCP_ALMDEVCALL_H

#include "../matrix.h"
#include "../region.h"
#include "turbine.h"
#include "bladeHandler.h"
#include "turbineHandler.h"

namespace Falm {

namespace Rmcp {

class RmcpAlmDevCall {
public:
    Matrix<Int> alm_flag;
    BladeHandler blades;
    TurbineHandler turbines;

    RmcpAlmDevCall() : alm_flag(), blades(), turbines() {}

    RmcpAlmDevCall(const Region &pdm) : alm_flag(pdm.shape, 1, HDC::Device, "ALM flag") {}

    void init(const Region &pdm, Json turbine_params, std::string workdir) {
        alm_flag.alloc(pdm.shape, 1, HDC::Device, "ALM flag");
        std::string bppath = turbine_params["bladeProperties"];
        if (bppath[0] == '/') {
           
        } else if (workdir.back() == '/') {
            bppath = workdir + bppath;
        } else {
            bppath = workdir + "/" + bppath;
        } 
        blades.alloc(bppath);
        printf("blades OK\n");
        turbines.alloc(turbine_params);
        printf("turbines OK\n");
    }

    void finalize() {
        alm_flag.release();
        blades.release();
        turbines.release();
    }

    void ALM(Matrix<Real> &u, Matrix<Real> &x, Matrix<Real> &ff, Real t, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim={8,8,8});

    void ALM(BladeHandler &blades, Matrix<Real> &u, Matrix<Real> &x, Matrix<Real> &ff, Real t, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim={8,8,8});

    void SetALMFlag(Matrix<Real> &x, Real t, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim={8,8,8});

    void CalcTorque(Matrix<Real> &x, Matrix<Real> &ff, RmcpTurbineArray &wf, const Region &pdm, const Region &map, dim3 block_dim={8,8,8});

    void SetALMFlag(Matrix<Real> &x, Real t, const Region &pdm, const Region &map, dim3 block_dim={8,8,8});

    void ALM(Matrix<Real> &u, Matrix<Real> &x, Matrix<Real> &ff, Real t, const Region &pdm, const Region &map, dim3 block_dim={8,8,8});

    // void CalcTorque(Matrix<REAL> &x, Matrix<REAL> &ff, const Region &pdm, const Region &map, dim3 block_dim={8,8,8});

};

}

}

#endif
