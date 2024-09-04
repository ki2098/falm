#ifndef FALM_ALM_ALM_H
#define FALM_ALM_ALM_H

#include "almDevCall.h"
#include "../CPM.h"

namespace Falm {

class AlmHandler : public AlmDevCall {
public:
    REAL euler_eps;

    AlmHandler() : AlmDevCall() {}

    void init(const std::string &workdir, const json &turbine_params, std::string ap_path, const CPM &cpm, REAL euler_eps) {
        AlmDevCall::init(workdir, turbine_params, ap_path, cpm);
        this->euler_eps = euler_eps;
    }

    void finalize() {
        AlmDevCall::finalize();
    }

    void Alm(Matrix<REAL> &x, Matrix<REAL> &y, Matrix<REAL> &z, Matrix<REAL> &uvw, Matrix<REAL> &ff, REAL t, dim3 block_size={8,8,8}) {
        AlmDevCall::UpdateAPX(x, y, z, t);
        AlmDevCall::CalcAPForce(x, y, z, uvw, t);
        falmMemcpy(aps.host.force, aps.dev.force, sizeof(REAL3)*aps.apcount, MCP::Dev2Hst);
        CPM_AllReduce(aps.host.force, 3*aps.apcount, getMPIDtype<REAL>(), MPI_SUM, MPI_COMM_WORLD);
        falmMemcpy(aps.dev.force, aps.host.force, sizeof(REAL3)*aps.apcount, MCP::Hst2Dev);
        AlmDevCall::DistributeAPForce(x, y, z, ff, euler_eps, block_size);
    }

    void DryDistribution(Matrix<REAL> &x, Matrix<REAL> &y, Matrix<REAL> &z, Matrix<REAL> &phi, dim3 block_size={8,8,8}) {
        AlmDevCall::DryDistribution(x, y, z, phi, euler_eps, block_size);
    }

};

}

#endif