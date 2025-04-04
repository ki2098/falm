#ifndef FALM_ALM_ALMDEVCALL_H
#define FALM_ALM_ALMDEVCALL_H

#include "../matrix.h"
#include "../dev/devutil.cuh"
#include "../falmath.h"
#include "apHandler.h"
#include "bladeHandler.h"
#include "turbineHandler.h"
#include "../CPMBase.h"
#include "../cpputil.hpp"

namespace Falm {

namespace Alm {

class AlmDevCall {

public:
    TurbineHandler turbines;
    APHandler aps;
    Int3 mpi_shape;
    int rank;
    // Matrix<INT> x_offset, y_offset, z_offset;
    Int3 pdm_shape;
    Int3 pdm_offset;
    Int gc;
    std::string workdir;
    Real euler_eps;
    Int n_ap_per_blade;

    AlmDevCall() : turbines(), aps() {}

    void init(const std::string &workdir, const Json &turbine_params, std::string ap_path, const CPM &cpm) {
        // this->workdir = workdir;
        // turbines.alloc(turbine_params);
        // std::string blade_path = turbine_params["bladeProperties"];
        // blade_path = glue_path(workdir, blade_path);
        // ap_path = glue_path(workdir, ap_path);
        // BladeHandler::buildAP(blade_path, ap_path, turbines.n_turbine, turbines.n_blade, turbines.n_ap_per_blade, turbines.radius);
        // aps.alloc(ap_path);

        // mpi_shape = cpm.shape;
        // rank = cpm.rank;
        // x_offset.alloc(cpm.shape[0], 1, HDC::HstDev);
        // y_offset.alloc(cpm.shape[1], 1, HDC::HstDev);
        // z_offset.alloc(cpm.shape[2], 1, HDC::HstDev);
        // for (int i = 0; i < cpm.shape[0]; i ++) {
        //     x_offset(i) = cpm.pdm_list[IDX(i,0,0,cpm.shape)].offset[0] + cpm.gc - 1;
        // }
        // for (int j = 0; j < cpm.shape[1]; j ++) {
        //     y_offset(j) = cpm.pdm_list[IDX(0,j,0,cpm.shape)].offset[1] + cpm.gc - 1;
        // }
        // for (int k = 0; k < cpm.shape[2]; k ++) {
        //     z_offset(k) = cpm.pdm_list[IDX(0,0,k,cpm.shape)].offset[2] + cpm.gc - 1;
        // }
        // x_offset.sync(MCP::Hst2Dev);
        // y_offset.sync(MCP::Hst2Dev);
        // z_offset.sync(MCP::Hst2Dev);
        pdm_shape = cpm.pdm_list[cpm.rank].shape;
        pdm_offset = cpm.pdm_list[cpm.rank].offset;
        gc = cpm.gc;
    }

    void finalize() {
        turbines.release();
        aps.release();

        // x_offset.release();
        // y_offset.release();
        // z_offset.release();
    }

    void UpdateTurbineAngles(Real t, size_t block_size = 32);

    void UpdateAPX(Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z, Real t, size_t block_size=32);

    void CalcAPForce(Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z, Matrix<Real> &uvw, Real t, size_t block_size=32);

    void DistributeAPForce(Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z, Matrix<Real> &ff, Real euler_eps, dim3 block_size={8,8,8});

    void DryDistribution(Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z, Matrix<Real> &phi, Real euler_eps, dim3 block_size={8,8,8});

    void CalcTorqueAndThrust();
};

}

}

#endif