#ifndef FALM_RMCP_ALM_H
#define FALM_RMCP_ALM_H

#include "almDevCall.h"
#include "../CPM.h"

namespace Falm {

class RmcpAlm : public RmcpAlmDevCall {
public:

    RmcpAlm() : RmcpAlmDevCall() {}

    RmcpAlm(const CPM &cpm) : RmcpAlmDevCall(cpm.pdm_list[cpm.rank]) {}

    void init(const CPM &cpm, json turbine_params, std::string workdir) {
        RmcpAlmDevCall::init(cpm.pdm_list[cpm.rank], turbine_params, workdir);
    }

    // void finalize() {
    //     RmcpAlmDevCall::r
    // }

    void CalcTorque(Matrix<REAL> &x, Matrix<REAL> &ff, RmcpTurbineArray &wf, CPM &cpm, dim3 block_dim={8,8,8}) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region  map(pdm.shape, cpm.gc);
        RmcpAlmDevCall::CalcTorque(x, ff, wf, pdm, map, block_dim);
        for (INT __ti = 0; __ti < wf.nTurbine; __ti ++) {
            CPM_AllReduce(&wf.tptr[__ti].torque, 1, getMPIDtype<REAL>(), MPI_SUM, MPI_COMM_WORLD);
        }
    }

    void ALM(Matrix<REAL> &u, Matrix<REAL> &x, Matrix<REAL> &ff, REAL t, RmcpTurbineArray &wf, CPM &cpm, dim3 block_dim={8,8,8}) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region  map(pdm.shape, cpm.gc);
        RmcpAlmDevCall::ALM(u, x, ff, t, wf, pdm, map, block_dim);
    }

    void ALM(Matrix<REAL> &u, Matrix<REAL> &x, Matrix<REAL> &ff, REAL t, CPM &cpm, dim3 block_dim={8,8,8}) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region  map(pdm.shape, cpm.gc);
        RmcpAlmDevCall::ALM(u, x, ff, t, pdm, map, block_dim);
    }

    void ALM(BladeHandler &blades, Matrix<REAL> &u, Matrix<REAL> &x, Matrix<REAL> &ff, REAL t, RmcpTurbineArray &wf, CPM &cpm, dim3 block_dim={8,8,8}) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region  map(pdm.shape, cpm.gc);
        RmcpAlmDevCall::ALM(blades, u, x, ff, t, wf, pdm, map, block_dim);
    }

    void SetALMFlag(Matrix<REAL> &x, REAL t, RmcpTurbineArray &wf, CPM &cpm, dim3 block_dim={8,8,8}) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region  map(pdm.shape, cpm.gc);
        RmcpAlmDevCall::SetALMFlag(x, t, wf, pdm, map, block_dim);
    }

    void SetALMFlag(Matrix<REAL> &x, REAL t, CPM &cpm, dim3 block_dim={8,8,8}) {
        Region &pdm = cpm.pdm_list[cpm.rank];
        Region  map(pdm.shape, cpm.gc);
        RmcpAlmDevCall::SetALMFlag(x, t, pdm, map, block_dim);
    }

    void print_info(bool is_output_rank) {
        if (is_output_rank) {
            printf("TURBINE INFO START\n");
            printf("\tBlade length %lf", turbines.host.radius);
            printf("\tBlade number %d\n", turbines.host.n_blade);
            printf("\tBlade property file %s\n\n", blades.property_file_path.c_str());
            for (int i = 0; i < turbines.host.n_turbine; i ++) {
                printf("\tTurbine %d\n", i);
                printf("\t\tBase (%lf %lf %lf)\n", turbines.host.base[i][0], turbines.host.base[i][1],  turbines.host.base[i][2]);
                printf("\t\tBase velocity (%lf %lf %lf)\n", turbines.host.base_velocity[i][0], turbines.host.base_velocity[i][1], turbines.host.base_velocity[i][2]);
                for (int j = 0; j < 3; j ++) {
                    if (turbines.turbine_param["turbineArray"][i]["baseLocation"][j].is_string()) {
                        printf("\t\tCoordinate[%d] subject to %s\n", j, turbines.turbine_param["turbineArray"][i]["baseLocation"][j].get_ref<std::string&>().c_str());
                    }
                }
                printf("\t\tAngle type %d\n", int(turbines.host.angle_type[i]));
                printf("\t\tAngle (%lf %lf %lf)\n", rad2deg(turbines.host.angle[i][0]), rad2deg(turbines.host.angle[i][1]), rad2deg(turbines.host.angle[i][2]));
                printf("\t\tAngular velocity (%lf %lf %lf)\n", rad2deg(turbines.host.angular_velocity[i][0]), rad2deg(turbines.host.angular_velocity[i][1]), rad2deg(turbines.host.angular_velocity[i][2]));
                for (int j = 0; j < 3; j ++) {
                    if (turbines.turbine_param["turbineArray"][i]["angle"][j].is_string()) {
                        printf("\t\tAngle[%d] subject to %s\n", j, turbines.turbine_param["turbineArray"][i]["angle"][j].get_ref<std::string&>().c_str());
                    }
                }
                printf("\t\tRotation center (%lf %lf %lf)\n", turbines.host.hub[i][0], turbines.host.hub[i] [1], turbines.host.hub[i][2]);
                printf("\t\tRotation speed %lf\n", turbines.host.tip_rate[i]);
            }
            printf("TURBINE INFO END\n");
        }
    }
};

}

#endif
