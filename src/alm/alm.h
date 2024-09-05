#ifndef FALM_ALM_ALM_H
#define FALM_ALM_ALM_H

#include "almDevCall.h"
#include "../CPM.h"

namespace Falm {

namespace Alm {

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

    void print_info(int output_rank) {
        if (output_rank == rank) {
            printf("TURBINE INFO START\n");
            printf("\tBlade length %lf", turbines.host.radius);
            printf("\tBlade number %d\n", turbines.host.n_blade);
            printf("\tBlade point number %d\n", turbines.n_ap_per_blade);
            printf("\tBlade property file %s\n\n", glue_path(workdir, turbines.turbine_param["bladeProperties"]).c_str());
            for (int i = 0; i < turbines.host.n_turbine; i ++) {
                printf("\tTurbine %d\n", i);
                printf("\t\tBase (%lf %lf %lf)\n", turbines.host.base[i][0], turbines.host.base[i][1],  turbines.host.base[i][2]);
                printf("\t\tBase velocity (%lf %lf %lf)\n", turbines.host.base_velocity[i][0], turbines.host.base_velocity[i][1], turbines.host.base_velocity[i][2]);
                for (int j = 0; j < 3; j ++) {
                    if (turbines.turbine_param["turbineArray"][i]["baseLocation"][j].is_string()) {
                        printf("\t\tCoordinate[%d] subject to %s\n", j, glue_path(workdir, turbines.turbine_param["turbineArray"][i]["baseLocation"][j]).c_str());
                    }
                }
                printf("\t\tAngle type %d\n", int(turbines.host.angle_type[i]));
                printf("\t\tAngle (%lf %lf %lf)\n", rad2deg(turbines.host.angle[i][0]), rad2deg(turbines.host.angle[i][1]), rad2deg(turbines.host.angle[i][2]));
                printf("\t\tAngular velocity (%lf %lf %lf)\n", rad2deg(turbines.host.angular_velocity[i][0]), rad2deg(turbines.host.angular_velocity[i][1]), rad2deg(turbines.host.angular_velocity[i][2]));
                for (int j = 0; j < 3; j ++) {
                    if (turbines.turbine_param["turbineArray"][i]["angle"][j].is_string()) {
                        printf("\t\tAngle[%d] subject to %s\n", j, glue_path(workdir, turbines.turbine_param["turbineArray"][i]["angle"][j]).c_str());
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

}

#endif