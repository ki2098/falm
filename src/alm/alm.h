#ifndef FALM_ALM_ALM_H
#define FALM_ALM_ALM_H

#include <string>
#include "almDevCall.h"
#include "../CPM.h"

namespace Falm {

namespace Alm {

class AlmHandler : public AlmDevCall {
public:
    Json turbine_param, turbine_prop;
    std::ofstream cpctOut;
    std::string cpctPath;

    AlmHandler() : AlmDevCall() {}

    void init(const std::string &workdir, const Json &turbine_params, const CPM &cpm, std::string cpctPath) {
        this->workdir = workdir;
        this->cpctPath = cpctPath;
        mpi_shape = cpm.shape;
        rank = cpm.rank;

        turbine_param = turbine_params;

        this->euler_eps = turbine_param["projectionWidth"];
        this->n_ap_per_blade = turbine_param["bladePointNumber"];

        std::string turbine_prop_path = turbine_param["turbineProperties"];
        turbine_prop_path = glue_path(workdir, turbine_prop_path);
        std::string ap_path = turbine_param["apFile"];
        ap_path = glue_path(workdir, ap_path);

        std::ifstream turbine_prop_file(turbine_prop_path);
        turbine_prop = Json::parse(turbine_prop_file);
        turbine_prop_file.close();

        turbines.alloc(turbine_prop, turbine_param["turbineArray"]);

        if (rank == 0) {
            BladeHandler::buildAP(turbine_prop["bladeProperties"], ap_path, turbines.n_turbine, turbines.n_blade, this->n_ap_per_blade, turbines.radius, turbines.hub_radius);
        }
        CPM_Barrier(MPI_COMM_WORLD);
        aps.alloc(ap_path);
        AlmDevCall::init(workdir, turbine_param, ap_path, cpm);
        
        if (turbine_param["writePowerThrust"].get<bool>() && rank == 0) {
            cpctOut.open(cpctPath);
            cpctOut << "t";
            for (int tid = 0; tid < turbines.n_turbine; tid ++) {
                cpctOut << ",P" << tid << ",T" << tid;
            }
            cpctOut << std::endl;
        }
    }

    void finalize() {
        AlmDevCall::finalize();
        if (cpctOut.is_open()) {
            cpctOut.close();
        }
    }

    void Alm(Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z, Matrix<Real> &uvw, Matrix<Real> &ff, Real t, dim3 block_size={8,8,8}) {
        AlmDevCall::UpdateTurbineAngles(t);
        AlmDevCall::UpdateAPX(x, y, z, t);
        AlmDevCall::CalcAPForce(x, y, z, uvw, t);
        falmMemcpy(aps.host.force, aps.dev.force, sizeof(Real3)*aps.apcount, MCP::Dev2Hst);
        CPM_AllReduce(aps.host.force, 3*aps.apcount, getMPIDtype<Real>(), MPI_SUM, MPI_COMM_WORLD);
        falmMemcpy(aps.dev.force, aps.host.force, sizeof(Real3)*aps.apcount, MCP::Hst2Dev);
        AlmDevCall::DistributeAPForce(x, y, z, ff, euler_eps, block_size);
        AlmDevCall::CalcTorqueAndThrust();
        CPM_AllReduce(turbines.host.torque, turbines.n_turbine, getMPIDtype<Real>(), MPI_SUM, MPI_COMM_WORLD);
        CPM_AllReduce(turbines.host.thrust, turbines.n_turbine, getMPIDtype<Real>(), MPI_SUM, MPI_COMM_WORLD);
    }

    void DryDistribution(Matrix<Real> &x, Matrix<Real> &y, Matrix<Real> &z, Matrix<Real> &phi, dim3 block_size={8,8,8}) {
        AlmDevCall::DryDistribution(x, y, z, phi, euler_eps, block_size);
    }

    void print_info(int output_rank) {
        if (output_rank == rank) {
            printf("TURBINE INFO START\n");
            printf("\tRadius %lf\n", turbines.host.radius);
            printf("\tHub radius %lf\n", turbines.host.hub_radius);
            printf("\tBlade number %d\n", turbines.host.n_blade);
            printf("\tBlade point number %d\n", n_ap_per_blade);
            printf("\tEuler projection width %lf\n", euler_eps);
            printf("\tTurbine property file %s\n\n", glue_path(workdir, turbine_param["turbineProperties"]).c_str());
            for (int i = 0; i < turbines.host.n_turbine; i ++) {
                printf("\tTurbine %d\n", i);
                printf("\t\tBase (%lf %lf %lf)\n", turbines.host.base[i][0], turbines.host.base[i][1],  turbines.host.base[i][2]);
                printf("\t\tBase velocity (%lf %lf %lf)\n", turbines.host.base_velocity[i][0], turbines.host.base_velocity[i][1], turbines.host.base_velocity[i][2]);
                for (int j = 0; j < 3; j ++) {
                    if (turbines.turbine_arr[i]["baseLocation"][j].is_string()) {
                        printf("\t\tCoordinate[%d] subject to %s\n", j, glue_path(workdir, turbines.turbine_arr[i]["baseLocation"][j]).c_str());
                    }
                }
                // printf("\t\tAngle type %d\n", int(turbines.host.angle_type[i]));
                // printf("\t\tAngle (%lf %lf %lf)\n", rad2deg(turbines.host.angle[i][0]), rad2deg(turbines.host.angle[i][1]), rad2deg(turbines.host.angle[i][2]));
                // printf("\t\tAngular velocity (%lf %lf %lf)\n", rad2deg(turbines.host.angular_velocity[i][0]), rad2deg(turbines.host.angular_velocity[i][1]), rad2deg(turbines.host.angular_velocity[i][2]));
                // for (int j = 0; j < 3; j ++) {
                //     if (turbines.turbine_arr[i]["angle"][j].is_string()) {
                //         printf("\t\tAngle[%d] subject to %s\n", j, glue_path(workdir, turbines.turbine_arr[i]["angle"][j]).c_str());
                //     }
                // }
                printf("\t\tAngle type %s\n", get_euler_angle_name(turbines.host.angle_type[i]).c_str());
                auto motion = turbines.host.motion[i];
                if (motion[1] == 0) {
                    printf("\t\tAngle value %lf [deg]\n", rad2deg(motion[0]*sin(motion[2])));
                } else {
                    printf("\t\tAngle motion %lfsin(%lft + %lf) [rad]\n", motion[0], motion[1], motion[2]);
                }
                printf("\t\tRotation center (%lf %lf %lf)\n", turbines.host.hub[i][0], turbines.host.hub[i] [1], turbines.host.hub[i][2]);
                printf("\t\tRotation speed %lf\n", turbines.host.tip_rate[i]);
            }
            if (cpctOut.is_open()) {
                printf("\tPower and Thrust output to %s\n", cpctPath.c_str());
            }
            printf("TURBINE INFO END\n");
        }
    }

    void writePowerThrust(Real t, Real U) {
        if (!cpctOut.is_open()) {
            return;
        }

        cpctOut << t;
        for (int tid = 0; tid < turbines.n_turbine; tid ++) {
            Real P = turbines.host.torque[tid]*turbines.host.tip_rate[tid];
            Real T = turbines.host.thrust[tid];
            cpctOut << "," << P << "," << T;
        }
        cpctOut << std::endl;
    }

};

}

}

#endif