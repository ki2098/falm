#ifndef FALM_FALM_H
#define FALM_FALM_H

#include <vector>
#include "FalmCFD.h"
#include "FalmEq.h"
#include "falmath.h"
#include "mesher/mesher.hpp"
#include "rmcp/alm.h"
#include "alm/alm.h"
#include "vcdm/VCDM.h"
#include "FalmIO.h"

namespace Falm {

struct FalmBasicVar {
    Matrix<Real> xyz, kx, g, ja; // Non-uniform structured cartesian mesh variables
    Matrix<Real> u, uu, uc, p, nut; // basic physical fields
    Matrix<Real> poi_a, poi_rhs, poi_res; // variables for pressure poisson equation
    Matrix<Real> ff; // actuator model force
    Matrix<Real> divergence; // velocity divergence
    Matrix<Real> utavg, ptavg; // time-averaged variables
    Matrix<Real> uvwp; // output buffer

    void release_all() {
        xyz.release();
        kx.release();
        g.release();
        ja.release();
        u.release();
        uu.release();
        p.release();
        nut.release();
        poi_a.release();
        poi_rhs.release();
        poi_res.release();
        ff.release();
        divergence.release();
        utavg.release();
        ptavg.release();
        uvwp.release();
    }
};

struct FalmMeshInfo {
    bool convert;
    std::string convertSrc;
    std::string cvFile;
    std::string cvCenter;
};

class FalmCore {
public:
    Json               params;
    FalmCFD           falmCfd;
    FalmEq             falmEq;
    FalmMeshInfo falmMeshInfo;
    CPM                   cpm;
    FalmBasicVar           fv;
    FalmBaseMesh    gBaseMesh;
    FalmBaseMesh     baseMesh;
    
    Int it;
    Real dt;
    Real startTime;
    Int  maxIt;
    Int  timeAvgStartIt;
    Int  timeAvgEndIt;
    Int  timeAvgCount = 0;
    Int  outputStartIt;
    Int  outputEndIt;
    Int  outputIntervalIt;
    std::string     workdir;
    std::string outputPrefix;
    std::string    setupFile;
    std::string       cvFile;
    std::vector<FalmSnapshotInfo> timeSlices;
    int ngpu;
    int gpuid;

    Real gettime() {return dt * it;}

    bool is_time_averaging() {
        return (it >= timeAvgStartIt && it <= timeAvgEndIt);
    }

    void env_init(int &argc, char **&argv) {
        CPM_Init(&argc, &argv, cpm);
        cudaGetDeviceCount(&ngpu);
        gpuid = cpm.rank % ngpu;
        cudaSetDevice(gpuid);
        timeSlices = {};
        FalmMV::init();
    }

    void env_finalize() {
        if (cpm.rank == 0) {
            FalmIO::writeIndexFile(wpath(outputPrefix + ".json"), cpm, timeSlices);
        }
        CPM_Barrier(MPI_COMM_WORLD);
        FalmMV::release();
        fv.release_all();
        falmEq.release();
        falmCfd.release();
        gBaseMesh.release();
        baseMesh.release();
        CPM_Finalize(cpm);

        std::string message_file_name = outputPrefix + "_endmsg.txt";
        std::ofstream mfs(message_file_name);
        mfs << "Falm/Main ends normally.";
        mfs.close();
    }

    std::string wpath(const std::string &str) {
        if (str[0] == '/') {
           return str;
        } else if (workdir.back() == '/') {
            return workdir + str;
        } else {
            return workdir + "/" + str;
        }
    }

    void print_info(int output_rank = 0) {
        if (cpm.rank == output_rank) {
            printf("SETUP INFO START\n");

            printf("Working dir %s\n", workdir.c_str());
            printf("Read setup file %s\n", wpath(setupFile).c_str());
            printf("Control Params:\n");
            printf("\tdt %e\n", dt);
            printf("\tstart time %e\n", startTime);
            printf("\titeration number %d\n", maxIt);
            printf("\ttime avg start iteration %d\n", timeAvgStartIt);
            printf("\ttime avg end iteration %d\n", timeAvgEndIt);
            printf("\toutput start iteration %d\n", outputStartIt);
            printf("\toutput end iteration %d\n", outputEndIt);
            printf("\toutput every %d iterations\n", outputIntervalIt);
            printf("\toutput to %s\n", wpath(outputPrefix).c_str());

            printf("CFD Solver Params\n");
            printf("\tRe = %e, 1/Re = %e\n", falmCfd.Re, falmCfd.ReI);
            printf("\tAdvection scheme %s\n", FalmCFD::advscheme2str(falmCfd.AdvScheme).c_str());
            printf("\tSGS model %s\n", FalmCFD::sgs2str(falmCfd.SGSModel).c_str());
            if (falmCfd.SGSModel == SGSType::Smagorinsky) {
                printf("\tCsmagorinsky %e\n", falmCfd.CSmagorinsky);
            }

            printf("Poisson Solver Params\n");
            printf("\tsolver type %s\n", FalmEq::type2str(falmEq.type).c_str());
            printf("\tmax iteration number %d\n", falmEq.maxit);
            printf("\ttolerance %e\n", falmEq.tol);
            if (falmEq.type == LSType::SOR) {
                printf("\trelaxation factor %e\n", falmEq.relax_factor);
            }
            if (falmEq.type == LSType::PBiCGStab) {
                printf("Preconditioner Params\n");
                printf("\tsolver type %s\n", FalmEq::type2str(falmEq.pc_type).c_str());
                printf("\titeration number %d\n", falmEq.pc_maxit);
                if (falmEq.pc_type == LSType::SOR) {
                    printf("\trelaxation factor %e\n", falmEq.pc_relax_factor);
                }
            }
            if (params.contains("inflow")) {
                printf("Inflow Params\n");
                printf("\tProfile %s\n", params["inflow"]["type"].get<std::string>().c_str());
                printf("\tVelocity %e\n", params["inflow"]["velocity"].get<Real>());
            }
            printf("SETUP INFO END\n");
        }

        // int *ngh = (int*)malloc(sizeof(int)*6*cpm.size);
        std::vector<int> ngh(6*cpm.size);
        // MPI_Gather(cpm.neighbour, 6, MPI_INT, ngh, 6, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Allgather(cpm.neighbour, 6, MPI_INT, ngh.data(), 6, MPI_INT, MPI_COMM_WORLD);

        if (cpm.rank == output_rank) {
            printf("MPI INFO START\n");
            Int gc = cpm.gc;
            printf("\tGlobal voxel (%d %d %d)\n", cpm.global.shape[0], cpm.global.shape[1], cpm.global.shape[2]);
            printf("\tGlobal inner voxel (%d %d %d)\n", cpm.global.shape[0] - 2*gc, cpm.global.shape[1] - 2*gc, cpm.global.shape[2] - 2*gc);
            printf("\tGuide voxel length %d\n", cpm.gc);
            printf("\tMPI cluster shape (%d %d %d)\n", cpm.shape[0], cpm.shape[1], cpm.shape[2]);
            for (int i = 0; i < cpm.size; i ++) {
                printf("\tRank %d\n", i);
                printf("\t\tMPI idx (%d %d %d)\n", cpm.idx[0], cpm.idx[1], cpm.idx[2]);
                const Region &pdm = cpm.pdm_list[i];
                printf("\t\tVoxel (%d %d %d)\n", pdm.shape[0], pdm.shape[1], pdm.shape[2]);
                printf("\t\tInner voxel (%d %d %d)\n", pdm.shape[0] - 2*gc, pdm.shape[1] - 2*gc, pdm.shape[2] - 2*gc);
                printf("\t\tOffset (%d %d %d)\n", pdm.offset[0], pdm.offset[1], pdm.offset[2]);
                printf("\t\tNeighbour [%d %d %d %d %d %d]\n", ngh[i*6], ngh[i*6+1], ngh[i*6+2], ngh[i*6+3], ngh[i*6+4], ngh[i*6+5]);
            }
            printf("Using Cuda-aware-MPI %d\n", cpm.use_cuda_aware_mpi);
            printf("MPI INFO END\n");
            // printf("reduction buffer size %d\n", FalmMV::reduction_buffer_size);
        }

        if (cpm.rank == output_rank) {
            printf("GPU INFO START\n");
            printf("\t%d GPUs in 1 Node\n", ngpu);
            for (int i = 0; i < cpm.size; i ++) {
                printf("\tRank %d on GPU %d\n", i, i%ngpu);
            }
            printf("GPU INFO END\n");
        }

        // if (cpm.rank == output_rank && params.contains("turbine")) {
        //     printf("TURBINE INFO START\n");
        //     printf("\tTurbine blade length %lf\n", params["turbine"]["radius"].get<REAL>());
        //     printf("\tTurbine radial velocity %lf\n", params["turbine"]["radialVelocity"].get<REAL>());
        //     json position = params["turbine"]["position"];
        //     printf("\tTurbine at (%lf %lf %lf)\n", position[0].get<REAL>(), position[1].get<REAL>(), position[2].get<REAL>());
        //     printf("\tTurbine property file: %s\n", wpath(params["turbine"]["properties"]).c_str());
        //     printf("TURBINE INFO END\n");
        // }


        // free(ngh);
    }

    Int parse_settings(std::string setup_file_path) {
        int cutat = setup_file_path.find_last_of('/');
        if (cutat == std::string::npos) {
            workdir = ".";
            setupFile = setup_file_path;
        } else if (cutat == 0) {
            workdir = "/";
            setupFile = setup_file_path.substr(cutat + 1);
        } else {
            workdir = setup_file_path.substr(0, cutat);
            setupFile = setup_file_path.substr(cutat + 1);
        }
        std::ifstream setupfile(wpath(setupFile));
        if (!setupfile) {
            return FalmErr::setUpFileErr;
        }
        params = Json::parse(setupfile);

    {
        auto runprm = params["runtime"];
        startTime = runprm["time"]["start"];
        dt = runprm["time"]["dt"];
        maxIt = Int((runprm["time"]["end"].get<Real>() - startTime) / dt);
        if (runprm.contains("timeAvg")) {
            if (runprm["timeAvg"].contains("start")) {
                Real tavgstart = runprm["timeAvg"]["start"];
                if (tavgstart < startTime) {
                    tavgstart = startTime;
                }
                timeAvgStartIt = Int((tavgstart - startTime) / dt);
            } else {
                timeAvgStartIt = 0;
            }
            if (runprm["timeAvg"].contains("end")) {
                Real tavgend = runprm["timeAvg"]["end"];
                timeAvgEndIt = Int((tavgend - startTime) / dt);
            } else {
                timeAvgEndIt = maxIt;
            }
        } else {
            timeAvgStartIt = 0;
            timeAvgEndIt = -1;
        }
        if (runprm.contains("output")) {
            if (runprm["output"].contains("start")) {
                Real ostart = runprm["output"]["start"];
                if (ostart < startTime) {
                    ostart = startTime;
                }
                outputStartIt = Int((ostart - startTime) / dt);
            } else {
                outputStartIt = 0;
            }
            if (runprm["output"].contains("end")) {
                Real oend = runprm["output"]["end"];
                outputEndIt = Int((oend - startTime) / dt);
            } else {
                outputEndIt = maxIt;
            }
            outputIntervalIt = Int(runprm["output"]["interval"].get<Real>() / dt);
            outputPrefix = runprm["output"]["prefix"];
        } else {
            outputStartIt = 0;
            outputEndIt = -1;
        }
    }

    {
        auto lsprm = params["solver"]["linearSolver"];
        FalmEq tmp;
        tmp.type = FalmEq::str2type(lsprm["type"]);
        tmp.maxit = lsprm["iteration"];
        tmp.tol = lsprm["tolerance"];
        if (tmp.type == LSType::SOR) {
            tmp.relax_factor = lsprm["relaxationFactor"];
        }
        if (tmp.type == LSType::PBiCGStab) {
            auto pcprm = lsprm["preconditioner"];
            tmp.pc_type = FalmEq::str2type(pcprm["type"]);
            tmp.pc_maxit = pcprm["iteration"];
            if (tmp.pc_type == LSType::SOR) {
                tmp.pc_relax_factor = pcprm["relaxationFactor"];
            }
        } else {
            tmp.pc_type = LSType::Empty;
        }
        falmEq.init(tmp.type, tmp.maxit, tmp.tol, tmp.relax_factor, tmp.pc_type, tmp.pc_maxit, tmp.pc_relax_factor);
    }

    {
        auto cfdprm = params["solver"]["cfd"];
        FalmCFD tmp;
        tmp.Re = cfdprm["Re"];
        tmp.AdvScheme = FalmCFD::str2advscheme(cfdprm["advectionScheme"]);
        if (cfdprm.contains("SGS")) {
            tmp.SGSModel = FalmCFD::str2sgs(cfdprm["SGS"]);
        } else {
            tmp.SGSModel = SGSType::Empty;
        }
        if (tmp.SGSModel == SGSType::Smagorinsky) {
            tmp.CSmagorinsky = cfdprm["Cs"];
        }
        falmCfd.init(tmp.Re, tmp.AdvScheme, tmp.SGSModel, tmp.CSmagorinsky);
    }

    {
        auto meshprm = params["mesh"];
        falmMeshInfo.cvFile = meshprm["controlVolumeFile"];
        falmMeshInfo.cvCenter = meshprm["controlVolumeCenter"];
        if (meshprm.contains("convert")) {
            falmMeshInfo.convert = true;
            falmMeshInfo.convertSrc = meshprm["convert"];
        }
    }

        setupfile.close();

        return FalmErr::success;
    }

    void computation_init(const Int3 &division, int gc) {
        if (falmMeshInfo.convert && cpm.rank == 0) {
            Mesher::build_mesh(falmMeshInfo.cvCenter, wpath(falmMeshInfo.convertSrc), wpath(falmMeshInfo.cvFile), gc);
        }
        CPM_Barrier(MPI_COMM_WORLD);
        Int3 idmax;
        Int _gc;
        FalmIO::readControlVolumeFile(wpath(falmMeshInfo.cvFile), gBaseMesh, idmax, _gc);
        gBaseMesh.sync(MCP::Hst2Dev);

        assert(_gc == gc);
        cpm.initPartition(idmax, gc, division);
        Int3 &shape  = cpm.pdm_list[cpm.rank].shape;
        Int3 &offset = cpm.pdm_list[cpm.rank].offset;
        if (cpm.rank == 0 && outputEndIt >= outputStartIt) {
            FalmIO::writeIndexFile(wpath(outputPrefix + ".json"), cpm, timeSlices);
            FalmIO::writeControlVolumeFile(wpath(outputPrefix + ".cv"), gBaseMesh, idmax, gc);
            FalmIO::writeSetupFile(wpath(outputPrefix + "_setup.json"), params);
        }
        
        fv.xyz.alloc(shape, 3, HDC::HstDev, "coordinate");
        fv.kx.alloc(shape, 3, HDC::HstDev, "d kxi / d x");
        fv.g.alloc(shape, 3, HDC::HstDev, "metric tensor");
        fv.ja.alloc(shape, 1, HDC::HstDev, "jacobian");
        printf("(%d %d %d) (%d %d %d)\n", shape[0], shape[1], shape[2], offset[0], offset[1], offset[1]);

        baseMesh.alloc(shape[0], shape[1], shape[2], HDC::HstDev);
        for (Int i = 0; i < shape[0]; i ++) {
            baseMesh.x(i) = gBaseMesh.x(i + offset[0]);
            baseMesh.hx(i) = gBaseMesh.hx(i + offset[0]);
        }
        for (Int j = 0; j < shape[1]; j ++) {
            baseMesh.y(j) = gBaseMesh.y(j + offset[1]);
            baseMesh.hy(j) = gBaseMesh.hy(j + offset[1]);
        }
        for (Int k = 0; k < shape[2]; k ++) {
            baseMesh.z(k) = gBaseMesh.z(k + offset[2]);
            baseMesh.hz(k) = gBaseMesh.hz(k + offset[2]);
        }
        baseMesh.sync(MCP::Hst2Dev);

        for (Int k = 0; k < shape[2]; k ++) {
        for (Int j = 0; j < shape[1]; j ++) {
        for (Int i = 0; i < shape[0]; i ++) {
            Int idx = IDX(i, j, k, shape);
            fv.xyz(idx, 0) = gBaseMesh.x(i + offset[0]);
            fv.xyz(idx, 1) = gBaseMesh.y(j + offset[1]);
            fv.xyz(idx, 2) = gBaseMesh.z(k + offset[2]);
            Real3 pitch;
            pitch[0] = gBaseMesh.hx(i + offset[0]);
            pitch[1] = gBaseMesh.hy(j + offset[1]);
            pitch[2] = gBaseMesh.hz(k + offset[2]);
            Real volume = PRODUCT3(pitch);
            Real3 dkdx = Real(1) / pitch;
            fv.ja(idx) = volume;
            // printf("%d %d %d %e %e %e\n", i + offset[0], j + offset[1], k + offset[2], gBaseMesh.hx(i + offset[0]), gBaseMesh.hy(j + offset[1]), gBaseMesh.hz(k + offset[2]));
            fv.g(idx, 0) = volume*dkdx[0]*dkdx[0];
            fv.g(idx, 1) = volume*dkdx[1]*dkdx[1];
            fv.g(idx, 2) = volume*dkdx[2]*dkdx[2];
            fv.kx(idx, 0) = dkdx[0];
            fv.kx(idx, 1) = dkdx[1];
            fv.kx(idx, 2) = dkdx[2];
        }}}

        fv.xyz.sync(MCP::Hst2Dev);
        fv.kx.sync(MCP::Hst2Dev);
        fv.g.sync(MCP::Hst2Dev);
        fv.ja.sync(MCP::Hst2Dev);

        fv.u.alloc(shape, 3, HDC::HstDev, "velocity");
        fv.uu.alloc(shape, 3, HDC::HstDev, "contravariant face velocity");
        fv.p.alloc(shape, 1, HDC::HstDev, "pressure");
        fv.nut.alloc(shape, 1, HDC::HstDev, "turbulence viscosity");
        fv.poi_a.alloc(shape, 7, HDC::HstDev, "poisson coefficients", StencilMatrix::D3P7);
        fv.poi_rhs.alloc(shape, 1, HDC::HstDev, "poisson rhs");
        fv.poi_res.alloc(shape, 1, HDC::HstDev, "poisson residual");
        fv.ff.alloc(shape, 3, HDC::HstDev, "ALM force");
        fv.divergence.alloc(shape, 1, HDC::HstDev, "divergence");

        if (timeAvgEndIt > timeAvgStartIt) {
            fv.utavg.alloc(shape, 3, HDC::Device, "time-avg velocity");
            fv.ptavg.alloc(shape, 1, HDC::Device, "time-avg pressure");
            fv.utavg.clear(HDC::Device);
            fv.ptavg.clear(HDC::Device);
        }
        fv.uvwp.alloc(shape, 4, HDC::HstDev, "uvwp buffer");

        falmCfd.alloc(shape);
        falmEq.alloc(shape);
    }

    void TAvg(dim3 block={8,8,8}) {
        if (is_time_averaging()) {
            // FalmMV::MatrixAdd(fv.utavg, fv.u, block);
            // FalmMV::MatrixAdd(fv.ptavg, fv.p, block);
            // timeAvgCount ++;
            timeAvgCount = it - timeAvgStartIt + 1;
            Real b = 1./timeAvgCount;
            Real a = 1. - b;
            FalmMV::Vecaxby(a, fv.utavg, b, fv.u, fv.utavg, block);
            FalmMV::Vecaxby(a, fv.ptavg, b, fv.p, fv.ptavg, block);
        }
    }

    void outputUVWP(dim3 block={8,8,8}) {
        if (it < outputStartIt || it > outputEndIt || it % outputIntervalIt != 0) {
            return;
        }
        
        Int3 shape = cpm.pdm_list[cpm.rank].shape;
        // Matrix<REAL> uvwp(shape, 4, HDC::Host, "output uvwp");
        falmMemcpy(&fv.uvwp(0, 0), &fv.u.dev(0), sizeof(Real) * fv.uvwp.shape[0] * 3, MCP::Dev2Hst);
        falmMemcpy(&fv.uvwp(0, 3), &fv.p.dev(0), sizeof(Real) * fv.uvwp.shape[0]    , MCP::Dev2Hst);
        size_t len = outputPrefix.size() + 32;
        // char *tmp = (char*)malloc(sizeof(char) * len);
        std::vector<char> tmp(len);
        sprintf(tmp.data(), "%s_%06d_%010d", outputPrefix.c_str(), cpm.rank, it);
        std::string fpath(tmp.data());
        FalmIO::writeVectorFile(wpath(fpath), cpm, fv.uvwp, it, gettime());
        // free(tmp);
        FalmSnapshotInfo snapshot = {it, gettime(), false};
        if (is_time_averaging()) {
            falmMemcpy(&fv.uvwp.dev(0, 0), &fv.utavg.dev(0), sizeof(Real) * fv.uvwp.shape[0] * 3, MCP::Dev2Dev);
            falmMemcpy(&fv.uvwp.dev(0, 3), &fv.ptavg.dev(0), sizeof(Real) * fv.uvwp.shape[0]    , MCP::Dev2Dev);
            // FalmMV::ScaleMatrix(fv.uvwp, 1.0 / timeAvgCount, block);
            fv.uvwp.sync(MCP::Dev2Hst);
            std::string tAvgPrefix = outputPrefix + "_tavg";
            len = tAvgPrefix.size() + 32;
            tmp = std::vector<char>(len);
            sprintf(tmp.data(), "%s_%06d_%010d", tAvgPrefix.c_str(), cpm.rank, it);
            fpath = std::string(tmp.data());
            FalmIO::writeVectorFile(wpath(fpath), cpm, fv.uvwp, it, gettime());
            // free(tmp);
            snapshot.tavg = true;
        }
        // std::pair<INT, REAL> pt{it, it * dt};
        timeSlices.push_back(snapshot);
    }



};

}

#endif