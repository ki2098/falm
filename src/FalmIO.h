#ifndef FALM_FALMIO_H
#define FALM_FALMIO_H

#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include "typedef.h"
#include "matrix.h"
#include "CPMBase.h"
#include "FalmMeshBase.h"

namespace Falm {

class FalmIO {
public:

static void writeSetupFile(std::string fpath, json params) {
    std::ofstream jfile(fpath);
    jfile << params.dump(2);
    jfile.close();
}

static void writeIndexFile(std::string fpath, CPM &cpm, const std::vector<FalmSnapshotInfo> &itlist) {
    json idxjson;
    INT gimax, gjmax, gkmax, gc;
    gimax = cpm.global.shape[0];
    gjmax = cpm.global.shape[1];
    gkmax = cpm.global.shape[2];
    gc = cpm.gc;
    idxjson["global"] = {gimax-2*gc, gjmax-2*gc, gkmax-2*gc};
    idxjson["guidePoint"] = gc;
    json jranks = json::array();
    for (int i = 0; i < cpm.size; i ++) {
        json tmp;
        tmp["rank"] = i;
        INT3 &shape = cpm.pdm_list[i].shape;
        INT3 &offset = cpm.pdm_list[i].offset;
        tmp["voxel"] = {shape[0]-2*gc, shape[1]-2*gc, shape[2]-2*gc};
        tmp["offset"] = {offset[0], offset[1], offset[2]};
        jranks.push_back(tmp);
    }
    idxjson["ranks"] = jranks;
    // idxjson["outputSteps"] = {};
    for (auto snapshot : itlist) {
        json spt;
        spt["time"] = snapshot.time;
        spt["step"] = snapshot.step;
        if (snapshot.tavg) {
            spt["timeAvg"] = 1;
        }
        idxjson["outputSteps"].push_back(spt);
    }
    if (typeid(REAL) == typeid(float)) {
        idxjson["dateType"] = "float32";
    } else if (typeid(REAL) == typeid(double)) {
        idxjson["dataType"] = "float64";
    } else {
        idxjson["dataType"] = "undefined";
    }
    idxjson["variables"] = {"u", "v", "w", "p"};

    std::ofstream jfile(fpath);
    jfile << idxjson.dump(2);
    jfile.close();
}

static void writeVectorFile(std::string fpath, CPM &cpm, Matrix<REAL> &v, size_t step, double time) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    size_t imax, jmax, kmax;
    imax = pdm.shape[0];
    jmax = pdm.shape[1];
    kmax = pdm.shape[2];
    size_t N = v.shape[1];

    std::ofstream vfile(fpath, std::ios::binary | std::ios::out);
    size_t dsize = sizeof(REAL);
    size_t gc = cpm.gc;
    vfile.write((char*)&imax, sizeof(size_t));
    vfile.write((char*)&jmax, sizeof(size_t));
    vfile.write((char*)&kmax, sizeof(size_t));
    vfile.write((char*)&N, sizeof(size_t));
    vfile.write((char*)&gc, sizeof(size_t));
    vfile.write((char*)&step, sizeof(size_t));
    vfile.write((char*)&time, sizeof(double));    
    vfile.write((char*)&dsize, sizeof(size_t));
    VECTOR<size_t, 3> shape{{imax, jmax, kmax}};
    for (size_t n = 0; n < N; n ++) {
    for (size_t k = 0; k < kmax; k ++) {
    for (size_t j = 0; j < jmax; j ++) {
    for (size_t i = 0; i < imax; i ++) {
        size_t idx = IDX(i, j, k, shape);
        vfile.write((char*)&v(idx, n), sizeof(REAL));
    }}}}
    vfile.close();
}

static void readControlVolumeFile(std::string srcpath, FalmBaseMesh &mesh, INT3 &idmax, INT &gc) {
    std::ifstream cvfs(srcpath);
    cvfs >> idmax[0] >> idmax[1] >> idmax[2] >> gc;
    mesh.alloc(idmax[0], idmax[1], idmax[2], HDC::HstDev);
    for (int i = 0; i < idmax[0]; i ++) {
        cvfs >> mesh.x(i) >> mesh.hx(i);
        // printf("%e %e\n", mesh.x(i), mesh.hx(i));
    }
    for (int j = 0; j < idmax[1]; j ++) {
        cvfs >> mesh.y(j) >> mesh.hy(j);
        // printf("%e %e\n", mesh.y(j), mesh.hy(j));
    }
    for (int k = 0; k < idmax[2]; k ++) {
        cvfs >> mesh.z(k) >> mesh.hz(k);
        // printf("%e %e\n", mesh.z(k), mesh.hz(k));
    }
    cvfs.close();
}

static void writeControlVolumeFile(std::string path, FalmBaseMesh &mesh, INT3 &idmax, INT gc) {
    FILE *cvfile = fopen(path.c_str(), "w");
    if (cvfile) {
        fprintf(cvfile, "%d %d %d %d\n", idmax[0], idmax[1], idmax[2], gc);
            for (int i = 0; i < idmax[0]; i ++) {
                fprintf(cvfile, "\t%.15e\t%.15e\n", mesh.x(i), mesh.hx(i));
            }
            for (int j = 0; j < idmax[1]; j ++) {
                fprintf(cvfile, "\t%.15e\t%.15e\n", mesh.y(j), mesh.hy(j));
            }
            for (int k = 0; k < idmax[2]; k ++) {
                fprintf(cvfile, "\t%.15e\t%.15e\n", mesh.z(k), mesh.hz(k));
            }
    }
    fclose(cvfile);
}

};

}

#endif