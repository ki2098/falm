#ifndef _LID_CAVITY2D2_BOUNDARY_H_
#define _LID_CAVITY2D2_BOUNDARY_H_

#include "boundaryDev.h"
#include "../../src/CPM.h"

namespace LidCavity2d2 {

static void pressureBC(
    Falm::Matrix<Falm::REAL> &p,
    Falm::CPM            &cpm,
    Falm::STREAM             *stream = nullptr
) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    if (!cpm.validNeighbour(Falm::CPM::XPLUS)) {
        Falm::STREAM estream = (stream)? stream[0] : 0;
        dev_pressureBC_E(p, pdm, cpm.gc, estream);
    }
    if (!cpm.validNeighbour(Falm::CPM::XMINUS)) {
        Falm::STREAM wstream = (stream)? stream[1] : 0;
        dev_pressureBC_W(p, pdm, cpm.gc, wstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YPLUS)) {
        Falm::STREAM nstream = (stream)? stream[2] : 0;
        dev_pressureBC_N(p, pdm, cpm.gc, nstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YMINUS)) {
        Falm::STREAM sstream = (stream)? stream[3] : 0;
        dev_pressureBC_S(p, pdm, cpm.gc, sstream);
    }

    if (stream) {
        for (Falm::INT fid = 0; fid < 6; fid ++) {
            if (!cpm.validNeighbour(fid)) {
                Falm::falmWaitStream(stream[fid]);
            }
        }
    } else {
        Falm::falmWaitStream();
    }
}

static void velocityBC(
    Falm::Matrix<Falm::REAL> &u,
    Falm::CPM            &cpm,
    Falm::STREAM             *stream = nullptr
) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    if (!cpm.validNeighbour(Falm::CPM::XPLUS)) {
        Falm::STREAM estream = (stream)? stream[0] : 0;
        dev_velocityBC_E(u, pdm, cpm.gc, estream);
    }
    if (!cpm.validNeighbour(Falm::CPM::XMINUS)) {
        Falm::STREAM wstream = (stream)? stream[1] : 0;
        dev_velocityBC_W(u, pdm, cpm.gc, wstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YPLUS)) {
        Falm::STREAM nstream = (stream)? stream[2] : 0;
        dev_velocityBC_N(u, pdm, cpm.gc, nstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YMINUS)) {
        Falm::STREAM sstream = (stream)? stream[3] : 0;
        dev_velocityBC_S(u, pdm, cpm.gc, sstream);
    }

    if (stream) {
        for (Falm::INT fid = 0; fid < 6; fid ++) {
            if (!cpm.validNeighbour(fid)) {
                Falm::falmWaitStream(stream[fid]);
            }
        }
    } else {
        Falm::falmWaitStream();
    }
}

static void forceFaceVelocityZero(
    Falm::Matrix<Falm::REAL> &uu,
    Falm::CPM            &cpm,
    Falm::STREAM             *stream = nullptr
) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    if (!cpm.validNeighbour(Falm::CPM::XPLUS)) {
        Falm::STREAM fstream = (stream)? stream[0] : 0;
        dev_forceFaceVelocityZero_E(uu, pdm, cpm.gc, fstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::XMINUS)) {
        Falm::STREAM fstream = (stream)? stream[1] : 0;
        dev_forceFaceVelocityZero_W(uu, pdm, cpm.gc, fstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YPLUS)) {
        Falm::STREAM fstream = (stream)? stream[2] : 0;
        dev_forceFaceVelocityZero_N(uu, pdm, cpm.gc, fstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YMINUS)) {
        Falm::STREAM fstream = (stream)? stream[3] : 0;
        dev_forceFaceVelocityZero_S(uu, pdm, cpm.gc, fstream);
    }

    if (stream) {
        for (Falm::INT fid = 0; fid < 6; fid ++) {
            if (!cpm.validNeighbour(fid)) {
                Falm::falmWaitStream(stream[fid]);
            }
        }
    } else {
        Falm::falmWaitStream();
    }
}

static void copyZ5(
    Falm::Matrix<Falm::REAL> &field,
    Falm::CPM            &cpm,
    Falm::STREAM             *stream = nullptr
) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    Falm::INT idxcc = Falm::IDX(0, 0, cpm.gc  , pdm.shape);
    Falm::INT idxt1 = Falm::IDX(0, 0, cpm.gc+1, pdm.shape);
    Falm::INT idxt2 = Falm::IDX(0, 0, cpm.gc+2, pdm.shape);
    Falm::INT idxb1 = Falm::IDX(0, 0, cpm.gc-1, pdm.shape);
    Falm::INT idxb2 = Falm::IDX(0, 0, cpm.gc-2, pdm.shape);
    Falm::INT slice_size = pdm.shape[0] * pdm.shape[1];
    for (Falm::INT d = 0; d < field.shape[1]; d ++) {
        Falm::falmMemcpyAsync(&field.dev(idxt1, d), &field.dev(idxcc, d), sizeof(Falm::REAL) * slice_size, Falm::MCP::Dev2Dev);
        Falm::falmMemcpyAsync(&field.dev(idxt2, d), &field.dev(idxcc, d), sizeof(Falm::REAL) * slice_size, Falm::MCP::Dev2Dev);
        Falm::falmMemcpyAsync(&field.dev(idxb1, d), &field.dev(idxcc, d), sizeof(Falm::REAL) * slice_size, Falm::MCP::Dev2Dev);
        Falm::falmMemcpyAsync(&field.dev(idxb2, d), &field.dev(idxcc, d), sizeof(Falm::REAL) * slice_size, Falm::MCP::Dev2Dev);
    }
}

}

#endif