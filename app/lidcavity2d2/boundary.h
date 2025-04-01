#ifndef _LID_CAVITY2D2_BOUNDARY_H_
#define _LID_CAVITY2D2_BOUNDARY_H_

#include "boundaryDev.h"
#include "../../src/CPM.h"

namespace LidCavity2d2 {

static void pressureBC(
    Falm::Matrix<Falm::Real> &p,
    Falm::CPM            &cpm,
    Falm::Stream             *stream = nullptr
) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    if (!cpm.validNeighbour(Falm::CPM::XPLUS)) {
        Falm::Stream estream = (stream)? stream[0] : 0;
        dev_pressureBC_E(p, pdm, cpm.gc, estream);
    }
    if (!cpm.validNeighbour(Falm::CPM::XMINUS)) {
        Falm::Stream wstream = (stream)? stream[1] : 0;
        dev_pressureBC_W(p, pdm, cpm.gc, wstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YPLUS)) {
        Falm::Stream nstream = (stream)? stream[2] : 0;
        dev_pressureBC_N(p, pdm, cpm.gc, nstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YMINUS)) {
        Falm::Stream sstream = (stream)? stream[3] : 0;
        dev_pressureBC_S(p, pdm, cpm.gc, sstream);
    }

    if (stream) {
        for (Falm::Int fid = 0; fid < 6; fid ++) {
            if (!cpm.validNeighbour(fid)) {
                Falm::falmWaitStream(stream[fid]);
            }
        }
    } else {
        Falm::falmWaitStream();
    }
}

static void velocityBC(
    Falm::Matrix<Falm::Real> &u,
    Falm::CPM            &cpm,
    Falm::Stream             *stream = nullptr
) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    if (!cpm.validNeighbour(Falm::CPM::XPLUS)) {
        Falm::Stream estream = (stream)? stream[0] : 0;
        dev_velocityBC_E(u, pdm, cpm.gc, estream);
    }
    if (!cpm.validNeighbour(Falm::CPM::XMINUS)) {
        Falm::Stream wstream = (stream)? stream[1] : 0;
        dev_velocityBC_W(u, pdm, cpm.gc, wstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YPLUS)) {
        Falm::Stream nstream = (stream)? stream[2] : 0;
        dev_velocityBC_N(u, pdm, cpm.gc, nstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YMINUS)) {
        Falm::Stream sstream = (stream)? stream[3] : 0;
        dev_velocityBC_S(u, pdm, cpm.gc, sstream);
    }

    if (stream) {
        for (Falm::Int fid = 0; fid < 6; fid ++) {
            if (!cpm.validNeighbour(fid)) {
                Falm::falmWaitStream(stream[fid]);
            }
        }
    } else {
        Falm::falmWaitStream();
    }
}

static void forceFaceVelocityZero(
    Falm::Matrix<Falm::Real> &uu,
    Falm::CPM            &cpm,
    Falm::Stream             *stream = nullptr
) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    if (!cpm.validNeighbour(Falm::CPM::XPLUS)) {
        Falm::Stream fstream = (stream)? stream[0] : 0;
        dev_forceFaceVelocityZero_E(uu, pdm, cpm.gc, fstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::XMINUS)) {
        Falm::Stream fstream = (stream)? stream[1] : 0;
        dev_forceFaceVelocityZero_W(uu, pdm, cpm.gc, fstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YPLUS)) {
        Falm::Stream fstream = (stream)? stream[2] : 0;
        dev_forceFaceVelocityZero_N(uu, pdm, cpm.gc, fstream);
    }
    if (!cpm.validNeighbour(Falm::CPM::YMINUS)) {
        Falm::Stream fstream = (stream)? stream[3] : 0;
        dev_forceFaceVelocityZero_S(uu, pdm, cpm.gc, fstream);
    }

    if (stream) {
        for (Falm::Int fid = 0; fid < 6; fid ++) {
            if (!cpm.validNeighbour(fid)) {
                Falm::falmWaitStream(stream[fid]);
            }
        }
    } else {
        Falm::falmWaitStream();
    }
}

static void copyZ5(
    Falm::Matrix<Falm::Real> &field,
    Falm::CPM            &cpm,
    Falm::Stream             *stream = nullptr
) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    Falm::Int idxcc = Falm::IDX(0, 0, cpm.gc  , pdm.shape);
    Falm::Int idxt1 = Falm::IDX(0, 0, cpm.gc+1, pdm.shape);
    Falm::Int idxt2 = Falm::IDX(0, 0, cpm.gc+2, pdm.shape);
    Falm::Int idxb1 = Falm::IDX(0, 0, cpm.gc-1, pdm.shape);
    Falm::Int idxb2 = Falm::IDX(0, 0, cpm.gc-2, pdm.shape);
    Falm::Int slice_size = pdm.shape[0] * pdm.shape[1];
    for (Falm::Int d = 0; d < field.shape[1]; d ++) {
        Falm::falmMemcpyAsync(&field.dev(idxt1, d), &field.dev(idxcc, d), sizeof(Falm::Real) * slice_size, Falm::MCP::Dev2Dev);
        Falm::falmMemcpyAsync(&field.dev(idxt2, d), &field.dev(idxcc, d), sizeof(Falm::Real) * slice_size, Falm::MCP::Dev2Dev);
        Falm::falmMemcpyAsync(&field.dev(idxb1, d), &field.dev(idxcc, d), sizeof(Falm::Real) * slice_size, Falm::MCP::Dev2Dev);
        Falm::falmMemcpyAsync(&field.dev(idxb2, d), &field.dev(idxcc, d), sizeof(Falm::Real) * slice_size, Falm::MCP::Dev2Dev);
    }
}

}

#endif