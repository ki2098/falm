#ifndef _LID_CAVITY2D2_BOUNDARY_H_
#define _LID_CAVITY2D2_BOUNDARY_H_

#include "boundaryDev.h"
#include "../../src/CPM.h"

namespace TURBINE1 {

static void pressure_bc(
    Falm::Matrix<Falm::REAL> &p, Falm::CPM &cpm, Falm::STREAM *stream = nullptr
) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    if (!cpm.validNeighbour(Falm::CPM::XMINUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::XMINUS] : 0;
        pbc_xminus(p, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::XPLUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::XPLUS] : 0;
        pbc_xplus(p, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::YMINUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::YMINUS] : 0;
        pbc_yminus(p, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::YPLUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::YPLUS] : 0;
        pbc_yplus(p, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::ZMINUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::ZMINUS] : 0;
        pbc_zminus(p, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::ZPLUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::ZPLUS] : 0;
        pbc_zplus(p, pdm, cpm.gc, s);
    }

    if (stream) {
        for (Falm::INT fid = 0; fid < Falm::CPM::NFACE; fid ++) {
            if (!cpm.validNeighbour(fid)) {
                Falm::falmWaitStream(stream[fid]);
            }
        }
    } else {
        Falm::falmWaitStream();
    }
}

static void velocity_bc(
    Falm::Matrix<Falm::REAL> &u, Falm::Matrix<Falm::REAL> &u_previous, Falm::Matrix<Falm::REAL> &x,
    Falm::REAL dt, Falm::CPM &cpm, Falm::STREAM *stream = nullptr
) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    if (!cpm.validNeighbour(Falm::CPM::XMINUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::XMINUS] : 0;
        vbc_xminus(u, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::XPLUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::XPLUS] : 0;
        vbc_xplus(u, u_previous, x, dt, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::YMINUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::YMINUS] : 0;
        vbc_yminus(u, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::YPLUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::YPLUS] : 0;
        vbc_yplus(u, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::ZMINUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::ZMINUS] : 0;
        vbc_zminus(u, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::ZPLUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::ZPLUS] : 0;
        vbc_zplus(u, pdm, cpm.gc, s);
    }

    if (stream) {
        for (Falm::INT fid = 0; fid < Falm::CPM::NFACE; fid ++) {
            if (!cpm.validNeighbour(fid)) {
                Falm::falmWaitStream(stream[fid]);
            }
        }
    } else {
        Falm::falmWaitStream();
    }
}

}

#endif