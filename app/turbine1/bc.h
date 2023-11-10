#ifndef FALM_APP_TURBINE1_BC_H
#define FALM_APP_TURBINE1_BC_H

#include "bcdev.h"
#include "../../src/CPM.h"

namespace TURBINE1 {

static void ubc(Falm::Matrix<Falm::REAL> &u, Falm::Matrix<Falm::REAL> &uprev, Falm::Matrix<Falm::REAL> &x, Falm::REAL dt, Falm::CPM &cpm, Falm::STREAM *stream = nullptr) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    Falm::INT &gc = cpm.gc;
    for (Falm::INT fid = 0; fid < Falm::CPM::NFACE; fid ++) {
        if (!cpm.validNeighbour(fid)) {
            Falm::STREAM s = (stream)? stream[fid] : 0;
            if (fid == Falm::CPM::XMINUS) {
                ubc_xminus(u, pdm, gc, s);
            } else if (fid == Falm::CPM::XPLUS) {
                ubc_xplus (u, uprev, x, dt, pdm, gc, s);
            } else if (fid == Falm::CPM::YMINUS) {
                ubc_yminus(u, pdm, gc, s);
            } else if (fid == Falm::CPM::YPLUS) {
                ubc_yplus (u, pdm, gc, s);
            } else if (fid == Falm::CPM::ZMINUS) {
                ubc_zminus(u, pdm, gc, s);
            } else if (fid == Falm::CPM::ZPLUS) {
                ubc_zplus (u, pdm, gc, s);
            }
        }
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

static void pbc(Falm::Matrix<Falm::REAL> &p, Falm::CPM &cpm, Falm::STREAM *stream = nullptr) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    Falm::INT &gc = cpm.gc;
    for (Falm::INT fid = 0; fid < Falm::CPM::NFACE; fid ++) {
        if (!cpm.validNeighbour(fid)) {
            Falm::STREAM s = (stream)? stream[fid] : 0;
            if (fid == Falm::CPM::XMINUS) {
                pbc_xminus(p, pdm, gc, s);
            } else if (fid == Falm::CPM::XPLUS) {
                pbc_xplus (p, pdm, gc, s);
            } else if (fid == Falm::CPM::YMINUS) {
                pbc_yminus(p, pdm, gc, s);
            } else if (fid == Falm::CPM::YPLUS) {
                pbc_yplus (p, pdm, gc, s);
            } else if (fid == Falm::CPM::ZMINUS) {
                pbc_zminus(p, pdm, gc, s);
            } else if (fid == Falm::CPM::ZPLUS) {
                pbc_zplus (p, pdm, gc, s);
            }
        }
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