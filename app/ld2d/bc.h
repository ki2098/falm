#ifndef FALM_APP_LD2D_BC_H
#define FALM_APP_LD2D_BC_H

#include "bcdevcall.h"
#include "../../src/CPM.h"

static void pbc(Falm::Matrix<Falm::REAL> &p, Falm::CPM &cpm, Falm::STREAM *stream = nullptr) {
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

static void ubc(Falm::Matrix<Falm::REAL> &u, Falm::CPM &cpm, Falm::STREAM *stream = nullptr) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    if (!cpm.validNeighbour(Falm::CPM::XMINUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::XMINUS] : 0;
        ubc_xminus(u, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::XPLUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::XPLUS] : 0;
        ubc_xplus(u, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::YMINUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::YMINUS] : 0;
        ubc_yminus(u, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::YPLUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::YPLUS] : 0;
        ubc_yplus(u, pdm, cpm.gc, s);
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

static void uubc(Falm::Matrix<Falm::REAL> &uu, Falm::CPM &cpm, Falm::STREAM *stream = nullptr) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    if (!cpm.validNeighbour(Falm::CPM::XMINUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::XMINUS] : 0;
        uubc_xminus(uu, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::XPLUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::XPLUS] : 0;
        uubc_xplus(uu, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::YMINUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::YMINUS] : 0;
        uubc_yminus(uu, pdm, cpm.gc, s);
    }
    if (!cpm.validNeighbour(Falm::CPM::YPLUS)) {
        Falm::STREAM s = (stream)? stream[Falm::CPM::YPLUS] : 0;
        uubc_yplus(uu, pdm, cpm.gc, s);
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

static inline void copy_z5(Falm::Matrix<Falm::REAL> &v, Falm::CPM &cpm) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    copy_z5(v, pdm, cpm.gc);
    Falm::falmWaitStream();
}

#endif