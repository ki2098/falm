#ifndef FALM_APP_LID3D_BC_H
#define FALM_APP_LID3D_BC_H

#include "bcdev.h"
#include "../../src/CPM.h"

namespace LID3D {

static void ubc(Falm::Matrix<Falm::Real> &u, Falm::CPM &cpm, Falm::Stream *stream = nullptr) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    Falm::Int &gc = cpm.gc;
    for (Falm::Int fid = 0; fid < Falm::CPM::NFACE; fid ++) {
        if (!cpm.validNeighbour(fid)) {
            Falm::Stream s = (stream)? stream[fid] : 0;
            if (fid == Falm::CPM::XMINUS) {
                ubc_xminus(u, pdm, gc, s);
            } else if (fid == Falm::CPM::XPLUS) {
                ubc_xplus (u, pdm, gc, s);
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
        for (Falm::Int fid = 0; fid < Falm::CPM::NFACE; fid ++) {
            if (!cpm.validNeighbour(fid)) {
                Falm::falmWaitStream(stream[fid]);
            }
        }
    } else {
        Falm::falmWaitStream();
    }
}

static void pbc(Falm::Matrix<Falm::Real> &p, Falm::CPM &cpm, Falm::Stream *stream = nullptr) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    Falm::Int &gc = cpm.gc;
    for (Falm::Int fid = 0; fid < Falm::CPM::NFACE; fid ++) {
        if (!cpm.validNeighbour(fid)) {
            Falm::Stream s = (stream)? stream[fid] : 0;
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
        for (Falm::Int fid = 0; fid < Falm::CPM::NFACE; fid ++) {
            if (!cpm.validNeighbour(fid)) {
                Falm::falmWaitStream(stream[fid]);
            }
        }
    } else {
        Falm::falmWaitStream();
    }
}

static void uubc(Falm::Matrix<Falm::Real> &uu, Falm::CPM &cpm, Falm::Stream *stream = nullptr) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    Falm::Int &gc = cpm.gc;
    for (Falm::Int fid = 0; fid < Falm::CPM::NFACE; fid ++) {
        if (!cpm.validNeighbour(fid)) {
            Falm::Stream s = (stream)? stream[fid] : 0;
            if (fid == Falm::CPM::XMINUS) {
                uubc_xminus(uu, pdm, gc, s);
            } else if (fid == Falm::CPM::XPLUS) {
                uubc_xplus (uu, pdm, gc, s);
            } else if (fid == Falm::CPM::YMINUS) {
                uubc_yminus(uu, pdm, gc, s);
            } else if (fid == Falm::CPM::YPLUS) {
                uubc_yplus (uu, pdm, gc, s);
            } else if (fid == Falm::CPM::ZMINUS) {
                uubc_zminus(uu, pdm, gc, s);
            } else if (fid == Falm::CPM::ZPLUS) {
                uubc_zplus (uu, pdm, gc, s);
            }
        }
    }
    if (stream) {
        for (Falm::Int fid = 0; fid < Falm::CPM::NFACE; fid ++) {
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