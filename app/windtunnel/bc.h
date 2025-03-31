#pragma once

#include "bcdevcall.h"
#include "../../src/falm.h"

static void ubc(Falm::Matrix<Falm::Real> &u, Falm::Matrix<Falm::Real> &uprev, Falm::Matrix<Falm::Real> &x, Falm::Real u_inlet, Falm::Real dt, Falm::CPM &cpm, Falm::Stream *stream = nullptr) {
    Falm::Region &pdm = cpm.pdm_list[cpm.rank];
    Falm::Int &gc = cpm.gc;
    for (Falm::Int fid = 0; fid < Falm::CPM::NFACE; fid ++) {
        if (!cpm.validNeighbour(fid)) {
            Falm::Stream s = (stream)? stream[fid] : 0;
            if (fid == Falm::CPM::XMINUS) {
                ubc_xminus(u, u_inlet, pdm, gc, s);
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