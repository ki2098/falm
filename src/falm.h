#ifndef FALM_FALM_H
#define FALM_FALM_H

#include "FalmCFD.h"
#include "FalmEq.h"
#include "FalmTime.h"
#include "falmath.h"
#include "mesher/mesher.hpp"
#include "rmcp/alm.h"
#include "vcdm/VCDM.h"

namespace Falm {

class FalmControl {
public:
    REAL dt;
    REAL startTime;
    INT  maxIt;
    INT  timeAvgStartIt;
    INT  timeAvgStopIt;
    INT  outputStartIt;
    INT  outputStopIt;
    INT  outputIntervalIt;
};

class FalmCore {
public:
    json             params;
    FalmControl falmControl;
    FalmCFD         falmCfd;
    FalmEq           falmEq;
    FalmTime       falmTime;

    static void parse(const json &jobj, FalmEq &eq) {
        auto lsprm = jobj["solver"]["linearSolver"];
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
        eq.init(tmp.type, tmp.maxit, tmp.tol, tmp.relax_factor, tmp.pc_type, tmp.pc_maxit, tmp.pc_relax_factor);
    }

    static void parse(const json &jobj, FalmCFD &cfd) {
        auto cfdprm = jobj["solver"]["cfd"];
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
        cfd.init(tmp.Re, tmp.AdvScheme, tmp.SGSModel, tmp.CSmagorinsky);
    }

    static void parse(const json &jobj, FalmControl &control) {
        auto runprm = jobj["runtime"];
    }
};

}

#endif