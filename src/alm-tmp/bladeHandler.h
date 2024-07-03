#ifndef FALM_ALM_BLADEHANDLER_H
#define FALM_ALM_BLADEHANDLER_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "../typedef.h"

namespace Falm {

class BladeHandler {
public:
    static void buildAP(std::string inputfname, std::string outputfname, int apcount, int apperblade, int bladeperturbine, double radius) {
        std::ifstream ifs(inputfname);
        auto bdjson = json::parse(ifs);
        auto aflist = bdjson["airfoils"];
        auto atlist = bdjson["attacks"];

        std::vector<double> attack(atlist.size());
        std::vector<double> rr(aflist.size());
        std::vector<double> chord(aflist.size());
        std::vector<double> twist(aflist.size());
        std::vector<std::vector<double>> cl(aflist.size(), std::vector<double>(atlist.size()));
        std::vector<std::vector<double>> cd(aflist.size(), std::vector<double>(atlist.size()));

        for (int i = 0; i < aflist.size(); i ++) {
            auto af = aflist[i];
            rr[i] = af["r/R"].get<double>();
            chord[i] = af["chord/R"].get<double>();
            twist[i] = af["twist"].get<double>();
            auto cllist = af["Cl"];
            auto cdlist = af["Cd"];
            for (int j = 0; j < atlist.size(); j ++) {
                cl[i][j] = cllist[j].get<double>();
                cd[i][j] = cdlist[j].get<double>();
            }
        }

        for (int j = 0; j < atlist.size(); j ++) {
            attack[j] = atlist[j].get<double>();
        }

        auto apjs = json::array();
        double dr = radius/apperblade;
        int apperturbine = apperblade*bladeperturbine;
        int imax = rr.size()-1;
        for (int apid = 0; apid < apcount; apid ++) {
            double apr = (apid%apperblade + 1)*dr;
            double apchord;
            double aptwist;
            int aptid = apid/apperturbine;
            int apbid = (apid%apperturbine)/apperblade;
            std::vector<double> apcd(attack.size());
            std::vector<double> apcl(attack.size());
            if (apr < rr[0]) {
                apchord = chord[0];
                aptwist = twist[0];
                apcd = cd[0];
                apcl = cl[0];
            } else if (apr >= rr[imax]) {
                apchord = chord[imax];
                aptwist = twist[imax];
                apcd = cd[imax];
                apcl = cl[imax];
            } else {
                int i;
                double p;
                for (i = 0; i < imax; i ++) {
                    if (rr[i] <= apr && rr[i+1] > apr) {
                        p = (apr - rr[i])/(rr[i+1] - rr[i]);
                        break;
                    }
                }
                apchord = (1. - p)*chord[i] + p*chord[i+1];
                aptwist = (1. - p)*twist[i] + p*twist[i+1];
                for (int j = 0; j < attack.size(); j ++) {
                    apcl[j] = (1. - p)*cl[i][j] + p*cl[i+1][j];
                    apcd[j] = (1. - p)*cd[i][j] + p*cd[i+1][j];
                }
            }
            json apj;
            apj["id"] = apid;
            apj["turbineId"] = aptid;
            apj["bladeId"] = apbid;
            apj["chord"] = apchord;
            apj["twist"] = aptwist;
            apj["r"] = apr;
            auto apattack = json::array();
            for (int j = 0; j < attack.size(); j ++) {
                json tmp;
                tmp["attack"] = attack[j];
                tmp["Cl"] = apcl[j];
                tmp["Cd"] = apcd[j];
                apattack.push_back(tmp);
            }
            apj["airfoilProperty"] = apattack;
            apjs.push_back(apj);
        }

        std::ofstream ofs(outputfname);
        ofs << apjs.dump(2);
        ofs.close();
    }
};

}

#endif
