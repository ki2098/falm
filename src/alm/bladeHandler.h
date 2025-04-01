#ifndef FALM_ALM_BLADEHANDLER_H
#define FALM_ALM_BLADEHANDLER_H

#include <fstream>
#include <string>
#include <vector>
#include "../typedef.h"

namespace Falm {

namespace Alm {

class BladeHandler {

public:
    static void buildAP(const Json &bladejson, std::string apfilepath, int turbinecount, int bladeperturbine, int apperblade, double radius, double hubradius=0.0) {
        int apcount = turbinecount*bladeperturbine*apperblade;
        auto aflist = bladejson["airfoils"];
        auto atlist = bladejson["attacks"];
        size_t imax = aflist.size();
        size_t jmax = atlist.size();
        
        std::vector<double> attack(jmax);
        std::vector<double> rr(imax);
        std::vector<double> chord(imax);
        std::vector<double> twist(imax);
        std::vector<std::vector<double>> cl(imax);
        std::vector<std::vector<double>> cd(jmax);
        for (size_t i = 0; i < imax; i ++) {
            auto af = aflist[i];
            rr[i] = af["r/R"].get<double>();
            chord[i] = af["chord/R"].get<double>();
            twist[i] = af["twist[deg]"].get<double>();
            auto cllist = af["Cl"];
            auto cdlist = af["Cd"];
            cl[i].resize(jmax);
            cd[i].resize(jmax);
            for (size_t j = 0; j < jmax; j ++) {
                cl[i][j] = cllist[j].get<double>();
                cd[i][j] = cdlist[j].get<double>();
            }
        }
        for (size_t j = 0; j < jmax; j ++) {
            attack[j] = atlist[j].get<double>();
        }

        auto aparrayjson = OrderedJson::array();
        double dr = (radius - hubradius)/apperblade;
        // int apperturbine = apperblade*bladeperturbine;
        for (int apid = 0; apid < apcount; apid ++) {
            double apr = (apid%apperblade + 0.5)*dr + hubradius;
            double apchord, aptwist;
            // int aptid = apid/apperturbine;
            // int apbid = (apid%apperturbine)/apperblade;
            std::vector<double> apcl;
            std::vector<double> apcd;
            if (apr < rr[0]) {
                apchord = chord[0];
                aptwist = twist[0];
                apcl = cl[0];
                apcd = cd[0];
            } else if (apr >= rr[imax-1]) {
                apchord = chord[imax-1];
                aptwist = twist[imax-1];
                apcl = cl[imax-1];
                apcd = cd[imax-1];
            } else {
                size_t i;
                double p;
                for (i = 0; i < imax-1; i ++) {
                    if (rr[i] <= apr && rr[i+1] > apr) {
                        p = (apr - rr[i])/(rr[i+1] - rr[i]);
                        break;
                    }
                }
                apchord = (1. - p)*chord[i] + p*chord[i+1];
                aptwist = (1. - p)*twist[i] + p*twist[i+1];
                apcl.resize(jmax);
                apcd.resize(jmax);
                for (size_t j = 0; j < jmax; j ++) {
                    apcl[j] = (1. - p)*cl[i][j] + p*cl[i+1][j];
                    apcd[j] = (1. - p)*cd[i][j] + p*cd[i+1][j];
                }
            }
            OrderedJson apjson;
            apjson["id"] = apid;
            // apjson["turbineId"] = aptid;
            // apjson["bladeId"] = apbid;
            apjson["r"] = apr;
            apjson["chord"] = apchord;
            apjson["twist[deg]"] = aptwist;
            apjson["Cl"] = apcl;
            apjson["Cd"] = apcd;
            aparrayjson.push_back(apjson);
        }
        OrderedJson tmp;
        tmp["aps"] = aparrayjson;
        tmp["attacks"] = attack;
        // tmp["dr"] = dr;

        std::ofstream apfile(apfilepath);
        apfile << tmp.dump(2);
        apfile.close();
    }

};

}

}

#endif