#ifndef MESHER_MESHER_H
#define MESHER_MESHER_H

#include <string>
#include <fstream>
#include <stdio.h>
#include "../nlohmann/json.hpp"

class Mesher {

public:
    static void build_mesh(std::string workdir, std::string cvcenter, std::string meshpath, std::string outputfile, int gc) {
        int imax, jmax, kmax;
        double *x, *y, *z, *hx, *hy, *hz;
        std::ifstream mjfile(workdir + "/" + meshpath + "/mesh.json");
        auto mjson = nlohmann::json::parse(mjfile);
        std::string coordcenter = mjson["coordinatePoint"];
        if (coordcenter == "node") {
            if (cvcenter == "innerNode") {
                gc -= 1;
                std::ifstream txtfile;
                std::string line;

                txtfile.open(workdir + "/" + meshpath + "/x.txt");
                std::getline(txtfile, line);
                imax = std::stoi(line) + 2 * gc;
                x = (double*)malloc(sizeof(double)*imax);
                for (int i = gc; i < imax - gc; i ++) {
                    std::getline(txtfile, line);
                    x[i] = std::stod(line);
                }
                txtfile.close();

                txtfile.open(workdir + "/" + meshpath + "/y.txt");
                std::getline(txtfile, line);
                jmax = std::stoi(line) + 2 * gc;
                y = (double*)malloc(sizeof(double)*jmax);
                for (int j = gc; j < jmax - gc; j ++) {
                    std::getline(txtfile, line);
                    y[j] = std::stod(line);
                }
                txtfile.close();

                txtfile.open(workdir + "/" + meshpath + "/z.txt");
                std::getline(txtfile, line);
                kmax = std::stoi(line) + 2 * gc;
                z = (double*)malloc(sizeof(double)*kmax);
                for (int k = gc; k < kmax - gc; k ++) {
                    std::getline(txtfile, line);
                    z[k] = std::stod(line);
                }

                hx = (double*)malloc(sizeof(double)*imax);
                for (int i = gc + 1; i < imax - gc - 1; i ++) {
                    hx[i] = 0.5 * (x[i + 1] - x[i - 1]);
                }
                for (int i = gc; i >= 0; i --) {
                    hx[i] = 2 * hx[i + 1] - hx[i + 2];
                }
                for (int i = imax - gc - 1; i < imax; i ++) {
                    hx[i] = 2 * hx[i - 1] - hx[i - 2];
                }

                hy = (double*)malloc(sizeof(double)*jmax);
                for (int j = gc + 1; j < jmax - gc - 1; j ++) {
                    hy[j] = 0.5 * (y[j + 1] - y[j - 1]);
                }
                for (int j = gc; j >= 0; j --) {
                    hy[j] = 2 * hy[j + 1] - hy[j + 2];
                }
                for (int j = jmax - gc - 1; j < jmax; j ++) {
                    hy[j] = 2 * hy[j - 1] - hy[j - 2];
                }

                hz = (double*)malloc(sizeof(double)*kmax);
                for (int k = gc + 1; k < kmax - gc - 1; k ++) {
                    hz[k] = 0.5 * (z[k + 1] - z[k - 1]);
                }
                for (int k = gc; k >= 0; k --) {
                    hz[k] = 2 * hz[k + 1] - hz[k + 2];
                }
                for (int k = kmax - gc - 1; k < kmax; k ++) {
                    hz[k] = 2 * hz[k - 1] - hz[k - 1];
                }

                for (int i = gc - 1; i >= 0; i --) {
                    x[i] = x[i + 1] - 0.5 * (hx[i] + hx[i + 1]);
                }
                for (int i = imax - gc; i < imax; i ++) {
                    x[i] = x[i - 1] + 0.5 * (hx[i] + hx[i - 1]);
                }

                for (int j = gc - 1; j >= 0; j --) {
                    y[j] = y[j + 1] - 0.5 * (hy[j] + hy[j + 1]);
                }
                for (int j = jmax - gc; j < jmax; j ++) {
                    y[j] = y[j - 1] + 0.5 * (hy[j] + hy[j - 1]);
                }

                for (int k = gc - 1; k >= 0; k --) {
                    z[k] = z[k + 1] - 0.5 * (hz[k] + hz[k + 1]);
                }
                for (int k = kmax - gc; k < kmax; k ++) {
                    z[k] = z[k - 1] + 0.5 * (hz[k] + hz[k - 1]);
                }
            }
        }

        std::string output_path = workdir + "/" + outputfile;
        FILE *ofile = fopen(output_path.c_str(), "w");
        if (ofile) {
            fprintf(ofile, "%d %d %d\n", imax, jmax, kmax);
            for (int i = 0; i < imax; i ++) {
                fprintf(ofile, "\t%e\t%e\n", x[i], hx[i]);
            }
            for (int j = 0; j < jmax; j ++) {
                fprintf(ofile, "\t%e\t%e\n", y[j], hy[j]);
            }
            for (int k = 0; k < kmax; k ++) {
                fprintf(ofile, "\t%e\t%e\n", z[k], hz[k]);
            }
        }
        fclose(ofile);
        free(x); free(y); free(z);
        free(hx); free(hy); free(hz);

        mjfile.close();
    }

};

#endif