#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>
#include "../../src/nlohmann/json.hpp"

using namespace std;

int cx, cy, cz, gc;
double *x, *y, *z, *xx, *yy, *zz;
double *J, *gx, *gy, gz;

double *u, *v, *w, *p, *nut;
double *up, *vp, *wp;
double *rhs;

int getId(int i, int j, int k, int ci, int cj, int ck) {
    return i*cj*ck + j*ck + k;
}

void prepareMesh(string path, int guideCell) {
    gc = guideCell;

    ifstream xFile(path + "/x.txt");
    int xBufSize;
    xFile >> xBufSize;
    double *xBuf = new double[xBufSize];
    for (int i = 0; i < xBufSize; i ++) {
        xFile >> xBuf[i];
    }
    cx = xBufSize - 1 + 2*gc;
    x = new double[cx];
    xx = new double[cx];
    for (int i = gc; i < cx - gc; i ++) {
        double dx = xBuf[i - gc + 1] - xBuf[i - gc];
        xx[i] = 1./dx;
        x[i] = xBuf[i - gc] + 0.5*dx;
    }
    for (int i = gc - 1; i >= 0; i --) {
        double dx = 2./xx[i + 1] - 1./xx[i + 2];
        xx[i] = 1./dx;
        x[i] = x[i + 1] - 0.5*(dx + 1./xx[i + 1]);
    }
    for (int i = cx - gc; i < cx; i ++) {
        double dx = 2./xx[i - 1] - 1./xx[i - 2];
        xx[i] = 1./dx;
        x[i] = x[i - 1] + 0.5*(dx + 1./xx[i - 1]);
    }
    delete[] xBuf;
    xFile.close();

    ifstream yFile(path + "/y.txt");
    int yBufSize;
    yFile >> yBufSize;
    double *yBuf = new double[yBufSize];
    for (int j = 0; j < yBufSize; j ++) {
        yFile >> yBuf[j];
    }
    cy = yBufSize - 1 + 2*gc;
    y = new double[cy];
    yy = new double[cy];
    for (int j = gc; j < cy - gc; j ++) {
        double dy = yBuf[j - gc + 1] - yBuf[j - gc];
        yy[j] = 1./dy;
        y[j] = yBuf[j - gc] + 0.5*dy;
    }
    for (int j = gc - 1; j >= 0; j --) {
        double dy = 2./yy[j + 1] - 1./yy[j + 2];
        yy[j] = 1./dy;
        y[j] = y[j + 1] - 0.5*(dy + 1./yy[j + 1]);
    }
    for (int j = cy - gc; j < cy; j ++) {
        double dy = 2./yy[j - 1] - 1./yy[j - 2];
        yy[j] = 1./dy;
        y[j] = y[j - 1] + 0.5*(dy + 1./yy[j - 1]);
    }
    delete[] yBuf;
    yFile.close();

    ifstream zFile(path + "/z.txt");
    int zBufSize;
    zFile >> zBufSize;
    double *zBuf = new double[zBufSize];
    for (int k = 0; k < zBufSize; k ++) {
        zFile >> zBuf[k];
    }
    cz = zBufSize - 1 + 2*gc;
    z = new double[cz];
    zz = new double[cz];
    for (int k = gc; k < cz - gc; k ++) {
        double dz = zBuf[k - gc + 1] - zBuf[k - gc];
        zz[k] = 1./dz;
        z[k] = zBuf[k - gc] + 0.5*dz;
    }
    for (int k = gc - 1; k >= 0; k --) {
        double dz = 2./zz[k + 1] - 1./zz[k + 2];
        zz[k] = 1./dz;
        z[k] = z[k + 1] - 0.5*(dz + 1./zz[k + 1]);
    }
    for (int k = cz - gc; k < cz; k ++) {
        double dz = 2./zz[k - 1] - 1./zz[k - 2];
        zz[k] = 1./dz;
        z[k] = z[k - 1] + 0.5*(dz + 1./zz[k - 1]);
    }
    delete[] zBuf;
    zFile.close();
}

void finalize() {
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] xx;
    delete[] yy;
    delete[] zz;
}

void calcInterVel(
    double *u, double *v, double *w,
    double *ua, double *va, double *wa,
    double *nut,
    double *rhs,
    double *xx, double *yy, double *zz,
    double *J,
    const double Re,
    const double dt,
    const double cx,
    const double cy,
    const double cz,
    const double gc
) {
    for (int i = gc; i < cx - gc; i ++) {
        for (int j = gc; j < cy - gc; j ++) {
            for (int k = gc; k < cz - gc; k ++) {
                int idCC = getId(i, j, k, cx, cy, cz);
                int idE1 = getId(i + 1, j, k, cx, cy, cz);
                int idE2 = getId(i + 2, j, k, cx, cy, cz);
                int idW1 = getId(i - 1, j, k, cx, cy, cz);
                int idW2 = getId(i - 2, j, k, cx, cy, cz);
                int idN1 = getId(i, j + 1, k, cx, cy, cz);
                int idN2 = getId(i, j + 2, k, cx, cy, cz);
                int idS1 = getId(i, j - 1, k, cx, cy, cz);
                int idS2 = getId(i, j - 2, k, cx, cy, cz);
                int idT1 = getId(i, j, k + 1, cx, cy, cz);
                int idT2 = getId(i, j, k + 2, cx, cy, cz);
                int idB1 = getId(i, j, k - 1, cx, cy, cz);
                int idB2 = getId(i, j, k - 2, cx, cy, cz);

                double uCC = u[idCC];
                double vCC = v[idCC];
                double wCC = w[idCC];
                double UCC = uCC*xx[idCC];
                double VCC = vCC*yy[idCC];
                double WCC = wCC*zz[idCC];

                double uE1 = u[idE1];
                double uE2 = u[idE2];
                double uW1 = u[idW1];
                double uW2 = u[idW2];
                double uN1 = u[idN1];
                double uN2 = u[idN2];
                double uS1 = u[idS1];
                double uS2 = u[idS2];
                double uT1 = u[idT1];
                double uT2 = u[idT2];
            }
        }
    }
}

int main() {
    prepareMesh("mesh/20", 2);

    cout << cx << " " << cy << " " << cz << " " << gc << endl;
    
    

    finalize();
}