#include "../src/rmcp/bladeHandler.h"
#include "../src/falmath.h"

using namespace Falm;

BladeHandler blades;

const double xorigin = -2.0;
const double xlength = 10.0;
const double yorigin = -2.5;
const double ylength =  5.0;
const double zorigin = -2.5;
const double zlength =  5.0;
const int    imax    = 200;
const int    jmax    = 100;
const int    kmax    = 100;

double *cd, *cdd, *cl, *cld;
double *x, *xd, *y, *yd, *z, *zd;

void init() {
    blades.alloc("bladeProperties.json");
    // cd = (double*)malloc(sizeof(double)*imax*jmax*kmax);
    // cl = (double*)malloc(sizeof(double)*imax*jmax*kmax);
    // cudaMalloc((void**)&cdd, sizeof(double)*imax*jmax*kmax);
    // cudaMalloc((void**)&cld, sizeof(double)*imax*jmax*kmax);
    // x = (double*)malloc(sizeof(double)*imax);
    // y = (double*)malloc(sizeof(double)*jmax);
    // z = (double*)malloc(sizeof(double)*kmax);
    // cudaMalloc((void**)&xd, sizeof(double)*imax);
    // cudaMalloc((void**)&yd, sizeof(double)*jmax);
    // cudaMalloc((void**)&zd, sizeof(double)*kmax);

    // cudaMemset(cdd, 0, sizeof(double)*imax*jmax*kmax);
    // cudaMemset(cld, 0, sizeof(double)*imax*jmax*kmax);

    // for (int i = 0; i < imax; i ++) {
    //     x[i] = (i + 0.5)*(xlength/imax);
    // }
    // for (int j = 0; j < jmax; j ++) {
    //     y[j] = (j + 0.5)*(ylength/jmax);
    // }
    // for (int k = 0; k < kmax; k ++) {
    //     z[k] = (k + 0.5)*(zlength/kmax);
    // }
}

void finalize() {
    blades.release();
    // free(cd);
    // free(cl);
    // cudaFree(cdd);
    // cudaFree(cld);
    // free(x);
    // free(y);
    // free(z);
    // cudaFree(xd);
    // cudaFree(yd);
    // cudaFree(zd);
}

int main() {
    init();
    double chord, phi, twist, attack, cl, cd;

    const double r = 0.9;
    phi = atan(1./(4.*r))*180/Pi;

    blades.get_airfoil_params(r, phi, chord, twist, cl, cd);
    attack = phi - twist;
    printf("%lf %lf %lf %lf %lf %lf\n", twist, phi, attack, chord, cl, cd);
    finalize();
}