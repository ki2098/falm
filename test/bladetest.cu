#include "../src/rmcp/bladeHandler.h"
#include "../src/falmath.h"

Falm::BladeHandler bladeHandler;

__global__ void get(Falm::BHFrame *bh, double r, double phi, double *param) {
    double chord, twist, cl, cd;
    bh->get_airfoil_params(r, phi, chord, twist, cl, cd);
    param[0] = chord;
    param[1] = twist;
    param[2] = cl;
    param[3] = cd;
}

int main() {
    bladeHandler.alloc("bladeProperties.json");

    const double r = 0.25;
    const double phi = atan(1./4.)*180/Falm::Pi;

    double *params, *params_dev;
    params = (double*)malloc(sizeof(double)*4);
    cudaMalloc((void**)&params_dev, sizeof(double)*4);

    get<<<1,1>>>(bladeHandler.devptr, r, phi, params_dev);

    cudaMemcpy(params, params_dev, sizeof(double)*4, cudaMemcpyDeviceToHost);

    double chord = params[0];
    double twist = params[1];
    double cl    = params[2];
    double cd    = params[3];
    double attack = phi - twist;

    printf("chord=%lf,twist=%lf,phi=%lf,attack=%lf,cl=%lf,cd=%lf\n", chord, twist, phi, attack, cl, cd);

    bladeHandler.release();
}