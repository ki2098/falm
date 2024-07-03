#include <stdio.h>
#include <string>
#include <fstream>

using namespace std;

size_t imax, jmax, kmax, nmax, gc, step, type;
double tt;

size_t id(size_t i, size_t j, size_t k, size_t n) {
    return n*imax*jmax*kmax + k*imax*jmax + j*imax + i;
}

int main() {
    string prefix = "data/uvwp2";
    string fname = prefix + "_0000015000";
    string cvname = prefix + ".cv";
    FILE *file = fopen(fname.c_str(), "rb");
    
    fread(&imax, sizeof(size_t), 1, file);
    fread(&jmax, sizeof(size_t), 1, file);
    fread(&kmax, sizeof(size_t), 1, file);
    fread(&nmax, sizeof(size_t), 1, file);
    fread(&gc  , sizeof(size_t), 1, file);
    fread(&step, sizeof(size_t), 1, file);
    fread(&tt  , sizeof(double), 1, file);
    fread(&type, sizeof(size_t), 1, file);

    double *data = (double*)malloc(sizeof(double)*imax*jmax*kmax*nmax);
    fread(data, sizeof(double), imax*jmax*kmax*nmax, file);

    fclose(file);

    ifstream ifs(cvname);
    double *x = (double*)malloc(sizeof(double)*imax);
    double *y = (double*)malloc(sizeof(double)*jmax);
    double *z = (double*)malloc(sizeof(double)*kmax);
    double dummy;
    ifs >> dummy >> dummy >> dummy >> dummy;
    for (size_t i = 0; i < imax; i ++) {
        ifs >> x[i] >> dummy;
    }
    for (size_t j = 0; j < jmax; j ++) {
        ifs >> y[j] >> dummy;
    }
    for (size_t k = 0; k < kmax; k ++) {
        ifs >> z[k] >> dummy;
    }
    ifs.close();

    file = fopen((fname+".csv").c_str(), "w");

    fprintf(file, "x,y,z,u,v,w,p\n");
    for (size_t k = 0; k < kmax; k ++) {
    for (size_t j = 0; j < jmax; j ++) {
    for (size_t i = 0; i < imax; i ++) {
        fprintf(file, "%e,%e,%e,%e,%e,%e,%e\n", x[i], y[j], z[k], data[id(i,j,k,0)], data[id(i,j,k,1)], data[id(i,j,k,2)], data[id(i,j,k,3)]);
    }}}
    printf("%lu %lu %lu\n", imax, jmax, kmax);

    free(data);
    free(x);
    free(y);
    free(z);
    fclose(file);
}